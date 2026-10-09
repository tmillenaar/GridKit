use ndarray::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use rayon::ThreadPool;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "parallel")]
use std::sync::{Arc, RwLock};

/// Minimum number of `(x, y)` pairs in a batched query before the parallel
/// helpers hand the work to rayon. Below this the scheduling overhead tends to
/// outweigh the win, and the per-point queries (`cell_at_point`, `Tile::corners`,
/// small neighbour lookups) stay single-threaded.
#[cfg(feature = "parallel")]
const PARALLEL_MIN_POINTS: usize = 4096;

/// Worker count used when neither an explicit override, the environment, nor a
/// detected core count is available.
#[cfg(feature = "parallel")]
const DEFAULT_PARALLEL_THREADS: usize = 4;

/// Explicit override set by [`set_num_threads`]; `0` means "not set".
#[cfg(feature = "parallel")]
static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

/// GridKit's own rayon pool, built lazily on the first parallel query.
///
/// Using a private pool (rather than rayon's global one) is what lets the
/// default and the overrides below be honoured regardless of what the host
/// process does with rayon. The lock and reference counting allow the pool to
/// be replaced safely after a thread-count change while active queries finish
/// on the old pool.
#[cfg(feature = "parallel")]
static POOL: RwLock<Option<Arc<ThreadPool>>> = RwLock::new(None);

/// Thread count from `GRIDKIT_NUM_THREADS`, then `RAYON_NUM_THREADS`.
#[cfg(feature = "parallel")]
fn env_threads() -> Option<usize> {
    ["GRIDKIT_NUM_THREADS", "RAYON_NUM_THREADS"]
        .iter()
        .find_map(|key| {
            std::env::var(key)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|n| *n > 0)
        })
}

/// The CPUs this process is allowed to run on, from `/proc/self/status`'s
/// `Cpus_allowed_list` (Linux only).
#[cfg(feature = "parallel")]
fn allowed_cpus() -> Option<Vec<u32>> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?
        .trim();
    let mut cpus = Vec::new();
    for part in list.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((start, end)) => {
                let start: u32 = start.trim().parse().ok()?;
                let end: u32 = end.trim().parse().ok()?;
                cpus.extend(start..=end);
            }
            None => cpus.push(part.parse().ok()?),
        }
    }
    Some(cpus)
}

/// Best-effort physical core count, respecting the process's CPU affinity.
///
/// Counts distinct `(physical_package_id, core_id)` pairs among the allowed
/// CPUs. Returns `None` when the topology is unavailable (e.g. non-Linux), so
/// the caller can fall back to the logical count.
#[cfg(feature = "parallel")]
fn physical_core_count() -> Option<usize> {
    let mut cores = std::collections::HashSet::new();
    for cpu in allowed_cpus()? {
        let base = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
        let (Ok(core), Ok(package)) = (
            std::fs::read_to_string(format!("{base}/core_id")),
            std::fs::read_to_string(format!("{base}/physical_package_id")),
        ) else {
            continue;
        };
        cores.insert((package.trim().to_string(), core.trim().to_string()));
    }
    (!cores.is_empty()).then_some(cores.len())
}

/// The number of rayon workers the batched queries will use.
///
/// Defaults to the number of physical cores: SMT siblings add little for this
/// memory-bound work and oversubscribing them can hurt. Falls back to the
/// logical core count, then to [`DEFAULT_PARALLEL_THREADS`], when the topology
/// cannot be read (e.g. non-Linux or a restricted `/proc`).
#[cfg(feature = "parallel")]
pub fn get_num_threads() -> usize {
    match NUM_THREADS.load(Ordering::Relaxed) {
        0 => env_threads().unwrap_or_else(|| {
            physical_core_count()
                .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
                .unwrap_or(DEFAULT_PARALLEL_THREADS)
        }),
        n => n,
    }
}

/// Override the number of worker threads used by the parallel grid queries.
///
/// Pass `threads >= 1`, or `None` to fall back to the default (physical cores,
/// then logical cores, then [`DEFAULT_PARALLEL_THREADS`]). The pool is built
/// lazily on the first parallel query. If it has already been built, it is
/// replaced with a new pool using the requested number of workers. This waits
/// for neither new nor active queries; active queries finish on the old pool
/// while new queries use the replacement.
/// `GRIDKIT_NUM_THREADS` and `RAYON_NUM_THREADS` provide the same override from
/// the environment.
#[cfg(feature = "parallel")]
pub fn set_num_threads(threads: Option<usize>) -> Result<(), String> {
    if threads == Some(0) {
        return Err("number of threads must be at least 1 (or None for the default)".to_string());
    }
    let mut pool = POOL
        .write()
        .map_err(|_| "the gridkit rayon pool lock is poisoned".to_string())?;
    NUM_THREADS.store(threads.unwrap_or(0), Ordering::Relaxed);
    if pool.is_some() {
        *pool = Some(Arc::new(build_pool()));
    }
    Ok(())
}

#[cfg(feature = "parallel")]
fn build_pool() -> ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(get_num_threads())
        .thread_name(|i| format!("gridkit-{i}"))
        .build()
        .expect("failed to build the gridkit rayon thread pool")
}

#[cfg(feature = "parallel")]
fn with_pool<F, R>(f: F) -> R
where
    F: FnOnce(&ThreadPool) -> R,
{
    {
        let mut pool = POOL
            .write()
            .expect("the gridkit rayon pool lock is poisoned");
        if pool.is_none() {
            *pool = Some(Arc::new(build_pool()));
        }
    }
    let pool = POOL
        .read()
        .expect("the gridkit rayon pool lock is poisoned");
    let pool = Arc::clone(pool.as_ref().expect("gridkit rayon pool was not initialized"));
    f(&pool)
}

/// Whether a batch of `n` `(x, y)` pairs should be spread over the pool.
///
/// Nested parallelism is deliberately avoided: the batched queries call each
/// other (e.g. `TriGrid::cells_near_point` calls `cell_at_points` and
/// `cell_corners`), so once we are already executing on a rayon worker the
/// outer query has claimed the pool and the inner one runs serially.
#[cfg(feature = "parallel")]
fn should_parallelize(n: usize) -> bool {
    n >= PARALLEL_MIN_POINTS && get_num_threads() > 1 && rayon::current_thread_index().is_none()
}

/// Run `f` over the row chunks of `raveled` in parallel. Called by the
/// `*_parallel` helpers once [`should_parallelize`] has agreed the batch is
/// worth it.
#[cfg(feature = "parallel")]
fn map_rows_2<A, B, F>(raveled: ArrayView2<A>, f: &F) -> Array2<B>
where
    A: Clone + Sync,
    B: Send + Clone,
    F: Fn(ArrayView2<A>) -> Array2<B> + Sync,
{
    let chunk = raveled.nrows().div_ceil(get_num_threads());
    let parts: Vec<Array2<B>> = with_pool(|pool| pool.install(|| {
        raveled
            .axis_chunks_iter(Axis(0), chunk)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|chunk| f(chunk))
            .collect()
    }));
    let views: Vec<ArrayView2<B>> = parts.iter().map(|part| part.view()).collect();
    concatenate(Axis(0), &views).expect("chunked rows concatenate along axis 0")
}

/// `map_rows_2` for `(N, 2) -> (N, K, 2)` (i.e. the fanout helpers).
#[cfg(feature = "parallel")]
fn map_rows_3<A, B, F>(raveled: ArrayView2<A>, f: &F) -> Array3<B>
where
    A: Clone + Sync,
    B: Send + Clone,
    F: Fn(ArrayView2<A>) -> Array3<B> + Sync,
{
    let chunk = raveled.nrows().div_ceil(get_num_threads());
    let parts: Vec<Array3<B>> = with_pool(|pool| pool.install(|| {
        raveled
            .axis_chunks_iter(Axis(0), chunk)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|chunk| f(chunk))
            .collect()
    }));
    let views: Vec<ArrayView3<B>> = parts.iter().map(|part| part.view()).collect();
    concatenate(Axis(0), &views).expect("chunked rows concatenate along axis 0")
}

pub fn iseven(val: i64) -> bool {
    val % 2 == 0
}

pub fn modulus(val: f64, modulus: f64) -> f64 {
    ((val % modulus) + modulus) % modulus
}

pub fn normalize_offset(offset: [f64; 2], dx: f64, dy: f64) -> [f64; 2] {
    // Note: In Rust, % is the remainder.
    //       In Python, % is the modulus.
    //       Here we want the modulus, see https://stackoverflow.com/q/31210357
    let offset_x = modulus(offset[0], dx);
    let offset_y = modulus(offset[1], dy);
    [offset_x, offset_y]
}

/// Relative tolerance used as `numpy.is_close` default.
pub const NUMERIC_RTOL: f64 = 1e-5;
/// Absolute tolerance used as `numpy.is_close` default.
pub const NUMERIC_ATOL: f64 = 1e-8;

/// Compare two floats the way `numpy.is_close` does: `|a - b| <= atol + rtol * |b|`.
pub fn is_close_with_atol_and_rtol(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

pub fn is_close_with_rtol(a: f64, b: f64, rtol: f64) -> bool {
    is_close_with_atol_and_rtol(a, b, rtol, NUMERIC_ATOL)
}

pub fn is_close_with_atol(a: f64, b: f64, atol: f64) -> bool {
    is_close_with_atol_and_rtol(a, b, NUMERIC_RTOL, atol)
}

pub fn is_close(a: f64, b: f64) -> bool {
    is_close_with_atol_and_rtol(a, b, NUMERIC_RTOL, NUMERIC_ATOL)
}

pub fn rotation_matrix_from_angle(angle_deg: f64) -> Array2<f64> {
    let angle_rad = angle_deg.to_radians();
    let cos_angle = angle_rad.cos();
    let sin_angle = angle_rad.sin();
    array![[cos_angle, -sin_angle], [sin_angle, cos_angle]]
}

/// Reshape `index` (any shape ending in an axis of length 2) to `(N, 2)`, run
/// the `(N, 2) -> (N, 2)` function `f` on it, and reshape the result back to
/// the original shape.
///
/// This is the Rust equivalent of numpy's `ravel` + `reshape`. It lets a
/// function written for a stack of `(x, y)` pairs be applied to an array of any
/// rank whose last axis is the `(x, y)` pair, e.g. the `(ny, nx, 2)` array
/// returned by [`crate::Tile::indices`].
///
/// Non-contiguous inputs are handled: they are copied into a contiguous layout
/// before being reshaped.
pub fn map_point_pairs<D, A, B, F>(index: ArrayView<A, D>, f: F) -> Array<B, D>
where
    D: Dimension,
    A: Clone,
    F: FnOnce(ArrayView2<A>) -> Array2<B>,
{
    let dim = index.raw_dim();
    assert_eq!(
        *dim.slice()
            .last()
            .expect("expected a last axis of length 2"),
        2
    );
    let pattern = dim.clone().into_pattern();
    let raveled = index.to_shape((dim.size() / 2, 2)).unwrap();
    let result = f(raveled.view());
    result
        .into_shape(pattern)
        .expect("output of `f` is not compatible with the input shape")
}

/// Parallel counterpart of [`map_point_pairs`], only available with the
/// `parallel` feature. The serial helper above is left untouched so callers that
/// do not opt in (and call sites like `cell_corners` that depend on LLVM
/// inlining the closure) keep exactly the same codegen.
///
/// The batch is split into one contiguous row chunk per rayon worker. Chunks are
/// independent and concatenated in order, so the result is identical to the
/// serial one.
#[cfg(feature = "parallel")]
pub fn map_point_pairs_parallel<D, A, B, F>(index: ArrayView<A, D>, f: F) -> Array<B, D>
where
    D: Dimension,
    A: Clone + Sync,
    B: Send + Clone,
    F: Fn(ArrayView2<A>) -> Array2<B> + Sync,
{
    let dim = index.raw_dim();
    assert_eq!(
        *dim.slice()
            .last()
            .expect("expected a last axis of length 2"),
        2
    );
    let pattern = dim.clone().into_pattern();
    let raveled = index.to_shape((dim.size() / 2, 2)).unwrap();
    let result = if should_parallelize(raveled.nrows()) {
        map_rows_2(raveled.view(), &f)
    } else {
        f(raveled.view())
    };
    result
        .into_shape(pattern)
        .expect("output of `f` is not compatible with the input shape")
}

/// Like [`map_point_pairs`], but for functions that add an axis before the
/// trailing `(x, y)` axis, e.g. `(N, 2) -> (N, K, 2)`.
///
/// The output shape is the leading axes of the input followed by `(K, 2)`,
/// which is one dimension more than the input. `K` is read from the output of
/// `f` so the same helper works for e.g. cell corners and near-point lookups.
pub fn map_point_pairs_fanout<D, A, B, F>(index: ArrayView<A, D>, f: F) -> Array<B, D::Larger>
where
    D: Dimension,
    A: Clone,
    F: FnOnce(ArrayView2<A>) -> Array3<B>,
{
    let dim = index.raw_dim();
    assert_eq!(
        *dim.slice()
            .last()
            .expect("expected a last axis of length 2"),
        2
    );
    let raveled = index.to_shape((dim.size() / 2, 2)).unwrap();
    let result = f(raveled.view());
    let mut new_dims = dim.slice()[..dim.ndim() - 1].to_vec();
    new_dims.push(result.shape()[1]);
    new_dims.push(2);
    result
        .into_shape(IxDyn(&new_dims))
        .expect("intermediate reshape failed")
        .into_dimensionality::<D::Larger>()
        .expect("output rank does not match the input rank plus one")
}

/// Parallel counterpart of [`map_point_pairs_fanout`]; see
/// [`map_point_pairs_parallel`] for why the serial helper is kept separate.
#[cfg(feature = "parallel")]
pub fn map_point_pairs_fanout_parallel<D, A, B, F>(
    index: ArrayView<A, D>,
    f: F,
) -> Array<B, D::Larger>
where
    D: Dimension,
    A: Clone + Sync,
    B: Send + Clone,
    F: Fn(ArrayView2<A>) -> Array3<B> + Sync,
{
    let dim = index.raw_dim();
    assert_eq!(
        *dim.slice()
            .last()
            .expect("expected a last axis of length 2"),
        2
    );
    let raveled = index.to_shape((dim.size() / 2, 2)).unwrap();
    let result = if should_parallelize(raveled.nrows()) {
        map_rows_3(raveled.view(), &f)
    } else {
        f(raveled.view())
    };
    let mut new_dims = dim.slice()[..dim.ndim() - 1].to_vec();
    new_dims.push(result.shape()[1]);
    new_dims.push(2);
    result
        .into_shape(IxDyn(&new_dims))
        .expect("intermediate reshape failed")
        .into_dimensionality::<D::Larger>()
        .expect("output rank does not match the input rank plus one")
}

/// Like [`map_point_pairs_fanout`], but the caller's closure fills a
/// preallocated `(N, K, 2)` output in place instead of returning an owned array.
///
/// This is the building block for the parallel fanout queries: it lets each
/// rayon worker write straight into its slice of the result, so there is no
/// per-chunk allocation and no final `concatenate` copy. `K` is passed in
/// because the output must be allocated before the closure runs.
pub fn map_point_pairs_fanout_fill<D, A, B, F>(
    index: ArrayView<A, D>,
    k: usize,
    fill: F,
) -> Array<B, D::Larger>
where
    D: Dimension,
    A: Clone,
    B: Clone + Default,
    F: FnOnce(ArrayView2<A>, ArrayViewMut3<B>),
{
    let dim = index.raw_dim();
    assert_eq!(
        *dim.slice()
            .last()
            .expect("expected a last axis of length 2"),
        2
    );
    let raveled = index.to_shape((dim.size() / 2, 2)).unwrap();
    let mut out = Array3::<B>::from_elem((raveled.nrows(), k, 2), B::default());
    fill(raveled.view(), out.view_mut());
    let mut new_dims = dim.slice()[..dim.ndim() - 1].to_vec();
    new_dims.push(k);
    new_dims.push(2);
    out.into_shape(IxDyn(&new_dims))
        .expect("intermediate reshape failed")
        .into_dimensionality::<D::Larger>()
        .expect("output rank does not match the input rank plus one")
}

/// Parallel counterpart of [`map_point_pairs_fanout_fill`]; see
/// [`map_point_pairs_parallel`] for why the serial helpers are kept separate.
///
/// Unlike [`map_point_pairs_fanout_parallel`] this does not allocate a result
/// per chunk and concatenate them afterwards; the single output array is
/// preallocated and each worker fills its own disjoint row chunk in place.
#[cfg(feature = "parallel")]
pub fn map_point_pairs_fanout_fill_parallel<D, A, B, F>(
    index: ArrayView<A, D>,
    k: usize,
    fill: F,
) -> Array<B, D::Larger>
where
    D: Dimension,
    A: Clone + Sync,
    B: Clone + Default + Send,
    F: Fn(ArrayView2<A>, ArrayViewMut3<B>) + Sync,
{
    let dim = index.raw_dim();
    assert_eq!(
        *dim.slice()
            .last()
            .expect("expected a last axis of length 2"),
        2
    );
    let raveled = index.to_shape((dim.size() / 2, 2)).unwrap();
    let n = raveled.nrows();
    let mut out = Array3::<B>::from_elem((n, k, 2), B::default());
    if should_parallelize(n) {
        let chunk = n.div_ceil(get_num_threads());
        let sources: Vec<ArrayView2<A>> = raveled.axis_chunks_iter(Axis(0), chunk).collect();
        let destinations: Vec<ArrayViewMut3<B>> =
            out.axis_chunks_iter_mut(Axis(0), chunk).collect();
        with_pool(|pool| pool.install(|| {
            sources
                .into_par_iter()
                .zip(destinations.into_par_iter())
                .for_each(|(source, destination)| fill(source, destination));
        }));
    } else {
        fill(raveled.view(), out.view_mut());
    }
    let mut new_dims = dim.slice()[..dim.ndim() - 1].to_vec();
    new_dims.push(k);
    new_dims.push(2);
    out.into_shape(IxDyn(&new_dims))
        .expect("intermediate reshape failed")
        .into_dimensionality::<D::Larger>()
        .expect("output rank does not match the input rank plus one")
}

/// These aliases are what the hot batched queries call. Without the `parallel`
/// feature they resolve to the untouched serial helpers (identical codegen);
/// with it they resolve to the rayon-backed versions above.
#[cfg(not(feature = "parallel"))]
pub use map_point_pairs as map_point_pairs_batched;
#[cfg(feature = "parallel")]
pub use map_point_pairs_parallel as map_point_pairs_batched;

#[cfg(not(feature = "parallel"))]
pub use map_point_pairs_fanout as map_point_pairs_fanout_batched;
#[cfg(feature = "parallel")]
pub use map_point_pairs_fanout_parallel as map_point_pairs_fanout_batched;

#[cfg(not(feature = "parallel"))]
pub use map_point_pairs_fanout_fill as map_point_pairs_fanout_fill_batched;
#[cfg(feature = "parallel")]
pub use map_point_pairs_fanout_fill_parallel as map_point_pairs_fanout_fill_batched;

/// Like [`map_point_pairs`], but for functions that reduce the trailing
/// `(x, y)` pair to a single value, e.g. `(N, 2) -> (N,)`.
///
/// The output shape is the input shape with the last axis removed.
pub fn map_point_pairs_reduce<D, A, B, F>(index: ArrayView<A, D>, f: F) -> Array<B, D::Smaller>
where
    D: Dimension + RemoveAxis,
    A: Clone,
    F: FnOnce(ArrayView2<A>) -> Array1<B>,
{
    let dim = index.raw_dim();
    assert_eq!(
        *dim.slice()
            .last()
            .expect("expected a last axis of length 2"),
        2
    );
    let smaller = dim.clone().remove_axis(Axis(dim.ndim() - 1));
    let raveled = index.to_shape((dim.size() / 2, 2)).unwrap();
    let result = f(raveled.view());
    result
        .into_shape(smaller.into_pattern())
        .expect("output of `f` is not compatible with the reduced input shape")
}

#[cfg(all(test, feature = "parallel"))]
mod parallel_tests {
    use super::*;

    /// Inputs well above [`PARALLEL_MIN_POINTS`] so the chunked path is taken
    /// on a multi-core machine (the ordinary unit tests all use tiny batches and
    /// only ever exercise the serial fallback).
    #[test]
    fn parallel_pairs_match_serial_above_the_threshold() {
        let input = Array2::<f64>::from_shape_fn((PARALLEL_MIN_POINTS * 3 + 7, 2), |(i, axis)| {
            ((i * 7 + axis * 13) % 997) as f64 - 500.
        });
        let work = |rows: ArrayView2<f64>| rows.mapv(|v| v * 0.5 - 1.);
        let serial = map_point_pairs(input.view(), work);
        let parallel = map_point_pairs_parallel(input.view(), work);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn parallel_fanout_matches_serial_above_the_threshold() {
        let input = Array2::<i64>::from_shape_fn((PARALLEL_MIN_POINTS * 2 + 1, 2), |(i, axis)| {
            i as i64 * 3 + axis as i64
        });
        let work = |rows: ArrayView2<i64>| {
            let mut out = Array3::<i64>::zeros((rows.shape()[0], 3, 2));
            for r in 0..rows.shape()[0] {
                for k in 0..3 {
                    out[[r, k, 0]] = rows[[r, 0]] + k as i64;
                    out[[r, k, 1]] = rows[[r, 1]] - k as i64;
                }
            }
            out
        };
        let serial = map_point_pairs_fanout(input.view(), work);
        let parallel = map_point_pairs_fanout_parallel(input.view(), work);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn parallel_fanout_fill_matches_serial_above_the_threshold() {
        let input = Array2::<i64>::from_shape_fn((PARALLEL_MIN_POINTS * 2 + 1, 2), |(i, axis)| {
            i as i64 * 3 + axis as i64
        });
        let work = |rows: ArrayView2<i64>, mut out: ArrayViewMut3<i64>| {
            for r in 0..rows.shape()[0] {
                for k in 0..out.shape()[1] {
                    out[[r, k, 0]] = rows[[r, 0]] + k as i64;
                    out[[r, k, 1]] = rows[[r, 1]] - k as i64;
                }
            }
        };
        let serial = map_point_pairs_fanout_fill(input.view(), 3, work);
        let parallel = map_point_pairs_fanout_fill_parallel(input.view(), 3, work);
        assert_eq!(serial, parallel);
    }

    #[test]
    fn num_threads_is_at_least_one() {
        assert!(get_num_threads() >= 1);
    }

    #[test]
    fn physical_core_count_is_sane() {
        // On Linux this should resolve; elsewhere it may be `None`.
        if let Some(physical) = physical_core_count() {
            let logical = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(physical);
            assert!(physical >= 1 && physical <= logical);
        }
    }

    #[test]
    fn zero_threads_is_rejected() {
        // Rejected before any pool/state mutation, so this is safe even if the
        // shared test pool has already been built by another test.
        assert!(set_num_threads(Some(0)).is_err());
    }
}
