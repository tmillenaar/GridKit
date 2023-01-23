import numpy
from gridkit.errors import AlignmentError

def _total_bounds(left_bounds, right_bounds):
    return (
        min(left_bounds[0], right_bounds[0]),
        min(left_bounds[1], right_bounds[1]),
        max(left_bounds[2], right_bounds[2]),
        max(left_bounds[3], right_bounds[3]),
    )

def total_bounds(grids):
    if len(grids) == 1:
        return grids[0].bounds

    combined_bounds =  grids[0].bounds
    for grid in grids[1:]:
        combined_bounds = _total_bounds(combined_bounds, grid.bounds)
    return combined_bounds

def _empty_combined_grid(grids, value=numpy.nan):
    reference = grids[0]
    if not all([grid.is_aligned_with(reference)[0] for grid in grids[1:]]):
        raise AlignmentError("Not all grids are aligned. Please `resample` all grids on the same grid.")

    combined_bounds = total_bounds(grids)
    combined_data = numpy.full( # data shape is y,x
        (
            round((combined_bounds[3]-combined_bounds[1])/reference.dy),
            round((combined_bounds[2]-combined_bounds[0])/reference.dx)
        ),
        value,
        dtype=float # Fixme: don't hardcode dtype. This if workaround for working with NaNs
    )

    return reference.__class__(combined_data, bounds=combined_bounds, crs=reference.crs)

def count(grids):
    empty_grid = _empty_combined_grid(grids, value=0)
    combined_grid = empty_grid.copy()
    for grid in grids:
        value_ids = grid != grid.nodata_value
        numpy_ids = combined_grid.grid_id_to_numpy_id(value_ids.T)
        combined_grid._data[numpy_ids] += 1
    return combined_grid

def sum(grids):
    empty_grid = _empty_combined_grid(grids, value=numpy.nan)
    combined_grid = empty_grid.copy()
    for grid in grids: # add zero instead of NaN where there is coverage to allow adding
        combined_grid.assign(0, bounds=grid.bounds, in_place=True)
    empty_grid = _empty_combined_grid(grids, value=0) # TODO: add `replace` function

    
    # breakpoint()
    # normal, reverse = _BoundedGridMeta._gen_operator(lambda left, right: np.nansum([left, right], axis=0))



    for grid in grids: # add the values
        combined_grid += empty_grid.assign(grid.data, bounds=grid.bounds, in_place=False)
    return combined_grid

def mean(grids):
    return sum(grids) / count(grids)
