// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use anyhow::Result;
use gridkit_rs::grid::GridTraits;
use std::num::NonZeroUsize;
use std::sync::Arc;
use vello::kurbo::{Affine, Circle, Ellipse, Line, RoundedRect, Stroke, Point, PathEl};
use vello::peniko::Color;
use vello::util::{RenderContext, RenderSurface};
use vello::{AaConfig, Renderer, RendererOptions, Scene};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::*;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::Window;

use gridkit_rs::tri_grid;
use numpy::ndarray::*;

use vello::wgpu;
// Simple struct to hold the state of the renderer
pub struct ActiveRenderState<'s> {
    // The fields MUST be in this order, so that the surface is dropped before the window
    surface: RenderSurface<'s>,
    window: Arc<Window>,
}

enum RenderState<'s> {
    Active(ActiveRenderState<'s>),
    // Cache a window so that it can be reused when the app is resumed after being suspended
    Suspended(Option<Arc<Window>>),
}

struct SimpleVelloApp<'s> {
    // The vello RenderContext which is a global context that lasts for the
    // lifetime of the application
    context: RenderContext,

    // An array of renderers, one per wgpu device
    renderers: Vec<Option<Renderer>>,

    // State for our example where we store the winit Window and the wgpu Surface
    state: RenderState<'s>,

    // A vello Scene which is a data structure which allows one to build up a
    // description a scene to be drawn (with paths, fills, images, text, etc)
    // which is then passed to a renderer for rendering
    scene: Scene,
    triangle_color: Color,
    cursor_xy: (f64, f64)
}

fn sign(p1: Point, p2: Point, p3: Point) -> f64 {
    (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
}

fn point_in_triangle(pt: Point, v1: Point, v2: Point, v3: Point) -> bool {
    let d1 = sign(pt, v1, v2);
    let d2 = sign(pt, v2, v3);
    let d3 = sign(pt, v3, v1);

    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

    !(has_neg && has_pos)
}

impl<'s> ApplicationHandler for SimpleVelloApp<'s> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let RenderState::Suspended(cached_window) = &mut self.state else {
            return;
        };

        // Get the winit window cached in a previous Suspended event or else create a new window
        let window = cached_window
            .take()
            .unwrap_or_else(|| create_winit_window(event_loop));

        // Create a vello Surface
        let size = window.inner_size();
        let surface_future = self.context.create_surface(
            window.clone(),
            size.width,
            size.height,
            wgpu::PresentMode::AutoNoVsync,
        );
        let surface = pollster::block_on(surface_future).expect("Error creating surface");

        // Create a vello Renderer for the surface (using its device id)
        self.renderers
            .resize_with(self.context.devices.len(), || None);
        self.renderers[surface.dev_id]
            .get_or_insert_with(|| create_vello_renderer(&self.context, &surface));

        // Save the Window and Surface to a state variable
        self.state = RenderState::Active(ActiveRenderState { window, surface });
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let RenderState::Active(state) = &self.state {
            self.state = RenderState::Suspended(Some(state.window.clone()));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // Ignore the event (return from the function) if
        //   - we have no render_state
        //   - OR the window id of the event doesn't match the window id of our render_state
        //
        // Else extract a mutable reference to the render state from its containing option for use below
        let render_state = match &mut self.state {
            RenderState::Active(state) if state.window.id() == window_id => state,
            _ => return,
        };

        match event {
            // Exit the event loop when a close is requested (e.g. window's close button is pressed)
            WindowEvent::CloseRequested => event_loop.exit(),

            // Resize the surface when the window is resized
            WindowEvent::Resized(size) => {
                self.context
                    .resize_surface(&mut render_state.surface, size.width, size.height);
            }

            // This is where all the rendering happens
            WindowEvent::RedrawRequested => {
                // Empty the scene of objects to draw. You could create a new Scene each time, but in this case
                // the same Scene is reused so that the underlying memory allocation can also be reused.
                self.scene.reset();

                // Get the RenderSurface (surface + config)
                let surface = &render_state.surface;

                // Get the window size
                let width = surface.config.width;
                let height = surface.config.height;

                // Re-add the objects to draw to the scene.
                add_shapes_to_scene(&mut self.scene, &self.triangle_color, self.cursor_xy);


                // Get a handle to the device
                let device_handle = &self.context.devices[surface.dev_id];

                // Get the surface's texture
                let surface_texture = surface
                    .surface
                    .get_current_texture()
                    .expect("failed to get surface texture");

                // Render to the surface's texture
                self.renderers[surface.dev_id]
                    .as_mut()
                    .unwrap()
                    .render_to_surface(
                        &device_handle.device,
                        &device_handle.queue,
                        &self.scene,
                        &surface_texture,
                        &vello::RenderParams {
                            base_color: Color::BLACK, // Background color
                            width,
                            height,
                            antialiasing_method: AaConfig::Msaa16,
                        },
                    )
                    .expect("failed to render to surface");

                // Queue the texture to be presented on the surface
                surface_texture.present();

                device_handle.device.poll(wgpu::Maintain::Poll);
            }

            WindowEvent::CursorMoved { position, .. } => {
                println!("{:?}", position);
                // Triangle points
                let p1 = Point::new(500.0, 500.0);
                let p2 = Point::new(600.0, 700.0);
                let p3 = Point::new(400.0, 700.0);

                // Convert the cursor position to match the triangle coordinate space
                let cursor_point = Point::new(position.x as f64, position.y as f64);

                // Check if the cursor is within the triangle
                let is_hovering = point_in_triangle(cursor_point, p1, p2, p3);

                // Change color if hovering, else reset to default
                self.cursor_xy = (position.x, position.y);
                self.triangle_color = if is_hovering {
                    Color::rgb(1.0, 0.0, 0.0) // Red on hover
                } else {
                    Color::rgb(0.3, 0.8, 0.4) // Green otherwise
                };

                // Request a redraw
                render_state.window.request_redraw();
            }

            _ => {}
        }

    }
}

fn main() -> Result<()> {
    // Setup a bunch of state:
    let mut app = SimpleVelloApp {
        context: RenderContext::new(),
        renderers: vec![],
        state: RenderState::Suspended(None),
        scene: Scene::new(),
        triangle_color: Color::rgb(0.3, 0.8, 0.4), // Initial green color
        cursor_xy: (0.,0.),
    };

    // Create and run a winit event loop
    let event_loop = EventLoop::new()?;
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
    Ok(())
}

/// Helper function that creates a Winit window and returns it (wrapped in an Arc for sharing between threads)
fn create_winit_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
    let attr = Window::default_attributes()
        .with_inner_size(LogicalSize::new(1044, 800))
        .with_resizable(true)
        .with_title("Vello Shapes");
    Arc::new(event_loop.create_window(attr).unwrap())
}

/// Helper function that creates a vello `Renderer` for a given `RenderContext` and `RenderSurface`
fn create_vello_renderer(render_cx: &RenderContext, surface: &RenderSurface) -> Renderer {
    Renderer::new(
        &render_cx.devices[surface.dev_id].device,
        RendererOptions {
            surface_format: Some(surface.format),
            use_cpu: false,
            antialiasing_support: vello::AaSupport::all(),
            num_init_threads: NonZeroUsize::new(1),
        },
    )
    .expect("Couldn't create renderer")
}

/// Add shapes to a vello scene. This does not actually render the shapes, but adds them
/// to the Scene data structure which represents a set of objects to draw.
fn add_shapes_to_scene(scene: &mut Scene, triangle_color: &Color, cursor_xy: (f64, f64)) {
    // Draw an outlined rectangle
    let stroke = Stroke::new(6.0);
    let rect = RoundedRect::new(10.0, 10.0, 240.0, 240.0, 20.0);
    let rect_stroke_color = Color::rgb(0.9804, 0.702, 0.5294);
    scene.stroke(&stroke, Affine::IDENTITY, rect_stroke_color, None, &rect);

    let triangle_path = [
            PathEl::MoveTo(Point::new(500.0, 500.0)),
            PathEl::LineTo(Point::new(600.0, 700.0)),
            PathEl::LineTo(Point::new(400.0, 700.0)),
            PathEl::ClosePath,
        ];

        let triangle_fill_color = triangle_color;//Color::rgb(0.3, 0.8, 0.4); // Green color for the triangle
        scene.fill(
            vello::peniko::Fill::NonZero,
            Affine::IDENTITY,
            triangle_fill_color,
            None,
            &triangle_path,
        );

    let hex_path = [
            PathEl::MoveTo(Point::new(500.0, 360.84391824)),
            PathEl::LineTo(Point::new(500.0, 418.57894516)),
            PathEl::LineTo(Point::new(450.0, 447.44645862)),
            PathEl::LineTo(Point::new(400.0, 418.57894516)),
            PathEl::LineTo(Point::new(400.0, 360.84391824)),
            PathEl::LineTo(Point::new(450.0, 331.97640478)),
            PathEl::ClosePath,
        ];

    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        triangle_fill_color,
        None,
        &hex_path,
    );

    // Draw a filled ellipse
    let ellipse = Ellipse::new((250.0, 420.0), (100.0, 160.0), -90.0);
    let ellipse_fill_color = Color::rgb(0.7961, 0.651, 0.9686);
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        ellipse_fill_color,
        None,
        &ellipse,
    );

    // Draw a straight line
    let line = Line::new((260.0, 20.0), (620.0, 100.0));
    let line_stroke_color = Color::rgb(0.5373, 0.7059, 0.9804);
    scene.stroke(&stroke, Affine::IDENTITY, line_stroke_color, None, &line);

    // Create gridkit shape
    let rot = (cursor_xy.0) / 2.;
    let grid = tri_grid::TriGrid::new(60., (0.,0.), rot);
    let start_id: Array2<i64> = Array2::from_shape_vec((1, 2), vec![0, 0]).unwrap();
    let ids = grid.all_neighbours(&start_id.view(), 6, true, true);
    let ids = ids.clone().into_shape((ids.shape()[0]*ids.shape()[1], 2)).unwrap();
    let corners = grid.cell_corners(&(ids.view()));

    let cx = cursor_xy.0 - grid.dx()/2.;
    let cy = cursor_xy.1 - grid.dy();
    for i in 0..corners.shape()[0] {
        let triangle_path = [
                PathEl::MoveTo(Point::new(cx + corners[Ix3(i, 0, 0)], cy + corners[Ix3(i, 0, 1)])),
                PathEl::LineTo(Point::new(cx + corners[Ix3(i, 1, 0)], cy + corners[Ix3(i, 1, 1)])),
                PathEl::LineTo(Point::new(cx + corners[Ix3(i, 2, 0)], cy + corners[Ix3(i, 2, 1)])),
                PathEl::ClosePath,
            ];

            let triangle_fill_color = Color::rgb(i as f64 / corners.shape()[0] as f64, 0.8, 0.4);//Color::rgb(0.3, 0.8, 0.4); // Green color for the triangle
            scene.fill(
                vello::peniko::Fill::NonZero,
                Affine::IDENTITY,
                triangle_fill_color,
                None,
                &triangle_path,
            );
    }
}
