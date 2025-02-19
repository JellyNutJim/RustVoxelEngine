use vulkano::{buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo}, device::Features as DeviceFeatures, pipeline::{ComputePipeline, Pipeline}};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::descriptor_set::{{allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo}, WriteDescriptorSet}, PersistentDescriptorSet};

use vulkano::descriptor_set::layout::DescriptorType;
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutCreateInfo};
use vulkano::shader::ShaderStages;

use vulkano::pipeline::compute::ComputePipelineCreateInfo;

use vulkano::descriptor_set::layout::DescriptorSetLayoutBinding;

use std::{error::Error, sync::Arc, f64::consts::PI};


use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderingAttachmentInfo, RenderingInfo,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions,
        Queue, QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::viewport::Viewport,
        PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, Version, VulkanError, VulkanLibrary,
};
use winit::{
    application::ApplicationHandler, dpi::PhysicalPosition, event::WindowEvent, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{CursorGrabMode, Window, WindowId}

};

use std::sync::{Mutex};
use std::thread;
use std::sync::mpsc::{channel, Sender, Receiver};

mod asset_load;
mod types;
mod world;

use world::{get_flat_world, get_empty, ShaderChunk, ShaderGrid};

use types::Vec3;

fn main() -> Result<(), impl Error> {
    // Create window
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    voxel_buffers: [Subbuffer<[u32]>; 2],
    current_voxel_buffer: usize,

    world_meta_data_buffer: Subbuffer<[i32]>,
    camera_buffer: SubbufferAllocator,

    rcx: Option<RenderContext>, 
    camera_location: CameraLocation,

    world_updater: WorldUpdater,
    update_receiver: Arc<Mutex<Receiver<WorldUpdateMessage>>>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    attachment_image_views: Vec<Arc<ImageView>>,

    compute_pipeline: Arc<ComputePipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

#[repr(C)]
#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(align(16))] 
struct CameraBufferData {
    origin: [f32; 4],
    pixel00_loc: [f32; 4],
    pixel_delta_u: [f32; 4],
    pixel_delta_v: [f32; 4],
    world_position: [f32; 4],
}

struct CameraLocation {
    location: Vec3,
    h_angle: f64,
    v_angle: f64,
    direction: Vec3,
}

use vulkano::command_buffer::CopyBufferInfo;


impl App {
    fn update_world(&mut self) {
        self.world_updater.request_update();
    }

    fn new(event_loop: &EventLoop<()>) -> Self {

        // Ready extensions
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(event_loop);

        // Create the instance.
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant Vulkan implementations. --------------> maybe remove
                // (e.g. MoltenVK)
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        // Get swapchain extension
        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        // Select physical device -> Ideally a discrete gpu, will allow selection at a later date
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                // `khr_dynamic_rendering` extension available.
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            .filter(|p| {
                // Changed line
                p.supported_extensions().contains(&device_extensions)
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(_i, q)| {

                        q.queue_flags.intersects(QueueFlags::COMPUTE)
                            
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| {
                // We assign a lower score to device types that are likely to be faster/better.
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no suitable physical device found");

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        // initiliase device and queues
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    ..DeviceFeatures::empty()
                },

                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        
        // Allocators
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                set_count: 10, // Adjust based on your needs
                update_after_bind: false,
                
                ..Default::default()
            },
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));


        let world = get_flat_world(42);

        // Extend vectors to desired size
        let mut voxels = world.1;
        let mut meta_data = world.0;

        // Resize to desired capacity (e.g., 1 million elements)
        voxels.resize(100_000_000, 0);  // Pad with zeros
        //meta_data.resize(1_000_000, 0);  // Pad with zeros

        let voxel_buffers = [
            Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                voxels.clone(),
            ).expect("failed to create buffer"),
            
            Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                voxels,
            ).expect("failed to create buffer"),
        ];

        //println!("{:?}", world.0.flatten());

        let world_meta_data_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            meta_data,
        )
        .expect("failed to create buffer");
    
        let camera_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let (world_updater, update_receiver) = WorldUpdater::new(
            device.clone(),
            queue.clone(),
            voxel_buffers.clone(),
            world_meta_data_buffer.clone(),
        );

        #[allow(unused_mut)]
        // Default values
        let mut camera_location = CameraLocation {location: Vec3::new(), direction: Vec3::new(), h_angle: 0.0, v_angle: 0.0};
        let rcx = None;

        let p = Vec3::new();
        let mut a = Vec3::from(1.0, 2.0, 3.0);
        a += p;

        println!("{:?}", a);


        App {
            instance,
            device,
            queue,
            command_buffer_allocator,
            descriptor_set_allocator,
            voxel_buffers,
            current_voxel_buffer: 0,
            world_meta_data_buffer,
            camera_buffer,
            rcx,
            camera_location,
            world_updater,
            update_receiver
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();
        
        // Confine cursor
        match window.set_cursor_grab(CursorGrabMode::Confined) {
            _ => {}
        }

        // Intially set cursor to centre of the screen
        window.set_cursor_position(PhysicalPosition::new(window_size.width as f64 / 2.0, window_size.height as f64 / 2.0)).expect("Cursor Error");
        
        // CURRENTLY BROKEN ON WINDOWS DISABLESM MOVE MOVEMENT EVEN DETECTION CAN USE USE DEVICE EVENT INSTEAD IF NEEDED
        //window.set_cursor_visible(false);

        
        let (swapchain, images) = {
            // Querying the capabilities of the surface
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            // Gets prefered image format from the surface, currently not using as im using a compute pipeline, will probaly come in use later
            // let (image_format, _) = self
            //     .device
            //     .physical_device()
            //     .surface_formats(&surface, Default::default())
            //     .unwrap()[0];

            // Please take a look at the docs for the meaning of the parameters we didn't mention.
            let image_format = vulkano::format::Format::R8G8B8A8_UNORM;
            Swapchain::new(
                self.device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),

                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,

                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),

                    ..Default::default()
                },
            )
            .unwrap()
        };

        let attachment_image_views = window_size_dependent_setup(&images);

        // Shader

        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/compute.glsl"
            } 
        }

        let shader = cs::load(self.device.clone()).expect("failed to create shader module");

        let cs = shader.entry_point("main").unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);

        let layout = {
            let bindings = [
                (
                    0,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::COMPUTE,
                        ..DescriptorSetLayoutBinding::descriptor_type(
                            DescriptorType::UniformBuffer,
                        )
                    }
                ),
                (
                    1,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::COMPUTE,
                        ..DescriptorSetLayoutBinding::descriptor_type(
                            DescriptorType::StorageBuffer,
                        )
                    }
                ),
                (
                    2,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::COMPUTE,
                        ..DescriptorSetLayoutBinding::descriptor_type(
                            DescriptorType::StorageBuffer,
                        )
                    }
                ),
                (
                    3,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::COMPUTE,
                        ..DescriptorSetLayoutBinding::descriptor_type(
                            DescriptorType::StorageImage,
                        )
                    }
                )
            ].into();
        
            DescriptorSetLayout::new(
                self.device.clone(),
                DescriptorSetLayoutCreateInfo {
                    bindings,
                    ..Default::default()
                },
            ).unwrap()
        };
        
        let pipeline_layout = PipelineLayout::new(
            self.device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![layout],
                ..Default::default()
            },
        )
        .unwrap();

        let compute_pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout),
        )
        .unwrap();


        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };


        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            attachment_image_views,
            compute_pipeline,
            viewport,
            recreate_swapchain,
            previous_frame_end,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();
        let w_size = rcx.window.inner_size();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            #[allow(unused_variables)]
            WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {
                // Simple movement

                let dis = Vec3::from(0.25, 0.25, 0.25);
                let up = Vec3::from(0.0, 1.0, 0.0);

                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => { self.camera_location.location = self.camera_location.location + self.camera_location.direction * dis }
                    PhysicalKey::Code(KeyCode::KeyS) => { self.camera_location.location = self.camera_location.location - self.camera_location.direction * dis }
                    PhysicalKey::Code(KeyCode::KeyA) => { self.camera_location.location = self.camera_location.location - (self.camera_location.direction.cross(up)) * dis  }
                    PhysicalKey::Code(KeyCode::KeyD) => { self.camera_location.location = self.camera_location.location + (self.camera_location.direction.cross(up)) * dis }
                    PhysicalKey::Code(KeyCode::Space) => { self.camera_location.location += Vec3::from(0.0, 0.25, 0.0)  }
                    PhysicalKey::Code(KeyCode::ControlLeft) => { self.camera_location.location += Vec3::from(0.0, -0.25, 0.0)  }
                    PhysicalKey::Code(KeyCode::KeyN) => { self.update_world(); }

                    
                    
                    PhysicalKey::Code(KeyCode::Escape) => { std::process::exit(0) }
                    _ =>  { print!("Non-Assigned Key")}
                }


            }
            #[allow(unused_variables)]
            WindowEvent::CursorMoved { device_id, position } => {
                // Only apply rotation and mouse locking when window is in focus
                if !rcx.window.has_focus() { 
                    return
                }

                // Get absolute pixel change from centre -> will need to change this as speed could be effected by resolution
                let x_change = position.x - (w_size.width as f64 / 2.0);
                let y_change = position.y - (w_size.height as f64 / 2.0);

                // Currently resets horizontal axis when it equals 2 PI as fully rotation is 2 pi -> -pi to +pi 
                self.camera_location.h_angle = (self.camera_location.h_angle + x_change * 0.001) % (PI*2.0);
                self.camera_location.v_angle = (self.camera_location.v_angle + (y_change * -0.001)).clamp(-PI/2.02, PI/2.02); 

                // Convert spherical angles to direction vector
                self.camera_location.direction = Vec3 {
                    x: self.camera_location.h_angle.cos() * self.camera_location.v_angle.cos(),
                    y: self.camera_location.v_angle.sin(),
                    z: self.camera_location.h_angle.sin() * self.camera_location.v_angle.cos()
                };

                // Set cursor position to centre of the window
                rcx.window.set_cursor_position(PhysicalPosition::new(rcx.window.inner_size().width as f64 / 2.0, rcx.window.inner_size().height as f64 / 2.0)).expect("Cursor Error");
                
            }
            WindowEvent::RedrawRequested => {

                if let Ok(rx_lock) = self.update_receiver.try_lock() {
                    if let Ok(WorldUpdateMessage::BufferUpdated(new_buffer_index)) = rx_lock.try_recv() {
                        // Buffer index only updates after both transfers are complete
                        self.current_voxel_buffer = new_buffer_index;
                    }
                }

                let window_size = rcx.window.inner_size();
                
                // Dont draw frame when minimised
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                // Free finished resources
                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Recreate everything upon screen resize
                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            image_format: rcx.swapchain.image_format(), // Maintain the same format
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");
                
                    rcx.swapchain = new_swapchain;
                    rcx.attachment_image_views = window_size_dependent_setup(&new_images);
                    rcx.viewport.extent = window_size.into();
                    rcx.recreate_swapchain = false;
                }

                // UPDATE HERE


                // Calculate camera buffer variables and set them to the buffer
                let uniform_camera_subbuffer = {
                    let look_from = self.camera_location.location;
                    //println!("{:?}", look_from);

                    let look_distance = 1.0;
                    let look_at = self.camera_location.location + self.camera_location.direction * look_distance;

                    let v_up = Vec3 {x: 0.0, y: 1.0, z: 0.0};
                    //let fov = 90;


                    let focal_length = (look_from - look_at).magnitude();
                    let viewport_height = 2.0;
                    let viewport_width = viewport_height * (w_size.width as f64 / w_size.height as f64);

                    let w = (look_from - look_at).norm();
                    let u = v_up.cross(w).norm();
                    let v = w.cross(u);

                    let viewport_u = u * viewport_width;
                    let viewport_v = -v * viewport_height;

                    let pixel_delta_u = viewport_u / w_size.width as f64;
                    let pixel_delta_v = viewport_v / w_size.height as f64;

                    let viewport_upper_left = look_from - (w * focal_length) - viewport_u/2.0 - viewport_v/2.0;
                    let pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;


                    // Caculate world positions at different scales
                    let world_position_1 = Vec3 {
                        x: look_from.x.floor(),
                        y: look_from.y.floor(),
                        z: look_from.z.floor(),
                    };

                    //println!("{:?} {:?} {:?} {:?}", world_position_1, world_position_2, world_position_4 ,world_position_8);

                    let c: CameraBufferData = CameraBufferData {
                        origin: [look_from.x as f32, look_from.y as f32, look_from.z as f32, 1.0],
                        pixel00_loc: [ pixel00_loc.x as f32, pixel00_loc.y as f32, pixel00_loc.z as f32, 1.0],
                        pixel_delta_u:[pixel_delta_u.x as f32, pixel_delta_u.y as f32, pixel_delta_u.z as f32, 1.0],
                        pixel_delta_v: [pixel_delta_v.x as f32, pixel_delta_v.y as f32, pixel_delta_v.z as f32, 1.0],
                        world_position: [world_position_1.x as f32, world_position_1.y as f32, world_position_1.z as f32, 1.0]
                    };

                    //println!("{:?}", look_from);
                    

                    let subbuffer = self.camera_buffer.allocate_sized().unwrap();
                    *subbuffer.write().unwrap() = c;
                    subbuffer

                };
                
                // Aquire next sqapchain image
                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                let layout = &rcx.compute_pipeline.layout().set_layouts()[0];
                //println!("Layout bindings: {:?}", layout.bindings());

                // Create descriptor set for the buffers
                let descriptor_set = PersistentDescriptorSet::new(
                    &self.descriptor_set_allocator,
                    layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, uniform_camera_subbuffer), 
                        WriteDescriptorSet::buffer(1, self.voxel_buffers[self.current_voxel_buffer].clone()),
                        WriteDescriptorSet::buffer(2, self.world_meta_data_buffer.clone()),
                        WriteDescriptorSet::image_view(3, rcx.attachment_image_views[image_index as usize].clone()),
                    ],
                    [],
                )
                .unwrap();

                // record command buffer
                let mut builder = AutoCommandBufferBuilder::primary(
                    &self.command_buffer_allocator,  // Add & here
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .bind_pipeline_compute(rcx.compute_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute, 
                        rcx.compute_pipeline.layout().clone(), 
                        0, 
                        descriptor_set.clone()
                    )
                    .unwrap()
                    .dispatch([window_size.width + 15 /  16, window_size.height + 15 / 16, 1])
                    .unwrap();
                    

                builder
                    .begin_rendering(RenderingInfo {
                        color_attachments: vec![Some(RenderingAttachmentInfo {
                            load_op: AttachmentLoadOp::Load, // Load the compute shader output
                            store_op: AttachmentStoreOp::Store,
                            clear_value: None, // No clear, as we're loading the computed image
                            ..RenderingAttachmentInfo::image_view(
                                rcx.attachment_image_views[image_index as usize].clone(),
                            )
                        })],
                        ..Default::default()
                    })
                    .unwrap()
                    .end_rendering()
                    .unwrap();
                // Finish recording the command buffer by calling `end`.
                let command_buffer = builder.build().unwrap();

                let future = rcx
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()

                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        rcx.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}


/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(images: &[Arc<Image>]) -> Vec<Arc<ImageView>> {
    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>()
}



// Message types for communication
enum WorldUpdateMessage {
    UpdateWorld,
    BufferUpdated(usize),
    Shutdown,
}

struct WorldUpdater {
    sender: Sender<WorldUpdateMessage>,
    update_thread: Option<thread::JoinHandle<()>>,
}

impl WorldUpdater {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        voxel_buffers: [Subbuffer<[u32]>; 2],
        world_meta_data_buffer: Subbuffer<[i32]>,
    ) -> (Self, Arc<Mutex<Receiver<WorldUpdateMessage>>>) {
        let (tx, rx) = channel();
        let tx_clone = tx.clone();
        let rx = Arc::new(Mutex::new(rx));
        let rx_thread = rx.clone();

        let update_thread = Some(thread::spawn(move || {
            let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
            let command_buffer_allocator = StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            );
            
            let mut shutdown = false;
            let mut current_buffer = 0;

            while !shutdown {
                // Lock the receiver to get the message
                if let Ok(msg) = rx_thread.lock().unwrap().recv() {
                    match msg {
                        WorldUpdateMessage::UpdateWorld => {
                            // Get next buffer index
                            let next_buffer = (current_buffer + 1) % 2;
                            
                            // Generate new world data
                            let world = get_empty();
                            let mut w = world.1;
                            w.resize(100_000_000, 0);

                            // Create staging buffers
                            let staging_buffer = Buffer::from_iter(
                                memory_allocator.clone(),
                                BufferCreateInfo {
                                    usage: BufferUsage::TRANSFER_SRC,
                                    ..Default::default()
                                },
                                AllocationCreateInfo {
                                    memory_type_filter: MemoryTypeFilter::PREFER_HOST 
                                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                                    ..Default::default()
                                },
                                w,
                            ).unwrap();

                            let staging_meta = Buffer::from_iter(
                                memory_allocator.clone(),
                                BufferCreateInfo {
                                    usage: BufferUsage::TRANSFER_SRC,
                                    ..Default::default()
                                },
                                AllocationCreateInfo {
                                    memory_type_filter: MemoryTypeFilter::PREFER_HOST 
                                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                                    ..Default::default()
                                },
                                world.0,
                            ).unwrap();

                            // Create and record command buffer
                            let mut builder = AutoCommandBufferBuilder::primary(
                                &command_buffer_allocator,
                                queue.queue_family_index(),
                                CommandBufferUsage::OneTimeSubmit,
                            ).unwrap();

                            builder
                                .copy_buffer(CopyBufferInfo::buffers(
                                    staging_buffer,
                                    voxel_buffers[next_buffer].clone(),
                                ))
                                .unwrap()
                                .copy_buffer(CopyBufferInfo::buffers(
                                    staging_meta,
                                    world_meta_data_buffer.clone(),
                                ))
                                .unwrap();

                            let command_buffer = builder.build().unwrap();

                            // Execute and wait for completion
                            let future = sync::now(device.clone())
                                .then_execute(queue.clone(), command_buffer)
                                .unwrap()
                                .then_signal_fence_and_flush()
                                .unwrap();

                            future.wait(None).unwrap();

                            // Update current buffer and notify main thread
                            current_buffer = next_buffer;
                            tx_clone.send(WorldUpdateMessage::BufferUpdated(next_buffer)).unwrap();
                        },
                        WorldUpdateMessage::Shutdown => {
                            shutdown = true;
                        },
                        _ => {}
                    }
                }
            }
        }));

        (
            WorldUpdater {
                sender: tx,
                update_thread,
            },
            rx
        )
    }

    pub fn request_update(&self) {
        self.sender.send(WorldUpdateMessage::UpdateWorld).unwrap();
    }
}

impl Drop for WorldUpdater {
    fn drop(&mut self) {
        self.sender.send(WorldUpdateMessage::Shutdown).unwrap();
        if let Some(handle) = self.update_thread.take() {
            handle.join().unwrap();
        }
    }
}