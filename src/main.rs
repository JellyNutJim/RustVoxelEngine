#![allow(dead_code, unused_imports)]
use vulkano::{buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo}, descriptor_set::layout, device::Features as DeviceFeatures, pipeline::{graphics::vertex_input::VertexInputState, ComputePipeline, Pipeline}};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::descriptor_set::{{allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo}, DescriptorSet, WriteDescriptorSet}, PersistentDescriptorSet};

use vulkano::descriptor_set::layout::DescriptorType;
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutCreateInfo};
use vulkano::shader::ShaderStages;

use vulkano::pipeline::compute::ComputePipelineCreateInfo;

use vulkano::descriptor_set::layout::{
    DescriptorSetLayoutBinding,
    DescriptorBindingFlags
};

mod vec3;
use vec3::Vec3;

use std::{error::Error, sync::Arc, time::Instant};
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
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, Version, VulkanError, VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};


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

    voxel_buffer: Subbuffer<[u32]>,
    camera_buffer: SubbufferAllocator,

    rcx: Option<RenderContext>,
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
struct Camera_Buffer_Data {
    origin: [f32; 3],
    look_at: [f32; 3],
    pixel00_loc: [f32; 3],
    pixel_delta_u: [f32; 3],
    pixel_delta_v: [f32; 3]
}

struct AppState {
    rotation: f64,
    last_frame: Instant,
}

impl App {
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
                    .position(|(i, q)| {

                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            
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

        // Set up buffers
        let chunk_count = 1;
        let chunk_size = 32 * 32 * 32;

        let mut voxel_data= vec![0; chunk_size * chunk_count];

        voxel_data[1 * (32 * 32) + 1 * 32 + 10] = 1;
        voxel_data[1 * (32 * 32) + 1 * 32 + 12] = 1;
        voxel_data[1 * (32 * 32) + 1 * 32 + 14] = 1;


        let voxel_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            voxel_data,
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

        let rcx = None;

        App {
            instance,
            device,
            queue,
            command_buffer_allocator,
            descriptor_set_allocator,
            voxel_buffer,
            camera_buffer,
            rcx,
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

        let (swapchain, images) = {
            // Querying the capabilities of the surface
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

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

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
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

                // PROCESS BUFFERS:


                let uniform_camera_subbuffer = {
                    //let camera = Camera {camera_pos: [0.0, 0.0, 0.0], look_at: [0.0, 0.0, -1.0], up: [0.0, 1.0, 0.0], fov: 90.0, aspect_ratio: 16.0/9.0 };
                    // Add conversion to 2d data
                    let window_size = rcx.window.inner_size();
                    let look_from = Vec3 {x: 0.0, y: 0.0, z: 1.0};
                    let look_at = Vec3 {x: 0.0, y: 0.0, z: 1.0};
                    let v_up = Vec3 {x: 0.0, y: 1.0, z: 0.0};
                    let fov = 90;


                    let focal_length = (look_from - look_at).magnitude();
                    let viewport_height = 2.0;
                    let viewport_width = viewport_height * (window_size.width as f64 / window_size.height as f64);

                    let w = (look_from - look_at).norm();
                    let u = v_up.cross(w).norm();
                    let v = w.cross(u);

                    let viewport_u = u * viewport_width;
                    let viewport_v = -v * viewport_height;

                    let pixel_delta_u = viewport_u / window_size.width as f64;
                    let pixel_delta_v = viewport_v/ window_size.height as f64;

                    let viewport_upper_left = look_from - (w * focal_length) - viewport_u/2.0 - viewport_v/2.0;
                    let pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;
                    
                    //println!("{:?} {:?} {:?}", pixel00_loc, pixel_delta_u, pixel_delta_v);
                    //println!("{} {} {}", pixel00_loc.x as f32, pixel00_loc.y as f32, pixel00_loc.z as f32);
                    //println!("{} {}", window_size.width, window_size.height);
                    

                    let camera = Camera_Buffer_Data {
                        origin: [look_from.x as f32, look_from.y as f32, look_from.z as f32],
                        look_at: [look_at.x as f32, look_at.y as f32, look_at.z as f32],
                        pixel00_loc: [pixel00_loc.x as f32, pixel00_loc.y as f32, pixel00_loc.z as f32],
                        pixel_delta_u: [pixel_delta_u.x as f32, pixel_delta_u.y as f32, pixel_delta_u.z as f32],
                        pixel_delta_v: [pixel_delta_v.x as f32, pixel_delta_v.y as f32, pixel_delta_v.z as f32],
                     };

                    let subbuffer = self.camera_buffer.allocate_sized().unwrap();
                    *subbuffer.write().unwrap() = camera;
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

                let descriptor_set = PersistentDescriptorSet::new(
                    &self.descriptor_set_allocator,
                    layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, uniform_camera_subbuffer), 
                        WriteDescriptorSet::buffer(1, self.voxel_buffer.clone()),
                        WriteDescriptorSet::image_view(2, rcx.attachment_image_views[image_index as usize].clone()),
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
                    
                // We add a draw command.

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
