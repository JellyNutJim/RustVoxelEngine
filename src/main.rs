use vulkano::{buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo}, command_buffer::PrimaryCommandBufferAbstract, device::Features as DeviceFeatures, pipeline::{ComputePipeline, Pipeline}, sync::Sharing};
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
    DeviceSize,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
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
    application::ApplicationHandler, dpi::PhysicalPosition, event::{MouseButton, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{CursorGrabMode, Fullscreen, Window, WindowId}

};

use vulkano::command_buffer::CopyBufferInfo;

use chrono::Local;
use rand::Rng;
use std::{time::Instant, fs::File, io::Write, path::Path};
use crossbeam_channel::Receiver;

mod asset_load;
mod noise_gen;
mod types;
mod world;
mod rendering;
mod testing;

use rendering::{ WorldUpdateMessage, WorldUpdater, Update, CameraBufferData, CameraLocation };
use world::{get_grid_from_seed, get_empty_grid, ShaderChunk, ShaderGrid};


use types::{Vec3, Voxel, Geometry};
use noise_gen::PerlinNoise; 

// Camera Settings
const CONFINE_CURSOR: bool = false;
const ORIENTATION_MOVEMENT: bool = true;
const POSTIONAL_MOVEMENT: bool = true;
const WORLD_INTERACTION: bool = false;
const AUTO_MOVE_FORWARDS: bool = false;
const STARTING_ORIENTATION: (f64, f64) = (PI, 0.0);

// Testing constants
const MEASURE_FRAME_TIMES: bool = false;
const MEASURE_MARCH_DATA: bool = false; // frame times must also be true
const PRINT_FRAME_STATS: bool = false;

// Render Options
const USE_BEAM_OPTIMISATION: bool = true;
const RESIZEABLE_WINDOW: bool = false;
const USE_VSYNC: bool = false;
const USE_FULLSCREEN: bool = false;
const RESOLUTION: (u32, u32) = (1920, 1080);

// Sarting conditions
const SEED: u64 = 42;
const USE_EMPTY_GRID: bool = false;


fn main() -> Result<(), impl Error> {

    // Demonstration of memory
    //test();

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

    world_meta_data_buffers: [Subbuffer<[i32]>; 2],
    noise_buffer: Subbuffer<[f32]>,
    camera_buffer: SubbufferAllocator,
    
    ray_distance_buffer: Subbuffer<[f32]>,

    stat_buffer: Subbuffer<[u32]>,

    rcx: Option<RenderContext>, 
    camera_location: CameraLocation,

    update_receiver: Receiver<WorldUpdateMessage>,
    world_updater: WorldUpdater,

    last_n_press: bool,
    last_frame_time: Instant,

    start: Instant,
    frame_times: Vec<f64>,
    render_data: Vec<(u32, u32, u32)>
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    attachment_image_views: Vec<Arc<ImageView>>,

    initial_pipeline: Arc<ComputePipeline>,
    main_pipeline: Arc<ComputePipeline>, // Renamed for clarity
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl App {
    fn update_world(&mut self) {
        let mut rng = rand::rng();

        self.world_updater.request_update(Update::SwitchSeed(rng.random_range(1..99999)));
    }

    fn place_voxel(&mut self, u: Update) {
        if let Update::AddVoxel(_, _, _, _) = u {
            self.world_updater.request_update(u);
        }
        else {
            panic!("Tried to add a voxel using incorrect update type");
        }
    }

    fn shift_world(&mut self, axis: usize, dir: i32) {
        self.world_updater.request_update(Update::Shift(axis, dir));
    }   

    // saves the frame times and, if used, march data. -> Currently just saves file to current directory
    fn save_performance_data(&self) {
        // Empty
        if self.frame_times.is_empty() {
            return;
        }
    
        let now = Local::now();
        let filename = format!("frame_times_{}.csv", now.format("%Y%m%d_%H%M%S"));
        
        let path = Path::new(&filename);
        let mut file = match File::create(&path) {
            Ok(file) => file,
            Err(e) => {
                println!("Failed to create file: {}", e);
                return;
            }
        };
        
        // Save frame times and march data if true
        if MEASURE_MARCH_DATA {
            if let Err(e) = writeln!(file, "frame_number,frame_time_ms,total_steps,rays_hit,rays_missed") {
                println!("{}", e);
                return;
            }
            
            for (i, (&time, &march_data)) in self.frame_times.iter().zip(self.render_data.iter()).enumerate() {
                let (total_steps, rays_hit, rays_missed) = march_data;
                if let Err(e) = writeln!(file, "{},{:.4},{},{},{}", 
                                         i+1, time * 1000.0, total_steps, rays_hit, rays_missed) {
                    println!("failed to write data: {}", e);
                    return;
                }
            }
        } else { 
            // Just save frame times
            if let Err(e) = writeln!(file, "frame_number,frame_time_ms") {
                println!("{}", e);
                return;
            }
            
            for (i, &time) in self.frame_times.iter().enumerate() {
                if let Err(e) = writeln!(file, "{},{:.4}", i+1, time * 1000.0) {
                    println!("failed to write data: {}", e);
                    return;
                }
            }
        }
        
        println!("Performance data saved to {}", filename);
    }

    fn new(event_loop: &EventLoop<()>) -> Self {

        // Ready extensions
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(event_loop);

        // Create the instance.
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                // Enable enumerating --------------> maybe remove
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

        // Select physical device -> Ideally a discrete gpu, will allow selection at a later date (probably never)
        let (physical_device, queue_family_indicies) = instance
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
                
               // Find compute queue family
               let compute_family_index = p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(_i, q)| {
                        q.queue_flags.intersects(QueueFlags::COMPUTE)
                    })
                    .map(|i| i as u32);
           
                // Find queue family that supports transfer, and ideally not graphics
                let transfer_family_index = p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(_i, q)| {
                        q.queue_flags.contains(QueueFlags::TRANSFER) && 
                        !q.queue_flags.contains(QueueFlags::COMPUTE)
                    })
                    .map(|i| i as u32)
                    .or_else(|| {
                        // Use Transfer and graphics if needed
                        p.queue_family_properties()
                            .iter()
                            .enumerate()
                            .position(|(_i, q)| {
                                q.queue_flags.contains(QueueFlags::TRANSFER)
                            })
                            .map(|i| i as u32)
                    });
                    
                // Ensure both queues are defined
                if let (Some(compute), Some(transfer)) = (compute_family_index, transfer_family_index) {
                    Some((p, (compute, transfer)))
                } else {
                    None
                }
            })
            .min_by_key(|(p, _)| {
                // GPU Ideally
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

        let (compute_index, transfer_index) = queue_family_indicies;

        // Create info prrior to initialisation
        let queue_create_infos = if compute_index == transfer_index {
            vec![QueueCreateInfo {
                queue_family_index: compute_index,
                queues: vec![1.0], 
                ..Default::default()
            }]
        } else {
            vec![
                QueueCreateInfo {
                    queue_family_index: compute_index,
                    queues: vec![1.0], 
                    ..Default::default()
                },
                QueueCreateInfo {
                    queue_family_index: transfer_index,
                    queues: vec![1.0],
                    ..Default::default()
                }
            ]
        };

        // Initiliase device and queues
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos,
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    ..DeviceFeatures::empty()
                },

                ..Default::default()
            },
        )
        .unwrap();

        // Get quues
        let compute_queue = queues.next().unwrap();
        let transfer_queue = if compute_index == transfer_index { compute_queue.clone() } else { queues.next().unwrap() };

        //println!("{} {}", transfer_index, compute_index);

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

        let width = 321.0;
        //let middle = Vec3::from(64.0 * width + (64.0 * width)/2.0, 830.0, 64.0 * width + (64.0 * width)/2.0);
        let mid = u32::max_value() as f64 / 100.0;  // Basically i32 mid untio
        println!("{}", mid); 
        //let middle = Vec3::from(mid,830.0, mid);


        let mut x = 17700.0;
        let y = 1400.0;
        let z = 10560.0;

        //let seed = 42;
        //let seed = 1023;

        let seed = SEED;


        let temp = PerlinNoise::new(seed, 0.00003);
        let mut found = false;
        
        // Ensures camera always starts on land
        while !found {
            if temp.get_noise_at_point(x, z) > 0.05 {
                found = true;
            } else {
                x += 1.0;
            }
        }

        let h_angle: f64 = STARTING_ORIENTATION.0;
        let v_angle: f64 = STARTING_ORIENTATION.1;

        let initial_direction = Vec3 {
            x: h_angle.cos() * v_angle.cos(),
            y: v_angle.sin(),
            z: h_angle.sin() * v_angle.cos()
        };

        let middle = Vec3::from(x, y, z);

        let camera_location = CameraLocation { 
            location: middle, 
            direction: initial_direction, 
            old_loc: middle, 
            h_angle: h_angle, 
            v_angle: v_angle, 
            sun_loc: Vec3::from(10000.0, 3000.0, 10000.0)
        };

        let mut initial_world = if USE_EMPTY_GRID {
            get_empty_grid(width as i32, [middle.x as i32, middle.y as i32, middle.z as i32])
        } else {
            get_grid_from_seed(42, width as i32, [middle.x as i32, middle.y as i32, middle.z as i32])
        };

        // For testing
        // let v32 = Voxel::from_quadrants(
        //     [
        //         0b_00000001,
        //         0b_00001100,
        //         0b_10001000,
        //         0b_11111111,

        //     ]
        // );

        //intial_world.insert_subchunk([middle.x as i32, middle.y as i32, (middle.z + 1.0) as i32], v32, 4, false);
        //intial_world.insert_voxel([middle.x as i32, middle.y as i32, (middle.z + 4.0) as i32], v32, false);

        println!("{:?}", middle);

        // Get Vector World Data
        let flat_world = initial_world.flatten_world();
        let mut voxels = flat_world.1;
        let mut meta_data = flat_world.0;

        println!("len: {} {}", voxels.len(), meta_data.len());

        // Resize
        voxels.resize(500000000, 0);  
        meta_data.resize(40000000, 0);

        //meta_data.resize(1_000_000, 0);  

        let voxel_buffers = [
            Buffer::new_slice::<u32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    sharing: Sharing::Exclusive,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                voxels.len() as DeviceSize,
            ).expect("failed to create buffer"),

            Buffer::new_slice::<u32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    sharing: Sharing::Exclusive,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                voxels.len() as DeviceSize,
            ).expect("failed to create buffer"),
        ];

        //println!("{:?}", world.0.flatten());

        let world_meta_data_buffers = [
            Buffer::new_slice::<i32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    sharing: Sharing::Exclusive,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                meta_data.len() as DeviceSize,
            )
            .expect("failed to create buffer"),

            Buffer::new_slice::<i32>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                    sharing: Sharing::Exclusive,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                meta_data.len() as DeviceSize,
            )
            .expect("failed to create buffer"),
        ];

        let perm = initial_world.generator.landmass.permutation.clone();
        let grad: Vec<f64> = initial_world.generator.landmass.gradients.clone().iter()
                .flat_map(|&gradient| gradient.into_iter())
                .collect();


        let mut combined_data: Vec<f32> = Vec::with_capacity(perm.len() + grad.len());

        combined_data.extend(perm.iter().map(|&p| p as f32));
        combined_data.extend(grad.iter().map(|&g| g as f32));

        //println!("com {}", combined_data.len());
        //println!("com {:?}", combined_data);

        // Stores every other pixel, supports up to 8k resolution screen size
        let ray_distance_buffer_size = ( 7680.0 * 4320.0 ) as i32 + 15;
        let stat_buffer_size = 3;

        let ray_distance_buffer = Buffer::new_slice::<f32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                sharing: Sharing::Exclusive,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            ray_distance_buffer_size as DeviceSize,
        ).expect("failed to recreate quarter resolution buffer");


        // Stat Buffer -----------------------------------------------------------------

        let stat_buffer = Buffer::new_slice::<u32>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            stat_buffer_size as DeviceSize,
        ).expect("failed to recreate buffer");

         // Noise Buffer -----------------------------------------------------------------

        let noise_buffer = Buffer::from_iter(
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
            combined_data,
        ).expect("failed to create buffer");
    

        let camera_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let temporary_accessible_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                // Specify that this buffer will be used as a transfer source.
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                // Specify use for upload to the device.
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            voxels,
        )
        .unwrap();

        let temporary_accessible_buffer2 = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                // Specify that this buffer will be used as a transfer source.
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                // Specify use for upload to the device.
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            meta_data,
        )
        .unwrap();

        let i = Instant::now();

        let mut cbb = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            transfer_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        cbb.copy_buffer(CopyBufferInfo::buffers(
            temporary_accessible_buffer.clone(),
            voxel_buffers[0].clone(),
        ))
        .unwrap()
        .copy_buffer(CopyBufferInfo::buffers(
            temporary_accessible_buffer,
            voxel_buffers[1].clone(),
        ))
        .unwrap()
        .copy_buffer(CopyBufferInfo::buffers(
            temporary_accessible_buffer2.clone(),
            world_meta_data_buffers[0].clone(),
        ))
        .unwrap()
        .copy_buffer(CopyBufferInfo::buffers(
            temporary_accessible_buffer2,
            world_meta_data_buffers[1].clone(),
        ))
        .unwrap();

        let cb = cbb.build().unwrap();
        
        // Execute the copy command and wait for completion before proceeding.
        cb.execute(transfer_queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        println!("{}", i.elapsed().as_millis());

        let (world_updater, update_receiver) = WorldUpdater::new(
            device.clone(),
            transfer_queue.clone(),
            compute_queue.clone(),
            voxel_buffers.clone(),
            world_meta_data_buffers.clone(),
            initial_world,
        );

        #[allow(unused_mut)]
        // Default values

        let rcx = None;

        let p = Vec3::new();
        let mut a = Vec3::from(1.0, 2.0, 3.0);
        a += p;

        println!("{:?}", a);

        let last_n_press = false;
        let last_frame_time = Instant::now();
        let start = Instant::now();
        let frame_times: Vec<f64> = Vec::new();
        let render_data: Vec<(u32, u32, u32)> = Vec::new();

        App {
            instance,
            device,
            queue: compute_queue,
            command_buffer_allocator,
            descriptor_set_allocator,
            voxel_buffers,
            current_voxel_buffer: 0,
            world_meta_data_buffers,
            noise_buffer,
            camera_buffer,
            ray_distance_buffer,
            stat_buffer,
            rcx,
            camera_location,
            world_updater,
            update_receiver,
            last_n_press,
            last_frame_time,
            start,
            frame_times,
            render_data,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes()
                    .with_inner_size(winit::dpi::LogicalSize::new(RESOLUTION.0, RESOLUTION.1))
                    .with_title("Engine")
                    .with_resizable(RESIZEABLE_WINDOW)
                )
                .unwrap(),
        );

        if USE_FULLSCREEN == true {
            if let Some(primary_monitor) = event_loop.primary_monitor() {
                // Define your desired resolution
                let desired_width = RESOLUTION.0;
                let desired_height = RESOLUTION.1;
                
                // Find a video mode that matches your desired resolution
                let video_mode = primary_monitor.video_modes()
                    .find(|mode| {
                        let size = mode.size();
                        size.width == desired_width && size.height == desired_height
                    });
                
                if let Some(mode) = video_mode {
                    // Use the specific video mode with your desired resolution
                    window.set_fullscreen(Some(Fullscreen::Exclusive(mode)));
                } else {
                    // Fall back to borderless if specific resolution not found
                    window.set_fullscreen(Some(Fullscreen::Borderless(Some(primary_monitor))));
                    println!("Desired resolution not available, using borderless fullscreen");
                }
            }
        }

        // if let Some(video_mode) = primary_monitor.video_modes().next() {
        //     window.set_fullscreen(Some(Fullscreen::Exclusive(video_mode)));
        // } else {
        //     window.set_fullscreen(Some(Fullscreen::Borderless(Some(primary_monitor))));
        // }
        
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();
        
        // Confine cursor
        if CONFINE_CURSOR == true {
            match window.set_cursor_grab(CursorGrabMode::Confined) {
                _ => {}
            }
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

            let present_mode = if USE_VSYNC == true {
                vulkano::swapchain::PresentMode::Fifo
            } else {
                vulkano::swapchain::PresentMode::Immediate
            };

            let image_format = vulkano::format::Format::R8G8B8A8_UNORM;
            Swapchain::new(
                self.device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),

                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::STORAGE,
                    present_mode,
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
        mod initial_beam {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/initial_beam.glsl",
                include: ["src/shaders/compute_utils"]
            } 
        }

        mod multi_stage_beam {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/secondary_beam.glsl",
                include: ["src/shaders/compute_utils"]
            } 
        }

        mod single_stage_beam {
            vulkano_shaders::shader! {
                ty: "compute",
                path: "src/shaders/single_stage_march.glsl",
                include: ["src/shaders/compute_utils"]
            } 
        }

        let initial_shader = initial_beam::load(self.device.clone()).expect("failed to create preprocess shader module");
        let initial_cs = initial_shader.entry_point("main").unwrap();
        let initial_stage = PipelineShaderStageCreateInfo::new(initial_cs);
        
        let main_shader = multi_stage_beam::load(self.device.clone()).expect("failed to create main shader module");
        let main_cs = main_shader.entry_point("main").unwrap();
        let main_stage = PipelineShaderStageCreateInfo::new(main_cs);

        let single_stage_beam_shader = single_stage_beam::load(self.device.clone()).expect("failed to create single stage beam shader module");
        let single_cs = single_stage_beam_shader.entry_point("main").unwrap();
        let single_stage = PipelineShaderStageCreateInfo::new(single_cs);

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
                            DescriptorType::StorageBuffer,
                        )
                    }
                ),
                (
                    4,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::COMPUTE,
                        ..DescriptorSetLayoutBinding::descriptor_type(
                            DescriptorType::StorageBuffer,
                        )
                    }
                ),
                (
                    5,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::COMPUTE,
                        ..DescriptorSetLayoutBinding::descriptor_type(
                            DescriptorType::StorageBuffer,
                        )
                    }
                ),
                (
                    6,
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

        let initial_pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(initial_stage, pipeline_layout.clone()),
        )
        .unwrap();

        // use beam or normal
        let main_pipeline = if USE_BEAM_OPTIMISATION == true {
            ComputePipeline::new(
                self.device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(main_stage, pipeline_layout.clone()),
            )
            .unwrap()
        } else {
            ComputePipeline::new(
                self.device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(single_stage, pipeline_layout.clone()),
            )
            .unwrap()
        };

        // Dynamic viewports
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
            initial_pipeline,
            main_pipeline,
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

                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => { std::process::exit(0) }
                    PhysicalKey::Code(KeyCode::KeyR) => { if MEASURE_FRAME_TIMES == true { self.save_performance_data(); } }
                    _ =>  { } 
                }
                
                if POSTIONAL_MOVEMENT == false {
                    return;
                }

                let dis = Vec3::from(0.25, 0.25, 0.25) * 2.0;
                let up = Vec3::from(0.0, 1.0, 0.0);

                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => { self.camera_location.location = self.camera_location.location + self.camera_location.direction * dis }
                    PhysicalKey::Code(KeyCode::KeyS) => { self.camera_location.location = self.camera_location.location - self.camera_location.direction * dis }
                    PhysicalKey::Code(KeyCode::KeyA) => { self.camera_location.location = self.camera_location.location - (self.camera_location.direction.cross(up)) * dis  }
                    PhysicalKey::Code(KeyCode::KeyD) => { self.camera_location.location = self.camera_location.location + (self.camera_location.direction.cross(up)) * dis }
                    PhysicalKey::Code(KeyCode::Space) => { self.camera_location.location += Vec3::from(0.0, 0.25, 0.0)  }
                    PhysicalKey::Code(KeyCode::ControlLeft) => { self.camera_location.location += Vec3::from(0.0, -0.25, 0.0)  }
                    PhysicalKey::Code(KeyCode::ArrowRight) => { self.camera_location.sun_loc -= Vec3::from(100.0, 0.0, 0.0) }
                    PhysicalKey::Code(KeyCode::ArrowLeft) => { self.camera_location.sun_loc += Vec3::from(100.0, 0.0, 0.0) }
                    PhysicalKey::Code(KeyCode::ArrowUp) => { self.camera_location.sun_loc += Vec3::from(0.0, 100.0, 0.0) }
                    PhysicalKey::Code(KeyCode::ArrowDown) => { self.camera_location.sun_loc -= Vec3::from(0.0, 100.0, 0.0) }

                    PhysicalKey::Code(KeyCode::KeyP) => { 
                        let now = Instant::now();
                        if self.last_n_press == false {
                            self.shift_world(2, -1);
                            //self.last_n_press = true;
                        }
                    }

                    PhysicalKey::Code(KeyCode::KeyN) => {
                        let now = Instant::now();
                        if self.last_n_press == false {
                            self.last_n_press = true;
                            self.update_world();
                        }
                    }

                    PhysicalKey::Code(KeyCode::KeyM) => {
                        let now = Instant::now();
                        if self.last_n_press == false {
                            self.current_voxel_buffer = (self.current_voxel_buffer + 1) % 2; 
                            println!("Set buffer index to {}", self.current_voxel_buffer);
                        }
                    }
                    _ =>  { print!("Non-Assigned Key")}
                }
            }
            #[allow(unused_variables)]
            WindowEvent::MouseInput { device_id, state, button } => {
                if WORLD_INTERACTION == false {
                    return;
                }


                match button {
                    MouseButton::Left => {
                        let voxel_loc = self.camera_location.location + self.camera_location.direction * 2.0;
                        let u = Update::AddVoxel(voxel_loc.x as i32, voxel_loc.y as i32, voxel_loc.z as i32, 858993459);

                        let now = Instant::now();
                        if self.last_n_press == false {
                            self.place_voxel(u);
                            self.last_n_press = true;
                        }
                    }
                    MouseButton::Right => {
                        let voxel_loc = self.camera_location.location + self.camera_location.direction * 2.0;
                        let u = Update::AddVoxel(voxel_loc.x as i32, voxel_loc.y as i32, voxel_loc.z as i32, 0);

                        let now = Instant::now();
                        if self.last_n_press == false {
                            self.place_voxel(u);
                            self.last_n_press = true;
                        }
                    }
                    _ => {}
                }
            }
            #[allow(unused_variables)]
            WindowEvent::CursorMoved { device_id, position } => {

                // Disable mouse movement
                if ORIENTATION_MOVEMENT == false {
                    return;
                }

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

                //println!("{}", self.camera_location.v_angle);

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
                let frame_start = std::time::Instant::now();
                
                if AUTO_MOVE_FORWARDS {
                    let delta_time = frame_start.duration_since(self.last_frame_time);

                    self.last_frame_time = frame_start;
                    let base_speed = Vec3::from(0.2, 0.2, 0.2);

                    let delta_seconds = delta_time.as_secs_f64() + delta_time.subsec_nanos() as f64 / 1_000_000_000.0;
                    let movement = self.camera_location.direction * base_speed * delta_seconds * 60.0; 
            
                    self.camera_location.location = self.camera_location.location + movement;
                }
                
                if let Some(future) = &mut rcx.previous_frame_end {
                    future.cleanup_finished();
                    rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                }

                if MEASURE_MARCH_DATA == true {
                    let stats = self.stat_buffer.read().unwrap();
                    let total_steps = stats[0]; // Total steps 
                    let rays_cast = stats[1];   // Total rays that hit some geometry
                    let rays_missed = stats[2];
                    
                    self.render_data.push((total_steps, rays_cast, rays_missed));

                    if PRINT_FRAME_STATS == true {
                        println!("Frame stats - Total steps: {}, Ray Hits: {}, Rays Missed: {}", 
                            total_steps, rays_cast, rays_missed);
                    }
                }

                            
                //println!("Checking for messages in RedrawRequested");
                while let Ok(msg) = self.update_receiver.try_recv() {
                    println!("Received {:?}", msg);
                    if let WorldUpdateMessage::BufferUpdated(new_buffer_index) = msg {
                        println!("Switching buffer to {}", new_buffer_index);
                        self.last_n_press = false;
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

                let present_mode = if USE_VSYNC == true {
                    vulkano::swapchain::PresentMode::Fifo
                } else {
                    vulkano::swapchain::PresentMode::Immediate
                };

                // Recreate everything upon screen resize
                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            image_format: rcx.swapchain.image_format(), 
                            present_mode,
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");
                
                    rcx.swapchain = new_swapchain;
                    rcx.attachment_image_views = window_size_dependent_setup(&new_images);
                    rcx.viewport.extent = window_size.into();
                    rcx.recreate_swapchain = false;
                }


                // Calculate camera buffer variables and set them to the buffer
                let uniform_camera_subbuffer = {
                    let look_from = self.camera_location.location;
                    
                    // Update world when exiting the centre octree
                    let curr_pos_chunk = (look_from / 64.0).floor();
                    let old_pos_chunk = (self.camera_location.old_loc / 64.0).floor();

                    let diff = curr_pos_chunk - old_pos_chunk;



                    if diff.x != 0.0 {
                        //println!("axis: {}, dir: {}", 0, diff.x);
                        self.world_updater.request_update(Update::Shift(0, diff.x as i32));
                    }

                    if diff.z != 0.0 {
                        //println!("axis: {}, dir: {}", 2, diff.z)
                        self.world_updater.request_update(Update::Shift(2, diff.z as i32));
                    }


                    let look_distance = 1.0;
                    let look_at = self.camera_location.location + self.camera_location.direction * look_distance;
                    self.camera_location.old_loc = look_from;


                    let v_up = Vec3 {x: 0.0, y: 1.0, z: 0.0};
                    //let fov = 90;

                    let focal_length = 1.0;
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

                    let t = self.start.elapsed().as_secs_f32();
                    let period = 2.0;

                    let normalized_t = (t % period) / period;
    
                    let s = if normalized_t < 0.5 {
                        normalized_t * 2.0
                    } else {
                        2.0 - normalized_t * 2.0
                    };


                    let time = [
                        t,
                        t.fract(),
                        s,
                        1.0, // Extra Number
                    ];

                    let c: CameraBufferData = CameraBufferData {
                        origin: [look_from.x as f32, look_from.y as f32, look_from.z as f32, 1.0],
                        pixel00_loc: [ pixel00_loc.x as f32, pixel00_loc.y as f32, pixel00_loc.z as f32, 1.0],
                        pixel_delta_u:[pixel_delta_u.x as f32, pixel_delta_u.y as f32, pixel_delta_u.z as f32, 1.0],
                        pixel_delta_v: [pixel_delta_v.x as f32, pixel_delta_v.y as f32, pixel_delta_v.z as f32, 1.0],
                        world_position: [world_position_1.x as f32, world_position_1.y as f32, world_position_1.z as f32, 1.0],
                        sun_position: [self.camera_location.sun_loc.x as f32, self.camera_location.sun_loc.y as f32, self.camera_location.sun_loc.z as f32, 1.0],
                        time: time,
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

                let layout = &rcx.main_pipeline.layout().set_layouts()[0];
                //println!("Layout bindings: {:?}", layout.bindings());

                // Create descriptor set for the buffers
                let descriptor_set = PersistentDescriptorSet::new(
                    &self.descriptor_set_allocator,
                    layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, uniform_camera_subbuffer), 
                        WriteDescriptorSet::buffer(1, self.voxel_buffers[self.current_voxel_buffer].clone()),
                        WriteDescriptorSet::buffer(2, self.world_meta_data_buffers[self.current_voxel_buffer].clone()),
                        WriteDescriptorSet::buffer(3, self.noise_buffer.clone()), 
                        WriteDescriptorSet::buffer(4, self.ray_distance_buffer.clone()), 
                        WriteDescriptorSet::buffer(5, self.stat_buffer.clone()), 
                        WriteDescriptorSet::image_view(6, rcx.attachment_image_views[image_index as usize].clone()),
                    ],
                    [],
                )
                .unwrap();

                // Dispatch sizes, initial beam only works on 1/4th of pixels

                let w_width = window_size.width;
                let w_height = window_size.height;

                // Regular dispatch size
                let full_dispatch_size = [
                    (w_width + 15) / 16,   
                    (w_height + 15) / 16,
                    1
                ];
                
                // 1/4th aka every other pixel for intial beam stage
                let reduced_dispatch_size = [
                    (w_width / 2 + 15) / 16,  
                    (w_height / 2 + 15) / 16, 
                    1
                ];

                // record command buffer
                let mut builder = AutoCommandBufferBuilder::primary(
                    &self.command_buffer_allocator,  // Add & here
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder.fill_buffer(self.stat_buffer.clone(), 0).unwrap();

                if USE_BEAM_OPTIMISATION == true {
                    builder
                        .bind_pipeline_compute(rcx.initial_pipeline.clone())
                        .unwrap()
                        .bind_descriptor_sets(
                            PipelineBindPoint::Compute, 
                            rcx.initial_pipeline.layout().clone(), 
                            0, 
                            descriptor_set.clone()
                        )
                        .unwrap()
                        .dispatch(reduced_dispatch_size)
                        .unwrap();
                }
                
                builder
                    .bind_pipeline_compute(rcx.main_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute, 
                        rcx.main_pipeline.layout().clone(), 
                        0, 
                        descriptor_set 
                    )
                    .unwrap()
                    .dispatch(full_dispatch_size)
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
                
                if MEASURE_FRAME_TIMES == true {
                    let frame_time = frame_start.elapsed();
                    self.frame_times.push(frame_time.as_secs_f64());
                    
                    if PRINT_FRAME_STATS == true {
                        println!("Frame time: {:?}, FPS: {}", frame_time, 1.0 / frame_time.as_secs_f32());
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