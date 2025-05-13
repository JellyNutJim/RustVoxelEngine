use std::{sync::{Arc, Mutex}, thread, time::Instant, fs::{OpenOptions, File}, io::{self, Write}, path::Path};
use crossbeam_channel::{unbounded, Sender, Receiver};
use chrono::Local;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, 
        AutoCommandBufferBuilder, 
        CommandBufferUsage,
        CopyBufferInfo,
        PrimaryCommandBufferAbstract  // Needed for execute()
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::{GpuFuture, Sharing},
};


use crate::{get_grid_from_seed, types::Geometry, OctreeGrid, Voxel};

#[derive(Debug)]
pub enum WorldUpdateMessage {
    UpdateWorld(Update),
    BufferUpdated(usize, [i32; 3], bool, usize, i32),
    Shutdown,
}

#[derive(Debug)]
#[allow(unused)]
pub enum Update {
    AddVoxel(i32, i32, i32, u32),
    Shift(usize, i32),
    SwitchSeed(u64),
}

#[allow(dead_code)]
pub struct WorldUpdater {
    sender: Sender<WorldUpdateMessage>,
    update_thread: Option<thread::JoinHandle<()>>,
    world: Arc<Mutex<OctreeGrid>>,
}

impl WorldUpdater {
    pub fn new(
        device: Arc<Device>,
        transfer_queue: Arc<Queue>,
        #[allow(unused)]
        compute_queue: Arc<Queue>,
        voxel_buffers: [Subbuffer<[u32]>; 2],
        world_meta_data_buffers: [Subbuffer<[i32]>; 2],
        intial_world: OctreeGrid,
    ) -> (Self, Receiver<WorldUpdateMessage>) {
        // Create a single channel pair
        let (command_tx, command_rx) = unbounded(); 
        let (update_tx, update_rx) = unbounded(); 

        // Conver world to mutex
        let world_mutex = Arc::new(Mutex::new(intial_world));
        let thread_world = world_mutex.clone();
    
        let update_thread = Some(thread::spawn(move || {

            let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
            let command_buffer_allocator = StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            );
            
            let mut shutdown = false;
            let mut current_buffer = 0;

            let mut persistent_voxel_buff: Vec<u32> = Vec::with_capacity(500000000);
            persistent_voxel_buff.resize(500000000, 0);

            let mut persistent_meta_buff: Vec<i32> = Vec::with_capacity(40000000);
            persistent_meta_buff.resize(40000000, 0);


            let voxel_transfer_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    sharing: Sharing::Exclusive,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                persistent_voxel_buff.clone(),
            ).unwrap();

            let meta_transfer_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    sharing: Sharing::Exclusive,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE 
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                persistent_meta_buff,
            ).unwrap();
    
            while !shutdown {
                match command_rx.recv() {  // Use rx_worker instead of tx_clone
                    Ok(WorldUpdateMessage::UpdateWorld(update)) => {
                        println!("Received");

                        // Lock world
                        let mut world = thread_world.lock().unwrap();
                        // Get next buffer index
                        let next_buffer = (current_buffer + 1) % 2;

                        // let gen_time;
                        // let stage_time;
                        // let flatten_time;
                        // let copy_time;
                        
                        let i = Instant::now();
                        match update {
                            Update::AddVoxel(x, y, z , t) => {
                                println!("Adding type: {} at: {} {} {}", t, x, y, z);

                                world.insert_geometry([x,y,z], Geometry::Voxel(Voxel::from(2)), false); 

                                world.update_flat_chunk_with_world_pos([x, y, z]);
                            }
                            Update::SwitchSeed(seed) => {
                                // DOES NOT CURRENTLY WORK NEEDS PLAYER LOCATION
                                *world = get_grid_from_seed(seed, 321, [(321*64)/2, (321*64)/2,(321*64)/2]);
                                world.flatten_world();
                            }
                            Update::Shift(axis, dir) => {
                                // Shift origin

                                let i = Instant::now();
                                world.shift(axis, dir);
                                println!("Generation Time {}", i.elapsed().as_millis());

                            }
                        };
                        
                        let generation_time = i.elapsed().as_millis();

                        // Generate new world data
                        //let mut rng = rand::rng();

                        let flat_world = world.get_flat_world();
                        println!("LENGTH: {}", flat_world.1.len());
                        
                        let i = Instant::now();
                        {
                            let mut mapped_ptr = voxel_transfer_buffer.write().unwrap();
                            mapped_ptr[..flat_world.1.len()].copy_from_slice(&flat_world.1);
                        }

                        {
                            let mut mapped_meta = meta_transfer_buffer.write().unwrap();
                            mapped_meta[..flat_world.0.len()].copy_from_slice(&flat_world.0);
                        }

                        let stage_prep_time = i.elapsed().as_millis();

                        let i = Instant::now();

                        let mut builder = AutoCommandBufferBuilder::primary(
                            &command_buffer_allocator,
                            transfer_queue.queue_family_index(),
                            CommandBufferUsage::OneTimeSubmit,
                        ).unwrap();
                
                        builder
                            .copy_buffer(CopyBufferInfo::buffers(
                                voxel_transfer_buffer.clone(),
                                voxel_buffers[next_buffer].clone(),
                            ))
                            .unwrap()
                            .copy_buffer(CopyBufferInfo::buffers(
                                meta_transfer_buffer.clone(),
                                world_meta_data_buffers[next_buffer].clone(),
                            ))
                            .unwrap();
                
                        let command_buffer = builder.build().unwrap();

                            
                        // Execute and wait for completion
                        command_buffer.execute(transfer_queue.clone())
                            .unwrap()
                            .then_signal_fence_and_flush()
                            .unwrap()
                            .wait(None )
                            .unwrap();

                        let copy_time = i.elapsed().as_millis();
                            
                        // Update current buffer and notify main thread
                        current_buffer = next_buffer;

                        // if let Err(e) = append_to_csv("times.csv", generation_time, stage_prep_time, copy_time) {
                        //     println!("Failed to write to CSV: {:?}", e);
                        // }


                        if let Err(_e) = update_tx.send(
                            match update {
                                Update::Shift(axis, dir) => { WorldUpdateMessage::BufferUpdated(next_buffer, world.origin, true, axis, dir) },
                                _ => { WorldUpdateMessage::BufferUpdated(next_buffer, world.origin, false, 0, 0) }
                            }

                            ) {
                            println!("Failed");
                            shutdown = true; // Exit if we can't send messages
                        } else {
                            println!("Success");
                        }
                        
                    },
                    Ok(WorldUpdateMessage::Shutdown) => {
                        println!("Shutdown");
                        shutdown = true;
                    },
                    Ok(_) => {
                        println!("Recevied");
                    },
                    Err(e) => {
                        println!("Error {:?}", e);
                        shutdown = true;
                    }
                }
            }
            println!("Worker thread shutting down");
        }));
    
        (
            WorldUpdater {
                sender: command_tx,
                update_thread,
                world: world_mutex,
            },
            update_rx  // Return the original receiver for the main thread
        )
    }

    pub fn request_update(&self, u: Update) {
        println!("Updating world");
        if let Err(_e) = self.sender.send(WorldUpdateMessage::UpdateWorld(u)) {
            println!("Thread is Dead");
        }
    }
}

impl Drop for WorldUpdater {
    fn drop(&mut self) {
        if let Err(e) = self.sender.send(WorldUpdateMessage::Shutdown) {
            println!("Failed to send shutdown message: {:?}", e);
        }
        if let Some(handle) = self.update_thread.take() {
            handle.join().unwrap();
        }
    }
}

fn append_to_csv(filename: &str, gen_time: u128, stage_time: u128, copy_time: u128) -> io::Result<()> {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    
    let file_exists = Path::new(filename).exists();
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(filename)?;
    
    if !file_exists {
        writeln!(file, "timestamp,seed,gen_time,stage_time,copy_time")?;
    }
    
    writeln!(file, "{},{},{},{},{}", timestamp, 1330, gen_time, stage_time, copy_time)?;
    
    Ok(())
}
