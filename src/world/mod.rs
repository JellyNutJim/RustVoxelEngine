// Shader Usage
mod shader_chunk;
mod shader_grid;
mod chunk_generator;
mod height_map;

// Rust usage
mod chunk;

pub use shader_chunk::ShaderChunk;
pub use shader_grid::ShaderGrid;
pub use chunk_generator::*;
pub use height_map::HeightMap;


use crate::{
    noise_gen::{PerlinNoise, ScalablePerlin}, 
    Voxel
};

// VOXEL TYPES
// 0 - air
// 1 - grass
// 2 - stone
// 3 - water
// 4 - sand


// WIDTH MUST ALWAYS BE ODD
pub fn get_grid_from_seed(seed: u64, width: i32, camera_origin: [i32; 3]) -> ShaderGrid {

    let x = Voxel::from_type(1);

    println!("{:032b}", x.get_voxel());



    //let mut p = ShaderGrid::from(chunks, 3);
    let camera_chunk = [(camera_origin[0] / 64) * 64, (camera_origin[1] / 64) * 64, (camera_origin[2] / 64) * 64];

    let origin = [camera_chunk[0] - ((width - 1) * 64) / 2 as i32,  camera_chunk[1] - ((width - 1) * 64) / 2 as i32, camera_chunk[2] - ((width - 1) * 64) / 2 as i32];

    println!("camera: {:?}", camera_origin);
    println!("camera chunk: {:?}", camera_chunk);
    println!("grid origin: {:?}", origin);

    let mut p = ShaderGrid::new(width as u32, origin, seed, 768);


    // Intial World Builder Pipelin
    create_intial_world_with_continents(&mut p);
    create_intial_close_land(&mut p);

    println!("space midpoint: {:?}", p.get_centre_point_in_space()); 
    println!("xz midpoint: {:?}", p.get_x_z_midpoint_in_space()); 


    // let p = p.flatten();
    // println!("Grid: {}", p.1.len());
    p
}


// Intial World Gen Helper Functions

// Layer 1 
fn create_intial_world_with_continents(world: &mut ShaderGrid) {
    for x in 0..world.width {
        for z in 0..world.width {
            let c_x = x * 64 + world.origin[0] as u32;
            let c_z = z * 64 + world.origin[2] as u32;

            let b = true;

            generate_res_16(world, c_x, c_z, b);
        }
    }
}

// Layer 2
fn create_intial_close_land(world: &mut ShaderGrid) {
    let layer_width = 51;
    let layer_half_width = layer_width / 2;


    let mid_xz = world.get_x_z_midpoint_in_space();
    let mid_chunk = [(mid_xz[0] / 64) * 64, (mid_xz[1] / 64) * 64];
    let starting_point = [
        mid_chunk[0] - layer_half_width * 64,
        mid_chunk[1] - layer_half_width * 64
    ];

    for x in 0..layer_width{
        for z in 0..layer_width {
            let c_x = x * 64 + starting_point[0] as u32;
            let c_z = z * 64 + starting_point[1] as u32;

            generate_res_1(world, c_x, c_z);
        }
    }
}