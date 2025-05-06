// Shader Usage
mod octree;
mod octree_grid;
mod chunk_generator;
mod height_map;
mod biome_map;

pub use octree::Octree;
pub use octree_grid::OctreeGrid;
pub use chunk_generator::*;
pub use height_map::HeightMap;
pub use biome_map::{BiomeMap, Biome};


use crate::{
    noise_gen::{PerlinNoise, ScalablePerlin}, 
    Voxel, FourHeightSurface, Geometry,
};

// VOXEL TYPES
// 0 - air
// 1 - grass
// 2 - stone
// 3 - water
// 4 - sand


// WIDTH MUST ALWAYS BE ODD
pub fn get_grid_from_seed(seed: u64, width: i32, camera_origin: [i32; 3]) -> OctreeGrid {
    let r8w: u32 = width as u32;
    let r4w: u32 = 101;
    let r2w: u32 = 51;
    let r1w: u32 = 31;

    //let mut p = ShaderGrid::from(chunks, 3);
    let camera_chunk = [(camera_origin[0] / 64) * 64, (camera_origin[1] / 64) * 64, (camera_origin[2] / 64) * 64];
    let origin = [camera_chunk[0] - ((width - 1) * 64) / 2 as i32,  camera_chunk[1] - ((width - 1) * 64) / 2 as i32, camera_chunk[2] - ((width - 1) * 64) / 2 as i32];

    println!("camera: {:?}", camera_origin);
    println!("camera chunk: {:?}", camera_chunk);
    println!("grid origin: {:?}", origin);

    let mut p = OctreeGrid::new(width as u32, origin, seed, 816, r8w, r4w, r2w, r1w);


    // Intial World Builder Pipelin
    create_res_8_land(&mut p);
    create_res_4_land(&mut p);

    create_intial_merged_land(&mut p);
    create_inner_land(&mut p);

    println!("space midpoint: {:?}", p.get_centre_point_in_space()); 
    println!("xz midpoint: {:?}", p.get_x_z_midpoint_in_space()); 


    // let p = p.flatten();
    // println!("Grid: {}", p.1.len());
    p
}

pub fn get_empty_grid(width: i32, camera_origin: [i32; 3]) -> OctreeGrid {
    let r8w: u32 = width as u32;
    let r4w: u32 = 151;
    let r2w: u32 = 101;
    let r1w: u32 = 51;

    //let mut p = ShaderGrid::from(chunks, 3);
    let camera_chunk = [(camera_origin[0] / 64) * 64, (camera_origin[1] / 64) * 64, (camera_origin[2] / 64) * 64];
    let origin = [camera_chunk[0] - ((width - 1) * 64) / 2 as i32,  camera_chunk[1] - ((width - 1) * 64) / 2 as i32, camera_chunk[2] - ((width - 1) * 64) / 2 as i32];

    println!("camera: {:?}", camera_origin);
    println!("camera chunk: {:?}", camera_chunk);
    println!("grid origin: {:?}", origin);

    let mut grid = OctreeGrid::new(width as u32, origin, 42, 816, r8w, r4w, r2w, r1w);

    // Set all generation levels to voxel depth
    for x in 0..grid.width {
        for y in 0..grid.width {
            for z in 0..grid.width { 
                let grid_index = x + y * grid.width + z * grid.width.pow(2);
                let index = grid.spatial_map[grid_index as usize] as usize;
                grid.trees[index].set_generation_level(5);
            }
        }
    }

    grid
}

fn get_octant_at_depth(pos: [u32; 3], depth: u32) -> u32 {
    //println!("{:?} {}", pos, depth);

    let octant_size = 64 / (2u32.pow(depth));

    let mid = (octant_size / 2) - 1;
    let pos = [pos[0] % octant_size, pos[1] % octant_size, pos[2] % octant_size];
    let mut octant: u32 = 0;
   
    if pos[0] > mid {
        octant += 1;
    }

    if pos[1] > mid {
        octant += 4;
    }

    if pos[2] > mid {
        octant += 2;
    }
    
    octant
}

pub fn create_octree_map() -> Vec<u32> {
    let mut octree_map: Vec<u32> = Vec::with_capacity(262144);
    
    for z in 0..64 {
        for y in 0..64 {
            for x in 0..64 {
                let mut octants: u32 = 0;
                for i in 0..6 {
                    octants = octants | (get_octant_at_depth([x, y, z], i as u32) << (3 * i));
                }
                octree_map.push(octants);
            }
        }
    }

    octree_map
}


// Intial World Gen Helper Functions

// Layer 1 
fn create_res_8_land(world: &mut OctreeGrid) {
    for x in 0..world.width {
        for z in 0..world.width {
            let c_x = x * 64 + world.origin[0] as u32;
            let c_z = z * 64 + world.origin[2] as u32;
            generate_res_8_maps(world, c_x, c_z, 4);
        }
    }


    for x in 0..world.width {
        for z in 0..world.width {
            let c_x = x * 64 + world.origin[0] as u32;
            let c_z = z * 64 + world.origin[2] as u32;

            for y in 0..world.width { 
                let grid_index = x + (y) * world.width  + z * world.width.pow(2);
                let index = world.spatial_map[grid_index as usize] as usize;
                world.trees[index].set_generation_level(2);
            }

            generate_res_8(world, c_x, c_z);
        }
    }
}

fn create_res_4_land(world: &mut OctreeGrid) {
    let layer_width = world.get_res_4_width();
    let layer_half_width = layer_width / 2;


    let mid_xz = world.get_x_z_midpoint_in_space();
    let mid_chunk = [(mid_xz[0] / 64) * 64, (mid_xz[1] / 64) * 64];
    let starting_point = [
        mid_chunk[0] - layer_half_width * 64,
        mid_chunk[1] - layer_half_width * 64
    ];

    let sp_c = world.get_chunk_pos_u32(&[starting_point[0], 0 , starting_point[1]]);

    for x in 0..layer_width{
        for z in 0..layer_width {
            let c_x = x * 64 + starting_point[0] as u32;
            let c_z = z * 64 + starting_point[1] as u32;
            generate_res_4_maps(world, c_x, c_z, true);
        }
    }

    for x in 0..layer_width{
        for z in 0..layer_width {
            let c_x = x * 64 + starting_point[0] as u32;
            let c_z = z * 64 + starting_point[1] as u32;

            for y in 0..world.width { 
                let grid_index = (sp_c[0] + x) + (y) * world.width  + (sp_c[2] + z) * world.width.pow(2);
                let index = world.spatial_map[grid_index as usize] as usize;
                world.trees[index].set_generation_level(3);
            }

            generate_res_4(world, c_x, c_z);
        }
    }
}

// Second Inner Most
fn create_intial_merged_land(world: &mut OctreeGrid) {
    let layer_width = world.get_res_2_width();
    let layer_half_width = layer_width / 2;


    let mid_xz = world.get_x_z_midpoint_in_space();
    let mid_chunk = [(mid_xz[0] / 64) * 64, (mid_xz[1] / 64) * 64];
    let starting_point = [
        mid_chunk[0] - layer_half_width * 64,
        mid_chunk[1] - layer_half_width * 64
    ];

    let sp_c = world.get_chunk_pos_u32(&[starting_point[0], 0 , starting_point[1]]);

    for x in 0..layer_width{
        for z in 0..layer_width {
            let c_x = x * 64 + starting_point[0] as u32;
            let c_z = z * 64 + starting_point[1] as u32;

            for y in 0..world.width { 
                let grid_index = (sp_c[0] + x) + (y) * world.width  + (sp_c[2] + z) * world.width.pow(2);
                let index = world.spatial_map[grid_index as usize] as usize;
                world.trees[index] = Octree::new(world.trees[index].get_origin());
                world.trees[index].set_generation_level(4);
            }

            generate_res_2(world, c_x, c_z);
        }
    }
}


// InnerMost
fn create_inner_land(world: &mut OctreeGrid) {
    let layer_width = world.get_res_1_width();
    let layer_half_width = layer_width / 2;

    let mid_xz = world.get_x_z_midpoint_in_space();
    let mid_chunk = [(mid_xz[0] / 64) * 64, (mid_xz[1] / 64) * 64];
    let starting_point = [
        mid_chunk[0] - layer_half_width * 64,
        mid_chunk[1] - layer_half_width * 64
    ];

    let sp_c = world.get_chunk_pos_u32(&[starting_point[0], 0 , starting_point[1]]);

    for x in 0..layer_width{
        for z in 0..layer_width {
            let c_x = x * 64 + starting_point[0] as u32;
            let c_z = z * 64 + starting_point[1] as u32;


            for y in 0..world.width { 
                let grid_index = (sp_c[0] + x) + (y) * world.width  + (sp_c[2] + z) * world.width.pow(2);
                let index = world.spatial_map[grid_index as usize] as usize;
                world.trees[index] = Octree::new(world.trees[index].get_origin());
                world.trees[index].set_generation_level(5);
            }

            generate_res_1(world, c_x, c_z);
        }
    }
}