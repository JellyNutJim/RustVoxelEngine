use super::{ShaderChunk, ShaderGrid};





pub fn generate_hills_and_water(world: &mut ShaderGrid, pos: (u32, u32)) {
    // Update vertical chunk
    let grid = world.noise.generate_grid_from_point(64, 64, 0.005, (pos.0, pos.1));

    for x in 0..64 {
        for z in 0..64 {
            let y = grid[x as usize][z as usize] * 64.0;

            let x_adjusted = x + pos.0 as i32;
            let z_adjusted = z + pos.1  as i32;

            let y = (y + 768.0) as i32;
            if y < 768 {
                let mut temp = y + 1;
                while temp < 769 {
                    world.insert_voxel([x_adjusted, temp, z_adjusted], 3);
                    temp += 1;
                }
            }

            world.insert_voxel([x_adjusted, y, z_adjusted], 1);
            world.insert_voxel([x_adjusted, y - 1, z_adjusted], 2);
            world.insert_voxel([x_adjusted, y - 2, z_adjusted], 2);
            world.insert_voxel([x_adjusted, y - 3, z_adjusted], 2);
        }
    }
}


// Resets heightmap level then adds continent height
pub fn create_smooth_islands(world: &mut ShaderGrid, pos: (u32, u32)) {
    // Update vertical chunk
    let large_scale_sea = world.sea_level - 16;
    let grid = world.noise.generate_grid_from_point(64, 64, 0.00003, (pos.0, pos.1));

    let chunk_pos = (
        pos.0 - world.origin[0] as u32,
        pos.1 - world.origin[2] as u32,
    );

    //println!("{:?}", pos);
    for x in 0..64 {
        for z in 0..64 {
            let complex_y = grid[x as usize][z as usize] * 16.0;
            let x_adj = x + chunk_pos.0 as usize;
            let z_adj = z + chunk_pos.1 as usize; 

            //println!("{} {}", x_adj, z_adj);
            world.height_map[x_adj][z_adj] = world.sea_level as f64 + complex_y * 16.0;

            if !(x % 16 == 0 && z % 16 == 0) {
                continue;
            }

            let mut y = (complex_y.floor() * 16.0) as i32;

            let x_adj = (x + pos.0 as usize) as i32;
            let z_adj = (z + pos.1 as usize) as i32;

            if y <= 0 {
                world.insert_subchunk([x_adj, y + (large_scale_sea as i32), z_adj], 2, 1);
                
                world.insert_subchunk([x_adj, large_scale_sea as i32 + 16, z_adj], 3, 1);
                continue;
            }

            if y >= 32 {
                y = 32;
            }

            world.insert_subchunk([x_adj, y + large_scale_sea as i32, z_adj], 1, 1);
        }
    }
}

pub fn create_hills(world: &mut ShaderGrid, pos: (u32, u32)) {
    // Update vertical chunk
    let grid = world.noise.generate_grid_from_point(64, 64, 0.005, (pos.0, pos.1));

    let chunk_pos = (
        pos.0 - world.origin[0] as u32,
        pos.1 - world.origin[2] as u32,
    );

    println!("{:?}", pos);
    println!("chunk pos: {:?}", chunk_pos);
    for x in 0..64 {
        for z in 0..64 {
            let complex_y = grid[x as usize][z as usize] * 16.0;
            let x_adj = x + chunk_pos.0 as usize;
            let z_adj = z + chunk_pos.1 as usize; 

            //println!("{} {}", x_adj, z_adj);
            //println!("{}", world.height_map[x_adj][z_adj]);

            let mut voxel_type = 1;

            let x_adj_space = (x + pos.0 as usize) as i32;
            let z_adj_space = (z + pos.1 as usize) as i32;

            let distance = (world.sea_level as f64 + 16.0 -  world.height_map[x_adj][z_adj]).abs();
            let fall_off: f64 = 1.0; // Controls how quickly it approaches 0 near 768
            let scaling = 1.0 - (-distance.powf(2.0) / fall_off.powf(2.0)).exp();

            let y = complex_y * 16.0;

            world.height_map[x_adj][z_adj] += (complex_y * scaling);

            if world.height_map[x_adj][z_adj]> (world.sea_level as f64 + 15.9) && world.height_map[x_adj][z_adj] < (world.sea_level as f64 + 16.3) {
                voxel_type = 4;
                world.insert_voxel([x_adj_space, world.height_map[x_adj][z_adj] as i32, z_adj_space], voxel_type);
                continue;
            }

            if world.height_map[x_adj][z_adj] < (world.sea_level as f64 + 15.9){
                world.insert_voxel([x_adj_space, (world.sea_level as f64 + 15.0) as i32, z_adj_space], 3);
                continue;
            }

            //println!("{scaling} {} {}", world.sea_level + 16, world.height_map[x_adj][z_adj]);

            world.insert_voxel([x_adj_space, world.height_map[x_adj][z_adj] as i32, z_adj_space], voxel_type);

        }
    }
}