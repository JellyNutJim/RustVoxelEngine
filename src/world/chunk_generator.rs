use super::ShaderGrid;

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

    let continent_perlin = world.noise.generate_grid_from_point(64, 64, 0.00003, (pos.0, pos.1));
    let beach_perlin = world.noise.generate_grid_from_point(64, 64, 0.005, (pos.0, pos.1));
    let bay_perlin = world.noise.generate_grid_from_point(64, 64, 0.001, (pos.0, pos.1));

    let chunk_pos = (
        pos.0 - world.origin[0] as u32,
        pos.1 - world.origin[2] as u32,
    );

    let sea_level = world.sea_level as f64;

    //println!("{:?}", pos);
    for x in 0..64 {
        for z in 0..64 {
            // Get noise at pos
            let continent_noise = continent_perlin[x as usize][z as usize] * 16.0 * 16.0 *1.1;
            let beach_noise = beach_perlin[x as usize][z as usize] * 16.0;
            let bay_noise = bay_perlin[x as usize][z as usize] * 16.0 * 16.0;
            
            let x_adj = x + chunk_pos.0 as usize;
            let z_adj = z + chunk_pos.1 as usize; 

            let mut voxel_type = 1;

            // Set intial height to continent height
            *world.height_map.get_mut(x_adj, z_adj) = world.sea_level as f64 + continent_noise;

            // Beach change
            let change = (beach_noise + bay_noise) / 2.0;
            let initial = world.height_map.get(x_adj, z_adj);

            // Calculate scaling factor based on sea level
            let distance = (world.sea_level as f64 + 16.0 -  world.height_map.get(x_adj, z_adj));
            let fall_off: f64 = 2.0;
            let scaling: f64;
            let distance = distance.abs();
            scaling = 1.0 - (-distance.powf(2.0) / fall_off.powf(2.0)).exp();

            if change > 0.0 {
                *world.height_map.get_mut(x_adj, z_adj) += change * scaling;
            }
            else {
                *world.height_map.get_mut(x_adj, z_adj) += change;
            }

            if initial < sea_level + 15.9 || world.height_map.get(x_adj, z_adj) < sea_level + 15.9 {
                
                if initial < world.height_map.get(x_adj, z_adj) {
                    *world.height_map.get_mut(x_adj, z_adj) = initial;
                }
            }

            // Update low resolution terrain
            if !(x % 16 == 0 && z % 16 == 0) {
                continue;
            }

            let mut y = ((world.height_map.get(x_adj, z_adj) as i32 - 3) / 4) * 4;

            if world.height_map.get(x_adj, z_adj) < sea_level + 15.9 {
                let x_adj = (x + pos.0 as usize) as i32;
                let z_adj = (z + pos.1 as usize) as i32;

                voxel_type = 3;

                world.insert_subchunk([x_adj, sea_level as i32, z_adj], voxel_type, 1);
                continue;
            }

            let x_adj = (x + pos.0 as usize) as i32;
            let z_adj = (z + pos.1 as usize) as i32;

            world.insert_subchunk([x_adj, y, z_adj], voxel_type, 1);
        }
    }
}

pub fn create_beach_hills(world: &mut ShaderGrid, pos: (u32, u32)) {

    // Update vertical chunk
    let interference_grid = world.noise.generate_grid_from_point(64, 64, 0.01, (pos.0, pos.1));

    let chunk_pos = (
        pos.0 - world.origin[0] as u32,
        pos.1 - world.origin[2] as u32,
    );

    //println!("{:?}", pos);
    //println!("chunk pos: {:?}", chunk_pos);
    for x in 0..64 {
        for z in 0..64 {

            // Interference
            let interference = interference_grid[x as usize][z as usize] * 16.0 * 16.0;

            let x_adj = x + chunk_pos.0 as usize;
            let z_adj = z + chunk_pos.1 as usize; 

            let mut voxel_type = 1;

            let x_adj_space = (x + pos.0 as usize) as i32;
            let z_adj_space = (z + pos.1 as usize) as i32;

            let distance = (world.sea_level as f64 + 16.0 -  world.height_map.get(x_adj, z_adj)).abs();
            let fall_off: f64 = 1.0; 
            let scaling = 1.0 - (-distance.powf(2.0) / fall_off.powf(2.0)).exp();
            *world.height_map.get_mut(x_adj, z_adj) += interference * 0.01 * scaling;

            if world.height_map.get(x_adj, z_adj) > (world.sea_level as f64 + 15.7) {
                if world.height_map.get(x_adj, z_adj) < (world.sea_level as f64 + 16.3) {
                    voxel_type = 4;
                }
            }

            if world.height_map.get(x_adj, z_adj) < (world.sea_level as f64 + 15.7){
                voxel_type = 3;
                world.insert_voxel([x_adj_space, world.sea_level as i32 + 15, z_adj_space], voxel_type);
                continue
            }

            world.insert_voxel([x_adj_space, world.height_map.get(x_adj, z_adj) as i32, z_adj_space], voxel_type);

        }
    }
}