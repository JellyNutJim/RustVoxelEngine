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


pub fn create_large_islands(world: &mut ShaderGrid, pos: (u32, u32)) {
    // Update vertical chunk
    let grid = world.noise.generate_grid_from_point(4, 4, 0.0003, (pos.0, pos.1));

    for x in 0..4 {
        for z in 0..4 {
            let mut y = ((grid[x as usize][z as usize] * 16.0).floor()) as i32;

            let x_adjusted = x * 16 + pos.0 as i32;
            let z_adjusted = z * 16 + pos.1 as i32;

            y *= 16;

            if (y <= 0) {
                world.insert_subchunk([x_adjusted, y + (768), z_adjusted], 2, 1);
                
                world.insert_subchunk([x_adjusted, 768 + 16, z_adjusted], 3, 1);
                continue;
            }

            if y >= 32 {
                y = 32;
            }

            world.insert_subchunk([x_adjusted, y + 768, z_adjusted], 1, 1);
        }
    }
}