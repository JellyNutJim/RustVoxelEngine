use super::{HeightMap, PerlinNoise, ShaderGrid, Voxel};

// Contains all noises used throughout the generation process
#[allow(unused)]
#[derive(Debug, Clone)]
pub struct GenPipeLine {
    pub landmass: PerlinNoise,

    // Biome Determination
    pub temperature: PerlinNoise,
    pub humidity: PerlinNoise,

    // Biome Specific Noise
    pub height: PerlinNoise,

    // Extras
    pub interference: PerlinNoise,
    pub features: PerlinNoise,
}

// Useful Functions
impl GenPipeLine {
    pub fn new(seed: u64) -> Self{
        Self {
            landmass: PerlinNoise::new(seed),

            temperature: PerlinNoise::new(seed + 1),
            humidity: PerlinNoise::new(seed - 1),

            height: PerlinNoise::new(seed - 2),

            interference: PerlinNoise::new(seed + 2),
            features: PerlinNoise::new(seed + 3),
        }
    }
}

// Voxel Generation
impl GenPipeLine {
    pub fn get_biome_at(&self) -> f64 {
        32.1
    }
}

// Chunk Generation
impl GenPipeLine {
    pub fn generate_biome_chunk(&self, m: &mut HeightMap) -> f64 {
        *m.get_mut(0, 0) = 32.0;
        m.get(0, 0)
    }
}


// Other Generation



// Generates and inserts one chunk of biome data
pub fn gen_biome(biome_map: &mut HeightMap, x: u32, z: u32) {

}



// Resets heightmap level then adds continent height
pub fn create_smooth_islands(world: &mut ShaderGrid, pos: (u32, u32)) {
    // Update vertical chunk
    //let large_scale_sea = world.sea_level - 16;

    let continent_perlin = world.generator.landmass.generate_grid_from_point(64, 64, 0.00003, (pos.0, pos.1));
    let beach_perlin = world.generator.landmass.generate_grid_from_point(64, 64, 0.005, (pos.0, pos.1));
    let bay_perlin = world.generator.landmass.generate_grid_from_point(64, 64, 0.001, (pos.0, pos.1));

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
            let distance = world.sea_level as f64 + 16.0 -  world.height_map.get(x_adj, z_adj);
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

            let y = ((world.height_map.get(x_adj, z_adj) as i32 - 3) / 4) * 4;

            if world.height_map.get(x_adj, z_adj) < sea_level + 15.9 {
                let x_adj = (x + pos.0 as usize) as i32;
                let z_adj = (z + pos.1 as usize) as i32;

                voxel_type = 3;

                world.insert_subchunk([x_adj, sea_level as i32, z_adj], Voxel::from_type(voxel_type), 1, false);
                continue;
            }

            let x_adj = (x + pos.0 as usize) as i32;
            let z_adj = (z + pos.1 as usize) as i32;

            world.insert_subchunk([x_adj, y, z_adj], Voxel::from_type(voxel_type), 1, true);
        }
    }
}

pub fn create_beach_hills(world: &mut ShaderGrid, pos: (u32, u32)) {

    // Update vertical chunk
    let interference_grid = world.generator.interference.generate_grid_from_point(64, 64, 0.01, (pos.0, pos.1));

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

            let x_adj_space = (x + pos.0 as usize) as i32;
            let z_adj_space = (z + pos.1 as usize) as i32;

            let distance = (world.sea_level as f64 + 16.0 -  world.height_map.get(x_adj, z_adj)).abs();
            let fall_off: f64 = 1.0; 
            let scaling = 1.0 - (-distance.powf(2.0) / fall_off.powf(2.0)).exp();
            *world.height_map.get_mut(x_adj, z_adj) += interference * 0.01 * scaling;

            let mut v = Voxel::new();
            let mut v_type = 1;
            let mut pos = [0.0 as f64, world.height_map.get(x_adj, z_adj).fract(), 0.0 as f64];
            
            
            if world.height_map.get(x_adj, z_adj) > (world.sea_level as f64 + 15.7) {
                if world.height_map.get(x_adj, z_adj) < (world.sea_level as f64 + 16.3) {
                    v_type = 4;
                }
            }

            // Water level
            if world.height_map.get(x_adj, z_adj) < (world.sea_level as f64 + 15.7) {
                world.insert_voxel([x_adj_space, world.sea_level as i32 + 15, z_adj_space], Voxel::from_type(3), false);
                continue
            }

            // Set up sub voxels
            for i in 0..1 {
                for j in 0..1 {
                    pos[0] = i as f64 * 0.5;
                    pos[2] = j as f64 * 0.5;

                    // temp to visualise difference

                    pos[1] += 0.01;

                    let octant = get_voxel_octant(pos);

                    if octant > 3 {
                        v.set_octant(octant - 4, 2);
                    }

                    v.set_octant(octant, v_type);
                }   
            }

            v.update_voxel();

            world.insert_voxel([x_adj_space, world.height_map.get(x_adj, z_adj) as i32, z_adj_space], v, true);

        }
    }
}

// Assuming pos within - 0 - 1
fn get_voxel_octant(pos: [f64; 3]) -> usize {
    
    let mut octant = 0;
    let mid = 0.5;
    
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
