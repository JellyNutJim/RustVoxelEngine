use super::{HeightMap, PerlinNoise, ScalablePerlin, ShaderChunk, ShaderGrid, Voxel};

// Contains all noises used throughout the generation process
#[allow(unused)]
#[derive(Debug, Clone)]
pub struct GenPipeLine {

    // Set Scale Perlin
    pub landmass: PerlinNoise,
    pub temperature: PerlinNoise,
    pub humidity: PerlinNoise,

    // Biome Specific Noise
    pub height: ScalablePerlin,

    // Extras
    pub interference: ScalablePerlin,
    pub features: ScalablePerlin,

    // Meta Data
    pub height_map: HeightMap,
    pub biome_map: HeightMap,
}

// Useful Functions
impl GenPipeLine {
    pub fn new(seed: u64, width: usize, sea_level: f64) -> Self{
        Self {
            landmass: PerlinNoise::new(seed, 0.00003),

            temperature: PerlinNoise::new(seed + 1, 0.003),
            humidity: PerlinNoise::new(seed - 1, 0.003),

            height: ScalablePerlin::new(seed - 2),

            interference: ScalablePerlin::new(seed + 2),
            features: ScalablePerlin::new(seed + 3),

            height_map: HeightMap::new(sea_level, width * 64 * 2), // Double the grid size to account for sub voxels
            biome_map: HeightMap::new(0.0,width * 64) // Biomes placed per voxel
        }
    }
}

// Voxel Generation
impl GenPipeLine {
    fn is_continent(&self, x: u32, z: u32) -> bool {
        //self.landmass
        false
    }

    fn get_biome_at(&self, x: u32, z: u32) -> f64 {
        23.0
    }
}

// Chunk Generation
impl GenPipeLine {
    pub fn generate_biome_chunk(&mut self, m: &mut Vec<ShaderChunk>) -> f64 {
        self.get_biome_at(32, 32);
        32.0
    }
}


// Base Biome Heightmap Generation


// Multiresolution Context Based Generation

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
            *world.generator.height_map.get_mut(x_adj, z_adj) = world.sea_level as f64 + continent_noise;

            // Beach change
            let change = (beach_noise + bay_noise) / 2.0;
            let initial = world.generator.height_map.get(x_adj, z_adj);

            // Calculate scaling factor based on sea level
            let distance = world.sea_level as f64 + 16.0 -  world.generator.height_map.get(x_adj, z_adj);
            let fall_off: f64 = 2.0;
            let scaling: f64;
            let distance = distance.abs();
            scaling = 1.0 - (-distance.powf(2.0) / fall_off.powf(2.0)).exp();

            if change > 0.0 {
                *world.generator.height_map.get_mut(x_adj, z_adj) += change * scaling;
            }
            else {
                *world.generator.height_map.get_mut(x_adj, z_adj) += change;
            }

            if initial < sea_level + 15.9 || world.generator.height_map.get(x_adj, z_adj) < sea_level + 15.9 {
                
                if initial < world.generator.height_map.get(x_adj, z_adj) {
                    *world.generator.height_map.get_mut(x_adj, z_adj) = initial;
                }
            }

            // Update low resolution terrain
            if !(x % 16 == 0 && z % 16 == 0) {
                continue;
            }

            let y = ((world.generator.height_map.get(x_adj, z_adj) as i32 - 3) / 4) * 4;

            if world.generator.height_map.get(x_adj, z_adj) < sea_level + 15.9 {
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

            let distance = (world.sea_level as f64 + 16.0 -  world.generator.height_map.get(x_adj, z_adj)).abs();
            let fall_off: f64 = 1.0; 
            let scaling = 1.0 - (-distance.powf(2.0) / fall_off.powf(2.0)).exp();
            *world.generator.height_map.get_mut(x_adj, z_adj) += interference * 0.01 * scaling;

            let mut v = Voxel::new();
            let mut v_type = 1;
            let mut pos = [0.0 as f64, world.generator.height_map.get(x_adj, z_adj).fract(), 0.0 as f64];
            
            
            if world.generator.height_map.get(x_adj, z_adj) > (world.sea_level as f64 + 15.7) {
                if world.generator.height_map.get(x_adj, z_adj) < (world.sea_level as f64 + 16.3) {
                    v_type = 4;
                }
            }

            // Water level
            if world.generator.height_map.get(x_adj, z_adj) < (world.sea_level as f64 + 15.7) {
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

            world.insert_voxel([x_adj_space, world.generator.height_map.get(x_adj, z_adj) as i32, z_adj_space], v, true);

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
