use std::{os::windows::io::BorrowedHandle, u8};

use super::{HeightMap, PerlinNoise, ScalablePerlin, ShaderChunk, ShaderGrid, Voxel};

// Contains all noises used throughout the generation process
#[allow(unused)]
#[derive(Debug, Clone)]
pub struct GenPipeLine {

    // Set Scale Perlin
    pub landmass: PerlinNoise,
    temperature: PerlinNoise,
    humidity: PerlinNoise,

    // Biome Specific Noise
    height: ScalablePerlin,

    // Extras
    interference: ScalablePerlin,
    features: ScalablePerlin,

    // Meta Data
    height_map: HeightMap,
    biome_map: HeightMap,
    sea_level: f64,
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
            biome_map: HeightMap::new(0.0,width * 64), // Biomes placed per voxel
            sea_level: sea_level,
        }
    }

    pub fn get_map_coords(&self, x: u32, z: u32, origin: [i32; 3]) -> (usize, usize){
        (
            (x - origin[0] as u32) as usize,
            (z - origin[2] as u32) as usize,
        )
    }

    pub fn shift_maps(&mut self, axis: usize, dir: i32) {
        self.height_map.shift(axis, dir, 128);
        self.biome_map.shift(axis, dir, 64);
    }
}

// Voxel Generation
impl GenPipeLine {
    fn is_continent(&self, x: f64, z: f64) -> bool {
        if self.landmass.get_noise_at_point(x, z) > 0.0 {
            true
        } else {
            false
        }
    }

    fn get_biome_at(&self, x: f64, z: f64) -> f64 {
        // Biome Pipelin
        if self.landmass.get_noise_at_point(x, z) > -16.0 {
            1.0
        } else {
            0.0
        }
    }
}

// Noise functions return a height, how this is applied will be based on height map application functions
impl GenPipeLine {

    // Get Continent height
    fn get_landmass_noise(&self, x: f64, z: f64) -> f64 {
        self.sea_level + self.landmass.get_noise_at_point(x, z) * 16.0 * 16.0 * 1.1
    }

    fn get_beach_noise(&self, x: f64, z: f64) -> f64 {
        self.height.get_noise_at_point(x, z, 0.005) * 16.0
    }

    fn get_beach_with_bays_noise(&self, x: f64, z: f64) -> f64 {
        (
            self.height.get_noise_at_point(x, z, 0.005) * 16.0 +
            self.height.get_noise_at_point(x, z, 0.001) * 16.0 * 16.0 
        ) / 2.0
    }
}

// Complex Biome Height Generators
impl GenPipeLine {
    fn generate_beach(&mut self, x: f64, z: f64, map_coords: (usize, usize), update: bool) -> (u32, f64) {
        let change = self.get_beach_noise(x, z);
        let height = self.height_map.get_mut(map_coords.0, map_coords.1);
        let original = *height;

        let distance = self.sea_level + 16.0 - *height;

        let mut fall_off: f64 = 3.5;
        if original > self.sea_level && change < 0.0{
            fall_off = 6.0;
        }

        if original < self.sea_level + 15.0 {
            fall_off = 10.0;
        }


        let distance = distance.abs();
        let scaling = 1.0 - (-distance.powf(2.0) / fall_off.powf(2.0)).exp();

        let updated_height = *height + change * scaling;
        
        if update == true {
            *height = updated_height;
        }

        if *height < 16.6 + self.sea_level || original < self.sea_level + 16.0{
            return (4, updated_height)
        }

        (1, updated_height)
    }

    fn generate_bays(&mut self, x: f64, z: f64, map_coords: (usize, usize), update: bool) -> (u32, f64) {

        let change  = self.get_beach_with_bays_noise(x, z);
        let height = self.height_map.get_mut(map_coords.0, map_coords.1);

        let distance = self.sea_level - *height;

        let mut fall_off: f64 = 2.0;

        if change > 0.0 {
            fall_off = 4.0;
        }

        let distance = distance.abs();
        let scaling = 1.0 - (-distance.powf(2.0) / fall_off.powf(2.0)).exp();

        let updated_height = *height + change * scaling;

        if update == true {
            *height = updated_height;
        }

        if *height < 16.1 + self.sea_level  {
            return (4, updated_height)
        }

        (1, updated_height)
    }       
}

// Height map application functions, combines different height levels to produce more varies terrain
impl GenPipeLine {

    // Generates Intial heightmap based on landmass height, and calculates biome positions
    fn new_chunk_map_gen(&mut self, x_pos: u32, z_pos: u32, map_coords: (usize, usize)) {
        // Update Heightmap
        for x in 0..64 {
            for z in 0..64 {
                let mut x_adj = (x_pos + x) as f64;
                let mut z_adj = (z_pos + z) as f64;
    
                // Insert new biome data
                self.biome_map.set(map_coords.0 + x as usize, map_coords.1 + z as usize, self.get_biome_at(x_adj, z_adj));
    
                // set base heightmap values
                for i in 0..1 {
                    for j in 0..1 {
                        x_adj += 0.5 * i as f64;
                        z_adj += 0.5 * j as f64;
    
                        // temp to visualise difference
                        self.height_map.set(
                            map_coords.0 * 2 + ((x * 2) as f64 + 1.0 * i as f64) as usize,
                            map_coords.1 * 2 + ((z * 2) as f64 + 1.0 * j as f64) as usize,
                            self.get_landmass_noise(x_adj, z_adj),
                        );
                    }   
                }
            }
        }
    }

    fn apply_full_biome_height(&mut self, x_pos: u32, z_pos: u32, map_coords: (usize, usize), update: bool) -> (u32, f64) {

        // Select correct biomes (for now just beach)
        self.generate_beach(
            (x_pos) as f64, 
            (z_pos) as f64, 
            map_coords,
            update,
        )
    }

    fn apply_biome_merging() {

    }

}

// Visible API, 

// Intialises the base heightmap, intialises the base biome map, inserts 16x16 voxels
pub fn generate_res_16(world: &mut ShaderGrid, x_pos: u32, z_pos: u32, insert: bool) {
    let mut map_c = world.generator.get_map_coords(x_pos, z_pos, world.origin);

    // Update intial biome and heightmap
    world.generator.new_chunk_map_gen(x_pos, z_pos, map_c);

    if insert == false {
        return;
    }

    map_c.0 *= 2;
    map_c.1 *= 2;

    for x in 0..8 {
        for z in 0..8 {
            let x_adj = x * 8;
            let z_adj = z * 8;

            let map_x = map_c.0 + (x_adj * 2);
            let map_z = map_c.1 + (z_adj * 2);

            let (mut voxel_type, mut y) = world.generator.apply_full_biome_height(x_pos + x_adj as u32, z_pos + z_adj as u32, (map_x, map_z), false);

            if y < 15.7 + world.generator.sea_level {
                voxel_type = 3;
                y = world.generator.sea_level;
            }

            let y = ((y as i32 - 3) / 8) * 8;

            let v = Voxel::from_type(voxel_type as u8);

            world.insert_subchunk([
                    (x_pos + x_adj as u32) as i32,
                    y,
                    (z_pos + z_adj as u32) as i32,
                ], 
                v, 2, true
            );

            
        }
    }
}

// pub fn generate_res_8(&mut self) {

// }

// pub fn generate_res_4(&mut self) {

// }

// pub fn generate_res_2(&mut self) {

// }

pub fn generate_res_1(world: &mut ShaderGrid, x_pos: u32, z_pos: u32) {
    let mut map_c = world.generator.get_map_coords(x_pos, z_pos, world.origin);
    map_c.0 *= 2;
    map_c.1 *= 2;
    // Apply biomes

    for x in 0..64 {
        for z in 0..64 {

            let c0 =  map_c.0 + (x * 2) as usize;
            let c1 =  map_c.1 + (z * 2) as usize;

            let x_adj = x_pos + x;
            let z_adj = z_pos + z;

            //let y = world.generator.get_landmass_noise((x_pos + x) as f64, (z_pos + z) as f64);
            let (voxel_type, y) = world.generator.apply_full_biome_height(
                x_adj, 
                z_adj,
        (
                        c0,
                        c1,
                    ),
                    true
                );
                
            
            //let y = world.generator.height_map.get(c0, c1);

            let v = Voxel::from_type(voxel_type as u8);

            world.insert_voxel([
                    x_adj as i32,
                    y as i32,
                    z_adj as i32,
                ], 
                v,  
                true
            );

            if y < 15.7 + world.generator.sea_level {
                world.insert_voxel([
                        x_adj as i32,
                        (world.generator.sea_level + 15.0) as i32,
                        z_adj  as i32,
                    ], 
                    Voxel::from_type(3),  
                    false
                );
            }

        }
    }
}
   

    

//Resets heightmap level then adds continent height
// pub fn create_smooth_islands(world: &mut ShaderGrid, pos: (u32, u32)) {
//     // Update vertical chunk
//     //let large_scale_sea = world.sea_level - 16;

//     let continent_perlin = world.generator.landmass.generate_grid_from_point(64, 64, 0.00003, (pos.0, pos.1));
//     let beach_perlin = world.generator.landmass.generate_grid_from_point(64, 64, 0.005, (pos.0, pos.1));
//     let bay_perlin = world.generator.landmass.generate_grid_from_point(64, 64, 0.001, (pos.0, pos.1));

//     let chunk_pos = (
//         pos.0 - world.origin[0] as u32,
//         pos.1 - world.origin[2] as u32,
//     );

//     let sea_level = world.sea_level as f64;

//     //println!("{:?}", pos);
//     for x in 0..64 {
//         for z in 0..64 {
//             // Get noise at pos
//             let continent_noise = continent_perlin[x as usize][z as usize] * 16.0 * 16.0 *1.1;
//             let beach_noise = beach_perlin[x as usize][z as usize] * 16.0;
//             let bay_noise = bay_perlin[x as usize][z as usize] * 16.0 * 16.0;
            
//             let x_adj = x + chunk_pos.0 as usize;
//             let z_adj = z + chunk_pos.1 as usize; 

//             let mut voxel_type = 1;

//             // Set intial height to continent height
//             *world.generator.height_map.get_mut(x_adj, z_adj) = world.sea_level as f64 + continent_noise;

//             // Beach change
//             let change = (beach_noise + bay_noise) / 2.0;
//             let initial = world.generator.height_map.get(x_adj, z_adj);

//             // Calculate scaling factor based on sea level
//             let distance = world.sea_level as f64 + 16.0 -  world.generator.height_map.get(x_adj, z_adj);
//             let fall_off: f64 = 2.0;
//             let scaling: f64;
//             let distance = distance.abs();
//             scaling = 1.0 - (-distance.powf(2.0) / fall_off.powf(2.0)).exp();

//             if change > 0.0 {
//                 *world.generator.height_map.get_mut(x_adj, z_adj) += change * scaling;
//             }
//             else {
//                 *world.generator.height_map.get_mut(x_adj, z_adj) += change;
//             }

//             if initial < sea_level + 15.9 || world.generator.height_map.get(x_adj, z_adj) < sea_level + 15.9 {
                
//                 if initial < world.generator.height_map.get(x_adj, z_adj) {
//                     *world.generator.height_map.get_mut(x_adj, z_adj) = initial;
//                 }
//             }

//             // Update low resolution terrain
//             if !(x % 16 == 0 && z % 16 == 0) {
//                 continue;
//             }

//             let y = ((world.generator.height_map.get(x_adj, z_adj) as i32 - 3) / 4) * 4;

//             if world.generator.height_map.get(x_adj, z_adj) < sea_level + 15.9 {
//                 let x_adj = (x + pos.0 as usize) as i32;
//                 let z_adj = (z + pos.1 as usize) as i32;

//                 voxel_type = 3;

//                 world.insert_subchunk([x_adj, sea_level as i32, z_adj], Voxel::from_type(voxel_type), 1, false);
//                 continue;
//             }

//             let x_adj = (x + pos.0 as usize) as i32;
//             let z_adj = (z + pos.1 as usize) as i32;

//             world.insert_subchunk([x_adj, y, z_adj], Voxel::from_type(voxel_type), 1, true);
//         }
//     }
// }

// pub fn create_beach_hills(world: &mut ShaderGrid, pos: (u32, u32)) {

//     // Update vertical chunk
//     let interference_grid = world.generator.interference.generate_grid_from_point(64, 64, 0.01, (pos.0, pos.1));

//     let chunk_pos = (
//         pos.0 - world.origin[0] as u32,
//         pos.1 - world.origin[2] as u32,
//     );

//     //println!("{:?}", pos);
//     //println!("chunk pos: {:?}", chunk_pos);
//     for x in 0..64 {
//         for z in 0..64 {

//             // Interference
//             let interference = interference_grid[x as usize][z as usize] * 16.0 * 16.0;

//             let x_adj = x + chunk_pos.0 as usize;
//             let z_adj = z + chunk_pos.1 as usize; 

//             let x_adj_space = (x + pos.0 as usize) as i32;
//             let z_adj_space = (z + pos.1 as usize) as i32;

//             let distance = (world.sea_level as f64 + 16.0 -  world.generator.height_map.get(x_adj, z_adj)).abs();
//             let fall_off: f64 = 1.0; 
//             let scaling = 1.0 - (-distance.powf(2.0) / fall_off.powf(2.0)).exp();
//             *world.generator.height_map.get_mut(x_adj, z_adj) += interference * 0.01 * scaling;

//             let mut v = Voxel::new();
//             let mut v_type = 1;
//             let mut pos = [0.0 as f64, world.generator.height_map.get(x_adj, z_adj).fract(), 0.0 as f64];
            
            
//             if world.generator.height_map.get(x_adj, z_adj) > (world.sea_level as f64 + 15.7) {
//                 if world.generator.height_map.get(x_adj, z_adj) < (world.sea_level as f64 + 16.3) {
//                     v_type = 4;
//                 }
//             }

//             // Water level
//             if world.generator.height_map.get(x_adj, z_adj) < (world.sea_level as f64 + 15.7) {
//                 world.insert_voxel([x_adj_space, world.sea_level as i32 + 15, z_adj_space], Voxel::from_type(3), false);
//                 continue
//             }

//             // Set up sub voxels
//             for i in 0..1 {
//                 for j in 0..1 {
//                     pos[0] = i as f64 * 0.5;
//                     pos[2] = j as f64 * 0.5;

//                     // temp to visualise difference

//                     pos[1] += 0.01;

//                     let octant = get_voxel_octant(pos);

//                     if octant > 3 {
//                         v.set_octant(octant - 4, 2);
//                     }

//                     v.set_octant(octant, v_type);
//                 }   
//             }

//             v.update_voxel();

//             world.insert_voxel([x_adj_space, world.generator.height_map.get(x_adj, z_adj) as i32, z_adj_space], v, true);

//         }
//     }
// }

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
