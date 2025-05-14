use std::{f64::consts::PI, u8};

use crate::types::SteepFourHeightSurface;

use super::{BiomeMap, Biome, HeightMap, PerlinNoise, ScalablePerlin, OctreeGrid, FourHeightSurface, Voxel, Geometry};

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
    biome_map: BiomeMap,
    sea_level: f64,
}

// Useful Functions
impl GenPipeLine {
    pub fn new(seed: u64, width: usize, sea_level: f64) -> Self{
        Self {
            landmass: PerlinNoise::new(seed, 0.00003),

            temperature: PerlinNoise::new(seed + 1, 0.0003),
            humidity: PerlinNoise::new(seed - 1, 0.03),

            height: ScalablePerlin::new(seed - 2),

            interference: ScalablePerlin::new(seed + 2),
            features: ScalablePerlin::new(seed + 3),

            height_map: HeightMap::new(sea_level, width * 64 * 2), // Double the grid size to account for sub voxels
            biome_map: BiomeMap::new(Biome::Single(0),width * 64), // Biomes placed per voxel
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

    // 1.0 -> Flat
    // 2.0 -> Hills
    // 3.0 -> Mountain
    // 4.0 -> River

    // This system is jank. Ideally biome would've been able to store a varaible amount of biomes per data entry, allowing for a simpler data
    // structure that could've allowed for more terain varieties. I never got the chance to fix this
    fn get_biome_at(&self, x: f64, z: f64, y: f64) -> Biome {
        let temp = self.temperature.get_noise_at_point(x, z).abs() * 256.0;
        let humid = self.humidity.get_noise_at_point(x, z).abs() * 256.0;

        //println!("{} {}", temp, y);

        //return 1.0;
        // Flat Lands

        if y < 10360.0 {
            if temp > 60.0 {
                let t = temp - 60.1;
            
                let elevation_factor = if y < 10330.0 {
                    1.0
                } else {
                    ((10360.0 - y) as f64) / 30.0
                };

                let t_adjusted = t * elevation_factor.clamp(0.0, 1.0);
                
                let dif = t_adjusted / 20.0;
                Biome::Double(1, 2, dif)
            } else {
                Biome::Single(1)         
            }
        }
        else {
            if y < 10420.0 {

                    let t;
                    t = y - 10360.1;
                    let dif = t / 60.0;
                    Biome::Double(1, 3, dif)
            }
            else {
                if temp > 60.0 {
                    let t;
                    t = temp - 60.1;
                    let dif = t / 20.0;

                    Biome::Double(2, 3, dif)
                }
                else {
                    Biome::Single(3)
                }
            }
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

    fn get_mountain_noise(&self, x: f64, z: f64) -> f64 { 
        self.height.get_noise_at_point(x, z, 0.0003) * 32.0 * 32.0 + self.interference.get_noise_at_point(x, z, 0.005) * 16.0
    }

    fn get_interference(&self, x: f64, z: f64) -> f64 {
        self.interference.get_noise_at_point(x, z, 0.5) * 16.0 * 16.0

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

    fn generate_hills(&mut self, x: f64, z: f64, map_coords: (usize, usize), update: bool) -> (u32, f64) {

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

    fn generate_mountains (&mut self, x: f64, z: f64, map_coords: (usize, usize), update: bool) -> (u32, f64) {

        let change  = self.get_mountain_noise(x, z).abs();
        let height = self.height_map.get_mut(map_coords.0, map_coords.1);

        let distance = (change - *height).abs();
        let fall_off: f64 = 0.1;
        let scaling = 1.0 - (-distance.powf(2.0) / fall_off.powf(2.0)).exp();

        let updated_height = *height + change; 

        if update == true {
            *height = updated_height;
        }

        (1, updated_height)
    }


    fn apply_interference(&mut self, x: f64, z: f64, map_coords: (usize, usize)) -> f64 {
        let change = self.get_interference(x, z);
        let height = self.height_map.get_mut(map_coords.0, map_coords.1);

        *height += change * 0.01;

        *height
    }

    fn get_biome_height(&mut self, biome_id: u8, x: f64, z: f64, map_coords: (usize, usize), update: bool) -> (u32, f64) {
        match biome_id {
            1 => { self.generate_beach(x, z, map_coords, update) }
            2 => { self.generate_hills(x, z, map_coords, update) }
            3 => { self.generate_mountains(x, z, map_coords, update) }
            _ => { self.generate_beach(x, z, map_coords, update) }
        } 
    }

    // Applies biome height, does not account for biome merging
    fn apply_full_biome_height(&mut self, x_pos: f64, z_pos: f64, height_map_coord: (usize, usize), biome_map_coord: (usize, usize)) -> (u32, f64) {

        let b = self.biome_map.get(biome_map_coord.0, biome_map_coord.1);
        let res: (u32, f64);

        match b {
            Biome::Single(b) => {
                res = self.get_biome_height(b, x_pos, z_pos, height_map_coord, true);
            }
            Biome::Double(b1, b2, mut p) => {

                let h1 =  self.get_biome_height(b1, x_pos, z_pos, height_map_coord, true);
                let h2 =  self.get_biome_height(b2, x_pos, z_pos, height_map_coord, true);


                let mut height = h1.1;


                if b2 == 3 {
                    if p > 2.0 {
                        p = 1.0
                    }
                }
                
                if p > 1.0 {
                    p = 1.0;
                }

                // Linear
                //let height = h1.1 * (1.0 - p) + h2.1 * p;

                // Smooth step
                //let height = h1.1 * (1.0 - (p * p * (3.0 - 2.0 * p))) + h2.1 * (p * p * (3.0 - 2.0 * p));

                // Cosine 
                let height = h1.1 * (1.0 - ((1.0 - (p * PI).cos()) * 0.5)) + h2.1 * ((1.0 - (p * PI).cos()) * 0.5);

                // Cubic
                // let cubic_p = p * p * (3.0 - 2.0 * p);
                // let height = h1.1 * (1.0 - cubic_p) + h2.1 * cubic_p;                

                self.height_map.set(height_map_coord.0, height_map_coord.1, height);
                
                //let height = self.height_map.get(height_map_coord.0, height_map_coord.1);
                res = (h1.0, height);
            }
            Biome::Triple(b1, b2, b3, p1, p2) => {
                let h1 =  self.get_biome_height(b1, x_pos, z_pos, height_map_coord, true);
                let h2 =  self.get_biome_height(b2, x_pos, z_pos, height_map_coord, true);
                let h3 =  self.get_biome_height(b3, x_pos, z_pos, height_map_coord, true);


                let height = h1.1 * (1.0 - p1) + h2.1 * p1;
                let height = height * (1.0 - p2) + h3.1 * p2;

                self.height_map.set(height_map_coord.0, height_map_coord.1, height);
                res = (h1.0, height);
            }
        }

        res
    }   
}

// Height map application functions, combines different height levels to produce more varies terrain
impl GenPipeLine {

    fn res_8_map_gen_line(&mut self, x_pos: u32, z_pos: u32, map_coords: (usize, usize), axis: usize) {
        let map_coords_2 = (map_coords.0 * 2, map_coords.1 * 2);

        // let row = if axis == 2 {
        //     ((0, 8), (7, 8))
        // } else {
        //     ((7, 8), (0, 8))
        // };

        let row = ((0, 8), (0, 8));

        for x in row.0.0..row.0.1 {
            for z in row.1.0..row.1.1 {
                // World position
                let x_adj = (x_pos + x * 8) as f64;
                let z_adj = (z_pos + z * 8) as f64;

                let x_usize = x as usize;
                let z_usize = z as usize;

                // Biome mapping
                let map_x = map_coords.0 + x_usize * 8;
                let map_z = map_coords.1 + z_usize * 8;

                // Heightmap mapping
                let height_map_x = map_coords_2.0 + x_usize * 16;
                let height_map_z = map_coords_2.1 + z_usize * 16;

                let height = self.get_landmass_noise(x_adj, z_adj);

                self.height_map.set(
                    height_map_x,
                    height_map_z,
                    height,
                );

                // Insert new biome data -> determined after base heightmap gen so it can be used for biome processing
                self.biome_map.set(map_x, map_z, self.get_biome_at(x_adj, z_adj, height));

                // Update height to represent biome type
                self.apply_full_biome_height(x_adj, z_adj, (height_map_x, height_map_z), (map_x, map_z));
            }
        }

    }

    // Populates the biome map, and the heightmap for sized 8 voxels (1/4 of the total biome and height data)
    fn res_8_map_gen(&mut self, x_pos: u32, z_pos: u32, map_coords: (usize, usize)) {
        let map_coords_2 = (map_coords.0 * 2, map_coords.1 * 2);

        for x in 0..8 {
            for z in 0..8 {

                // World position
                let x_adj = (x_pos + x * 8) as f64;
                let z_adj = (z_pos + z * 8) as f64;

                let x_usize = x as usize;
                let z_usize = z as usize;

                // Biome mapping
                let map_x = map_coords.0 + x_usize * 8;
                let map_z = map_coords.1 + z_usize * 8;

                // Heightmap mapping
                let height_map_x = map_coords_2.0 + x_usize * 16;
                let height_map_z = map_coords_2.1 + z_usize * 16;

                let height = self.get_landmass_noise(x_adj, z_adj);

                self.height_map.set(
                    height_map_x,
                    height_map_z,
                    height,
                );

                // Insert new biome data -> determined after base heightmap gen so it can be used for biome processing
                self.biome_map.set(map_x, map_z, self.get_biome_at(x_adj, z_adj, height));

                // Update height to represent biome type
                self.apply_full_biome_height(x_adj, z_adj, (height_map_x, height_map_z), (map_x, map_z));
            }
        }
    }

    // Assumes 1/4th has already been populated by res_8_map_gen, populates the other 3/4 of data
    fn res_4_map_gen(&mut self, x_pos: u32, z_pos: u32, map_coords: (usize, usize), edge: bool) {
        let map_coords_2 = (map_coords.0 * 2, map_coords.1 * 2);

        let max = if edge {
            72
        } else {
            64
        }; 

        for x in 0..max {
            for z in 0..max {

                // Skip already processed world positions
                if x % 8 == 0 && z % 8 == 0 {
                    continue;
                }

                let x_adj = (x_pos + x) as f64;
                let z_adj = (z_pos + z) as f64;

                let x_usize = x as usize;
                let z_usize = z as usize;

                // Biome mapping
                let map_x = map_coords.0 + x_usize;
                let map_z = map_coords.1 + z_usize;

                // Heightmap mapping
                let height_map_x = map_coords_2.0 + x_usize * 2;
                let height_map_z = map_coords_2.1 + z_usize * 2;
    
                // set base heightmap values, x/y * 2 as heightmap stores 4x the data it needs to at this level
                let height = self.get_landmass_noise(x_adj, z_adj);
                self.height_map.set(
                    height_map_x,
                    height_map_z,
                    height,
                );

                self.biome_map.set(map_x, map_z, self.get_biome_at(x_adj, z_adj, height));
                self.apply_full_biome_height(x_adj, z_adj, (height_map_x, height_map_z), (map_x, map_z));
            }
        }
    }

    // No longer used
    // Populates 100% of the biome and heightmap data, currently unused. Maybe useful for smaller render distances where memory must be conserved.
    fn new_chunk_map_gen(&mut self, x_pos: u32, z_pos: u32, map_coords: (usize, usize)) {
        // Update Heightmap
        for x in 0..64 {
            for z in 0..64 {
                let x_adj = (x_pos + x) as f64;
                let z_adj = (z_pos + z) as f64;

                let height = self.get_landmass_noise(x_adj, z_adj);

                // set base heightmap values
                self.height_map.set(
                    map_coords.0 * 2 + ((x * 2) as f64 ) as usize,
                    map_coords.1 * 2 + ((z * 2) as f64 ) as usize,
                    height,
                );
    
                // Insert new biome data
                self.biome_map.set(map_coords.0 + x as usize, map_coords.1 + z as usize, self.get_biome_at(x_adj, z_adj, height));
    

            }
        }
    }

    fn get_full_biome_height(&self, map_x: usize, map_z: usize) -> f64 {
        self.height_map.get(map_x, map_z)
    }

}


//static POSITIONS: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
static POSITIONS_EXLCUDING: [(usize, usize); 3] = [(0, 1), (1, 0), (1, 1)];

// Visible API, 

pub fn gen_res_8_map_line(world: &mut OctreeGrid, x_pos: u32, z_pos: u32, axis: usize) {
    let map_c = world.generator.get_map_coords(x_pos, z_pos, world.origin);

    world.generator.res_8_map_gen_line(x_pos, z_pos, map_c, axis);
}

pub fn generate_res_8_maps(world: &mut OctreeGrid, x_pos: u32, z_pos: u32) {
    let map_c = world.generator.get_map_coords(x_pos, z_pos, world.origin);
    world.generator.res_8_map_gen(x_pos, z_pos, map_c);
}

// Intialises the base heightmap, intialises the base biome map, inserts 16x16 voxels
pub fn generate_res_8(world: &mut OctreeGrid, x_pos: u32, z_pos: u32) {
    let mut map_c = world.generator.get_map_coords(x_pos, z_pos, world.origin);
    let sea_height = 8.0 + world.generator.sea_level;


    map_c.0 *= 2;
    map_c.1 *= 2;

    for x in 0..8 {
        for z in 0..8 {
            let x_adj = x_pos + x * 8;
            let z_adj = z_pos + z * 8;

            let c0 = map_c.0 + (x * 16) as usize;
            let c1 = map_c.1 + (z * 16) as usize;

            if c0 == 41072 || c1 == 41072 {
                continue
            }

            let geometry = generate_four_height_surfaces(world, 8, 8.0, 8, sea_height, (c0, c1));

            for (geom, height) in geometry {
                world.insert_subchunk([
                        x_adj as i32,
                        height as i32,
                        z_adj as i32,
                    ], 
                    geom, 
                    2, 
                    false
                );
            }

        }
    }
}

pub fn generate_res_4_maps(world: &mut OctreeGrid, x_pos: u32, z_pos: u32, edge: bool)  {
    let map_c = world.generator.get_map_coords(x_pos, z_pos, world.origin);
    world.generator.res_4_map_gen(x_pos, z_pos, map_c, edge);
}

// Essentially a higher resolution of gen 8, accept it generates all required base biome and heightmap data, not 1/8th or 1/4th but only inserts size 4 voxels
pub fn generate_res_4(world: &mut OctreeGrid, x_pos: u32, z_pos: u32) {
    let mut map_c = world.generator.get_map_coords(x_pos, z_pos, world.origin);
    let sea_height = 12.0 + world.generator.sea_level;
    
    map_c.0 *= 2;
    map_c.1 *= 2;


    for x in 0..16 {
        for z in 0..16 {
            let x_adj = x_pos + x * 4;
            let z_adj = z_pos + z * 4;

            let c0 =  map_c.0 + (x * 8) as usize;
            let c1 =  map_c.1 + (z * 8) as usize;

            let geometry = generate_four_height_surfaces(world, 4, 4.0, 4, sea_height, (c0, c1));

            for (geom, height) in geometry {
                world.insert_subchunk([
                        x_adj as i32,
                        height as i32,
                        z_adj as i32,
                    ], 
                    geom, 
                    3, 
                    false
                );
            }
        }
    }


}

// Shows a higher resolution, and applys biome merging
pub fn generate_res_2(world: &mut OctreeGrid, x_pos: u32, z_pos: u32) {
    let mut map_c = world.generator.get_map_coords(x_pos, z_pos, world.origin);
    let sea_height = 14.0 + world.generator.sea_level;

    map_c.0 *= 2;
    map_c.1 *= 2;

    for x in 0..32 {
        for z in 0..32 {

            let x_adj = x_pos + x * 2;
            let z_adj = z_pos + z * 2;

            let c0 =  map_c.0 + (x * 4) as usize;
            let c1 =  map_c.1 + (z * 4) as usize;

            let geometry = generate_four_height_surfaces(world, 2, 2.0, 2, sea_height, (c0, c1));

            for (geom, height) in geometry {
                world.insert_subchunk([
                        x_adj as i32,
                        height as i32,
                        z_adj as i32,
                    ], 
                    geom, 
                    4, 
                    false
                );
            }
        }
    }
}

pub fn generate_res_1(world: &mut OctreeGrid, x_pos: u32, z_pos: u32) {
    let mut map_c = world.generator.get_map_coords(x_pos, z_pos, world.origin);
    let sea_height = 15.0 + world.generator.sea_level;
    map_c.0 *= 2;
    map_c.1 *= 2;

    for x in 0..64 {
        for z in 0..64 {

            let c0 =  map_c.0 + (x * 2) as usize;
            let c1 =  map_c.1 + (z * 2) as usize;

            let x_adj = x_pos + x;
            let z_adj = z_pos + z;

            let geometry = generate_four_height_surfaces(world, 1, 1.0, 1, sea_height, (c0, c1));

            for (geom, height) in geometry {
                world.insert_geometry([
                        x_adj as i32,
                        height as i32,
                        z_adj as i32,
                    ], 
                    geom,  
                    false
                );
            }
        }
    }
}

use rand::{Rng, SeedableRng};


// let tree = world.generator.get_interference((x_pos + x) as f64, (z_pos + z) as f64);

// if y > 16.5 + world.sea_level as f64 && tree > 120.0 {
//     insert_tree(world, x_adj, y as u32, z_adj);
// }\

fn get_quad_height(y: f64) -> u8 {
    let y = y.fract();
    (y * 254.0).floor() as u8 + 1
}

fn get_leveled_quad_height(y: f64, level: f64, scale: f64) -> u8 {
    let y = y - level;
    ((y / scale) * 254.0).floor() as u8 + 1
}

fn generate_4_height_voxel(world: &OctreeGrid, v_len: &mut usize, voxels: &mut [(f64, FourHeightSurface); 4], map_pos: (usize, usize)) {
    let mut octant_height;

    // Get height values and then sort them 
    let mut positions = vec![(0, 0, world.generator.height_map.get(map_pos.0, map_pos.1))]; // x, z, height

    // Fill remaining posistion
    for &(i, j) in &POSITIONS_EXLCUDING {
        let height = world.generator.height_map.get(map_pos.0 + i * 2, map_pos.1 + j * 2);
        positions.push((i, j, height));
    }

    // Highest first so that they can be looped over once
    positions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut default: [u8; 4] = [0, 0, 0, 0];

    // Loop over other 3 quadrants
    for &(i, j, y) in &positions{ 
        let octant = i + j * 2;
        let y_level = y.trunc();

        octant_height = get_quad_height(y);
        let mut exists = false;
        
        // If new height lies within an existing voxel, insert, else set that voxels quadrant to 255 or 0 (full/empty)
        for k in 0..*v_len {
            if y_level == voxels[k].0 {
                voxels[k].1.set_quadrant(octant, octant_height);
                exists = true;
                break;
            } 
        }

        if !exists {
            let mut temp_v = FourHeightSurface::from(default);
            temp_v.set_quadrant(octant, octant_height);
            voxels[*v_len] = (y_level, temp_v);
            *v_len += 1;
        }

        // Update default, as we know any other voxels will be the same or lower
        default[octant] = 255;
    }
}

fn generate_4_height_leveled_voxel(world: &OctreeGrid, v_len: &mut usize, voxels: &mut [(f64, FourHeightSurface); 4], scale: usize, map_pos: (usize, usize)) {
    if map_pos.0 == 41072 || map_pos.1 == 41072 {
        return;
    }

    let mut octant_height;
    let scale_f64 = scale as f64;

    // Get height values and then sort them 
    let mut positions = vec![(
        0, 
        0, 
        world.generator.height_map.get(map_pos.0, map_pos.1),
    )]; 

    // Fill remaining posistion
    for &(i, j) in &POSITIONS_EXLCUDING {
        let height = world.generator.height_map.get(map_pos.0 + i * scale * 2, map_pos.1 + j * scale * 2);
        positions.push((i, j, height));
    }

    // Highest first so that they can be looped over once
    positions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut default: [u8; 4] = [0, 0, 0, 0];

    // Loop over other 3 quadrants
    for &(i, j, y) in &positions{ 
        let octant = i + j * 2;
        let y_level = ((y) / scale_f64).floor() * scale_f64;

        octant_height = get_leveled_quad_height(y, y_level, scale_f64);

        let mut exists = false;
        
        // If new height lies within an existing voxel, insert, else set that voxels quadrant to 255 or 0 (full/empty)
        for k in 0..*v_len {
            if y_level == voxels[k].0 {
                voxels[k].1.set_quadrant(octant, octant_height);
                exists = true;
                break;
            } 
        }

        if !exists {
            let mut temp_v = FourHeightSurface::from(default);
            temp_v.set_quadrant(octant, octant_height);
            voxels[*v_len] = (y_level, temp_v);
            *v_len += 1;
        }

        // Update default, as we know any other voxels will be the same or lower
        default[octant] = 255;
    }
}

fn generate_four_height_surfaces(world: &OctreeGrid,  scale: usize, scale_f64: f64, scale_u32: u32, sea_height: f64, map_pos: (usize, usize)) -> Vec<(Geometry, u32)> {
    let mut g_vec: Vec<(Geometry, u32)> = Vec::new();

    // Get height values and then sort them 
    let v_hs = [
        world.generator.height_map.get(map_pos.0, map_pos.1),
        world.generator.height_map.get(map_pos.0 + scale * 2, map_pos.1),
        world.generator.height_map.get(map_pos.0, map_pos.1 + scale * 2),
        world.generator.height_map.get(map_pos.0 + scale * 2, map_pos.1 + scale * 2),
    ]; 

    // Order height values to get max and min
    let mut ordered = v_hs.clone();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate start and end voxels
    let start_height_floor = (ordered[0].trunc() / scale_f64).floor() * scale_f64;
    let end_voxel = (ordered[3].trunc() / scale_f64) as u32 * scale_u32;
    let start_voxel = start_height_floor as u32;

    // If max is below sea height, place a water voxel at sea height
    if ordered[3] < sea_height {
        g_vec.push(
            ( Geometry::Voxel(Voxel::from(3)), sea_height as u32)
        );
    }  

    let v_hs_u32 = [
        ((((v_hs[0] - start_height_floor)) * 255.0) / scale_f64) as u32,
        ((((v_hs[1] - start_height_floor)) * 255.0) / scale_f64) as u32,
        ((((v_hs[2] - start_height_floor)) * 255.0) / scale_f64) as u32,
        ((((v_hs[3] - start_height_floor)) * 255.0) / scale_f64) as u32,
    ];

    // If highest and lowest componenet are within the same voxel, use fourheightsurface
    if start_voxel == end_voxel {
        // Calc heights

        if start_height_floor == sea_height {
            g_vec.push((Geometry::FourHeightSurface(FourHeightSurface::from_u32_water([v_hs_u32[0], v_hs_u32[1], v_hs_u32[2], v_hs_u32[3]])), start_voxel));
            return g_vec;
        }
        g_vec.push((Geometry::FourHeightSurface(FourHeightSurface::from_u32([v_hs_u32[0], v_hs_u32[1], v_hs_u32[2], v_hs_u32[3]])), start_voxel));
        return g_vec;
    }

    let range = ((end_voxel - start_voxel) / scale_u32) + 1;

    for i in 0..range {
        let height = start_voxel + (i * scale_u32);

        if height == sea_height as u32 {
            g_vec.push(
                (
                    Geometry::SteepFourHeightSurface(SteepFourHeightSurface::from_water_level([v_hs_u32[0], v_hs_u32[1], v_hs_u32[2], v_hs_u32[3]], i)),
                    start_voxel + (i * scale_u32)
                )
            );
        } else {
            g_vec.push(
                (
                    Geometry::SteepFourHeightSurface(SteepFourHeightSurface::from([v_hs_u32[0], v_hs_u32[1], v_hs_u32[2], v_hs_u32[3]], i)),
                    start_voxel + (i * scale_u32)
                )
            );
        }
    }

    g_vec
}


// fn insert_tree(world: &mut OctreeGrid, x: u32, y: u32, z: u32) {
//     let mut rng = rand::rngs::StdRng::seed_from_u64(world.seed + x as u64 + z as u64);
//     let height = rng.random_range(5..10);

//     for i in 0..height {
//         world.insert_geometry([
//             x as i32,
//             (y + i) as i32,
//             z as i32,
//         ], 
//         Geometry::Voxel(Voxel::from(2)),  
//         false
//         );
//     }

//     world.insert_geometry([
//         x as i32,
//         (y + height) as i32,
//         z as i32,
//     ], 
//     Geometry::Voxel(Voxel::from(2)),  
//     false
//     );

//     for i in 0..4 {

//     }

// }

    
// // Assuming pos within - 0 - 1
// fn get_voxel_octant(pos: [f64; 3]) -> usize {
    
//     let mut octant = 0;
//     let mid = 0.5;
    
//     if pos[0] > mid {
//         octant += 1;
//     }

//     if pos[1] > mid {
//         octant += 4;
//     }

//     if pos[2] > mid {
//         octant += 2;
//     }
    
//     octant
// }

// fn get_vertical_octant(y: f64) -> usize {
//     if y > 0.5 {
//         4
//     } else  {
//         0
//     }
// }