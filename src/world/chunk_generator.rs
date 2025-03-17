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

    fn get_interference(&self, x: f64, z: f64) -> f64 {
        let i = self.interference.get_noise_at_point(x, z, 0.5) * 16.0 * 16.0;

        if i < 0.0 {
            0.0
        } else {
            i
        }
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

    fn generate_fjords(&mut self, x: f64, z: f64, map_coords: (usize, usize), update: bool) -> (u32, f64) {

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

    fn apply_interference(&mut self, x: f64, z: f64, map_coords: (usize, usize)) -> f64 {
        let change = self.get_interference(x, z);
        let height = self.height_map.get_mut(map_coords.0, map_coords.1);

        *height += change * 0.01;

        *height
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
                for i in 0..2 {
                    for j in 0..2 {
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

    fn apply_full_biome_height(&mut self, x_pos: f64, z_pos: f64, map_coords: (usize, usize), update: bool) -> (u32, f64) {

        // Select correct biomes (for now just beach)
        let res = self.generate_fjords(
            x_pos, 
            z_pos, 
            map_coords,
            update,
        );

        res
    }   

    fn apply_biome_merging() {

    }

}

// Visible API, 

// Intialises the base heightmap, intialises the base biome map, inserts 16x16 voxels
pub fn generate_res_8(world: &mut ShaderGrid, x_pos: u32, z_pos: u32, insert: bool) {
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

            let (mut voxel_type, mut y) = world.generator.apply_full_biome_height((x_pos + x_adj as u32) as f64, (z_pos + z_adj as u32) as f64, (map_x, map_z), false);

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
static POSITIONS: [(usize, usize); 3] = [(0, 1), (1, 0), (1, 1)];

pub fn generate_res_2(world: &mut ShaderGrid, x_pos: u32, z_pos: u32) {
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

            for &(i, j) in &POSITIONS {
                world.generator.apply_full_biome_height(
                    x_adj as f64 + 0.5 * i as f64, 
                    z_adj as f64 + 0.5 * j as f64,
                    ( 
                        c0 + i,
                        c1 + j,
                    ),
                    true
                );
            }
    

            let (voxel_type, y) = world.generator.apply_full_biome_height(
                x_adj as f64, 
                z_adj as f64,
                (
                    c0,
                    c1,
                ),
                true
            );
                
            if x % 2 != 0 {
                continue;
            }

            if z % 2 != 0 {
                continue;
            }

            let y = ((y - 1.0) / 2.0) * 2.0;
            let v = Voxel::from_type(voxel_type as u8);

            world.insert_subchunk([
                    x_adj as i32,
                    y as i32,
                    z_adj as i32,
                ], 
                v,  
                4,
                true
            );

            if y < 15.7 + world.generator.sea_level {
                world.insert_subchunk([
                        x_adj as i32,
                        (world.generator.sea_level + 15.0) as i32,
                        z_adj  as i32,
                    ], 
                    Voxel::from_type(3), 
                    4, 
                    false
                );
            }

        }
    }
}

pub fn generate_res_1(world: &mut ShaderGrid, x_pos: u32, z_pos: u32) {
    let mut map_c = world.generator.get_map_coords(x_pos, z_pos, world.origin);
    map_c.0 *= 2;
    map_c.1 *= 2;
    // Apply biomes

    let mut voxels: [(f64, Voxel); 4] = Default::default();

    //let tvp = [54731, 10175]; // 792 ish
    let tvp = [54732 , 10639];

    for x in 0..64 {
        for z in 0..64 {

            let c0 =  map_c.0 + (x * 2) as usize;
            let c1 =  map_c.1 + (z * 2) as usize;

            let x_adj = x_pos + x;
            let z_adj = z_pos + z;

            let mut voxel_type;

            // Get first octant pre loop
            let mut y = world.generator.height_map.get(c0, c1);

            voxel_type = get_quad_height(y);

            let mut v = Voxel::new();
            let mut v_len = 1;

            v.set_surface_octant(0, voxel_type);

            if x_adj == tvp[0] && z_adj == tvp[1] {
                println!("y: 0 {}", y);  
            }
            
            voxels[0] = (y.trunc(), v);

            // Loop over other 3 quadrants
            for &(i, j) in &POSITIONS { 
                y = world.generator.height_map.get(c0 + i * 2, c1 + j * 2);
                let octant = i + j * 2;

                voxel_type = get_quad_height(y);

                if x_adj == tvp[0] && z_adj == tvp[1] {
                    println!("y: 1 {}", y);  
                }

                let mut exists = false;
                

                // If new height lies within an existing voxel, insert, else set that voxels quadrant to 255 or 0 (full/empty)
                 for k in 0..v_len {
                    let temp = y.trunc();
                    if temp == voxels[k].0 {
                        voxels[k].1.set_surface_octant(octant, voxel_type);
                        exists = true;
                        break;
                    } 
                    else if temp > voxels[k].0 {
                        voxels[k].1.set_octant(octant, 255); 
                    }    
                }

                if !exists {
                    let mut temp_v = Voxel::new();

                    temp_v.set_surface_octant(octant, voxel_type);
                    voxels[v_len] = (y.trunc(), temp_v);

                    v_len += 1;
                }
            }

            // Ajust Voxels

            if x_adj == tvp[0] && z_adj == tvp[1] {
                for k in 0..v_len {
                    println!("{:?}", voxels[k]);  
                }
            }

            for k in 0..v_len {

                voxels[k].1.update_4_part_voxel();

                world.insert_voxel([
                        x_adj as i32,
                        voxels[k].0 as i32,
                        z_adj as i32,
                    ], 
                    voxels[k].1,  
                    true
                );
            }

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

use rand::{Rng, SeedableRng};


// let tree = world.generator.get_interference((x_pos + x) as f64, (z_pos + z) as f64);

// if y > 16.5 + world.sea_level as f64 && tree > 120.0 {
//     insert_tree(world, x_adj, y as u32, z_adj);
// }\

fn get_quad_height(y: f64) -> u8 {
    let y = y.fract();
    (y * 254.0).floor() as u8 + 1
}

fn insert_tree(world: &mut ShaderGrid, x: u32, y: u32, z: u32) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(world.seed + x as u64 + z as u64);
    let height = rng.random_range(5..10);

    for i in 0..height {
        world.insert_voxel([
            x as i32,
            (y + i) as i32,
            z as i32,
        ], 
        Voxel::from_type(2),  
        false
        );
    }

    world.insert_voxel([
        x as i32,
        (y + height) as i32,
        z as i32,
    ], 
    Voxel::from_type(1),  
    false
    );

    for i in 0..4 {

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

fn get_vertical_octant(y: f64) -> usize {
    if y > 0.5 {
        4
    } else  {
        0
    }
}
