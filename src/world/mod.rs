// Shader Usage
mod shader_chunk;
mod shader_grid;

// Rust usage
mod chunk;

pub use shader_chunk::ShaderChunk;
pub use shader_grid::ShaderGrid;


// VOXEL TYPES
// 0 - air
// 1 - grass
// 2 - stone
// 3 - water


pub fn get_flat_world(seed: u64) -> (Vec<i32>, Vec<u32>) {


    //let mut p = ShaderGrid::from(chunks, 3);
    let width = 40;


    let origin = [(width * 64) as i32, 640, (width * 64) as i32];

    let mut p = ShaderGrid::new(width as u32, origin);
    p.insert_voxel([3840, 801, 3840], 1);

    p.insert_voxel([3842, 800, 3840], 1);

    // Elavation
    create_flat_with_water(&mut p, width, origin, seed);




    let p = p.flatten();
    println!("Grid: {}", p.1.len());

    p
}

fn create_continents(p: &mut ShaderGrid, width: usize, origin: [i32; 3], seed: u64) {
    let noise = PerlinNoise::new(seed); // Use any seed
    let w = width*4;
    let h = width*4;
    let scale = 0.01; // Adjust this to change the "zoom level" of the noise
    
    let noise_matrix = noise.generate_grid(w, h, scale);

    for x in 0..(w as i32) {
        for z in 0..(h  as i32) {
            let y = ((noise_matrix[x as usize][z as usize] * 16.0).floor()) as i32;
            let x_adjusted = x * 16 + origin[0] as i32;
            let z_adjusted = z * 16 + origin[2] as i32;

            //p.insert_subchunk([x_adjusted, y, z_adjusted], 1, 1);

            //println!("{y}");
 
            p.insert_subchunk([x_adjusted, y + 800, z_adjusted], 1, 1);
        }
    }


}

fn create_flat_with_water(p: &mut ShaderGrid, width: usize, origin: [i32; 3], seed: u64) {
    let noise = PerlinNoise::new(seed); // Use any seed
    let w = width*64;
    let h = width*64;
    let scale = 0.005; // Adjust this to change the "zoom level" of the noise
    
    let noise_matrix = noise.generate_grid(w, h, scale);

    // Elavation
    for x in (0..((64*width) as i32)).step_by(1) {
        for z in (0..((64*width) as i32)).step_by(1) {
            let y = noise_matrix[x as usize][z as usize] * 64.0;

            let x_adjusted = x + origin[0] as i32;
            let z_adjusted = z + origin[2]  as i32;

            let y = (y + 768.0) as i32;
            if y < 768 {
                let mut temp = y + 1;
                while temp < 769 {
                    p.insert_voxel([x_adjusted, temp, z_adjusted], 3);
                    temp += 1;
                }
            }

            p.insert_voxel([x_adjusted, y, z_adjusted], 1);
            p.insert_voxel([x_adjusted, y - 1, z_adjusted], 2);
            p.insert_voxel([x_adjusted, y - 2, z_adjusted], 2);
            p.insert_voxel([x_adjusted, y - 3, z_adjusted], 2);
        }
    }
}

pub fn get_empty() -> (Vec<i32>, Vec<u32>) {
    let width = 20;
    let mut p = ShaderGrid::new(width as u32, [-64 * (width as i32 / 2), -64, -64 * (width as i32/ 2)]);
    p.insert_voxel([0,0,0], 1);

    let p = p.flatten();

    p
}

// fn perlin(x: u32, y: u32, seed: u32) {
//     let mut rng = rand::rngs::StdRng::seed_from_u64(seed);


// }



use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

pub struct PerlinNoise {
    permutation: Vec<usize>,
    gradients: Vec<[f64; 2]>,
}

impl PerlinNoise {
    pub fn new(seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        // Create permutation table
        let mut permutation: Vec<usize> = (0..256).collect();
        for i in 0..255 {
            let j = rng.random_range(i..256);
            permutation.swap(i, j);
        }
        
        // Double the permutation table to avoid overflow
        let perm_double: Vec<usize> = permutation.iter()
            .chain(permutation.iter())
            .cloned()
            .collect();
        let permutation = perm_double;
        
        // Generate random gradient vectors
        let mut gradients = Vec::with_capacity(256);
        for _ in 0..256 {
            let angle = rng.random_range(0.0..2.0 * PI);
            gradients.push([angle.cos(), angle.sin()]);
        }
        
        PerlinNoise {
            permutation,
            gradients,
        }
    }
    
    fn fade(t: f64) -> f64 {
        // Smoothing function: 6t^5 - 15t^4 + 10t^3
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }
    
    fn grad(&self, hash: usize, x: f64, y: f64) -> f64 {
        let gradient = self.gradients[hash & 255];
        gradient[0] * x + gradient[1] * y
    }
    
    pub fn noise(&self, x: f64, y: f64) -> f64 {
        // Find unit grid cell containing point
        let x_floor = x.floor() as isize;
        let y_floor = y.floor() as isize;
        
        // Get relative position within cell
        let x = x - x_floor as f64;
        let y = y - y_floor as f64;
        
        // Wrap to permutation table
        let x_floor = x_floor & 255;
        let y_floor = y_floor & 255;
        
        // Calculate hashes for each corner
        let a = self.permutation[x_floor as usize] + y_floor as usize;
        let aa = self.permutation[a];
        let ab = self.permutation[a + 1];
        let b = self.permutation[(x_floor + 1) as usize] + y_floor as usize;
        let ba = self.permutation[b];
        let bb = self.permutation[b + 1];
        
        // Get fade curves
        let u = Self::fade(x);
        let v = Self::fade(y);
        
        // Interpolate between grid point gradients
        let x1 = self.grad(aa, x, y);
        let x2 = self.grad(ba, x - 1.0, y);
        let y1 = lerp(x1, x2, u);
        
        let x1 = self.grad(ab, x, y - 1.0);
        let x2 = self.grad(bb, x - 1.0, y - 1.0);
        let y2 = lerp(x1, x2, u);
        
        lerp(y1, y2, v)
    }
    
    pub fn generate_grid(&self, width: usize, height: usize, scale: f64) -> Vec<Vec<f64>> {
        let mut grid = vec![vec![0.0; width]; height];
        
        for y in 0..height {
            for x in 0..width {
                let nx = x as f64 * scale;
                let ny = y as f64 * scale;
                grid[y][x] = self.noise(nx, ny);
            }
        }
        
        grid
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}