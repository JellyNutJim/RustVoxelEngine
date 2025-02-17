// Shader Usage
mod shader_chunk;
mod shader_grid;

// Rust usage
mod chunk;

pub use shader_chunk::ShaderChunk;
pub use shader_grid::ShaderGrid;


pub fn get_flat_world() -> (Vec<i32>, Vec<u32>){
    // let mut chunks: Vec<ShaderChunk> = vec![
    //     ShaderChunk::new([0, 0, 0]),
    //     ShaderChunk::new([0, 64, 0]),
    //     ShaderChunk::new([0, 0, 64]),
    //     ShaderChunk::new([0, 64, 64]),
    //     ShaderChunk::new([64, 0, 0]),
    //     ShaderChunk::new([64, 64, 0]),
    //     ShaderChunk::new([64, 0, 64]),
    //     ShaderChunk::new([64, 64, 64]),




    // ];

    let mut chunks: Vec<ShaderChunk> = vec![
        // y = 0
        ShaderChunk::new([0, 0, 0]),
        ShaderChunk::new([64, 0, 0]),
        ShaderChunk::new([128, 0, 0]),

        ShaderChunk::new([0, 0, 64]),
        ShaderChunk::new([64, 0, 64]),
        ShaderChunk::new([128, 0, 64]),

        ShaderChunk::new([0, 0, 128]),
        ShaderChunk::new([64, 0, 128]),
        ShaderChunk::new([128, 0, 128]),

        // y = 1
        ShaderChunk::new([0, 64, 0]),
        ShaderChunk::new([64, 64, 0]),
        ShaderChunk::new([128, 64, 0]),
        
        ShaderChunk::new([0, 64, 64]),
        ShaderChunk::new([64, 64, 64]),
        ShaderChunk::new([128, 64, 64]),

        ShaderChunk::new([0, 64, 128]),
        ShaderChunk::new([64, 64, 128]),
        ShaderChunk::new([128, 64, 128]),
        
        // y = 2
        ShaderChunk::new([0, 128, 0]),
        ShaderChunk::new([64, 128, 0]),
        ShaderChunk::new([128, 128, 0]),
        
        ShaderChunk::new([0, 128, 64]),
        ShaderChunk::new([64, 128, 64]),
        ShaderChunk::new([128, 128, 64]),

        ShaderChunk::new([0, 128, 128]),
        ShaderChunk::new([64, 128, 128]),
        ShaderChunk::new([128, 128, 128]),

    ];

    // chunks.get_mut(0).unwrap().insert_voxel([1,1,1], 1);
    // chunks.get_mut(0).unwrap().insert_subchunk([2,2,2], 4, 1);
    // chunks.get_mut(0).unwrap().insert_subchunk([4,4,4], 3, 1);
    // chunks.get_mut(0).unwrap().insert_subchunk([8,8,8], 2, 1);
    // chunks.get_mut(0).unwrap().insert_subchunk([16,16,16], 1, 1);
    // chunks.get_mut(0).unwrap().insert_subchunk([32,32,32], 0, 1);


    //let mut p = ShaderGrid::from(chunks, 3);
    let width = 20;

    let noise = PerlinNoise::new(42); // Use any seed
    let w = width*64;
    let h = width*64;
    let scale = 0.01; // Adjust this to change the "zoom level" of the noise
    
    let noise_matrix = noise.generate_grid(w, h, scale);


    let mut p = ShaderGrid::new(width as u32, [-64 * (width as i32 / 2), -64, -64 * (width as i32/ 2)]);

    println!("INSERT");
    for x in 0..(64*width as i32) {
        for z in 0..(64*width as i32) {
            let mut y = noise_matrix[x as usize][z as usize] * 64.0;

            let x_adjusted = x - (64 * width / 2 ) as i32;
            let z_adjusted = z - (64 * width / 2 ) as i32;
            
            if y > 64.0{
                y = ((y / 4.0).floor() * 4.0);
            }

            let y = y as i32;



            p.insert_voxel([x_adjusted, y, z_adjusted], 1);
            p.insert_voxel([x_adjusted, y - 1, z_adjusted], 2);
            p.insert_voxel([x_adjusted, y - 2, z_adjusted], 2);
            p.insert_voxel([x_adjusted, y - 3, z_adjusted], 2);
        }
    }


    // for x in 0..((64 * width as i32) + 0) {
    //     for z in 0..(64 * width as i32) {

    //         let x_adjusted = x - (64 * width / 2 ) as i32;
    //         let z_adjusted = z - (64 * width / 2 ) as i32;

    //         p.insert_voxel([x_adjusted, -3, z_adjusted], 1);
    //     }
    // }

    //p.insert_voxel([0, 0, 0], 1);
    //p.insert_voxel([1*(64 * 4), -1, 1 * (64 * 4)], 1);

    // p.insert_voxel([2, 0, 0], 1);

    // p.insert_voxel([2, 0, 2], 1);

    // p.insert_voxel([0, 0, 2], 1);

    // p.insert_voxel([0, 2, 0], 1);

    // p.insert_voxel([0, -2, 0], 1);
    // p.insert_voxel([0, 2, 0], 1);
    
    // p.insert_voxel([1, -4, 1], 1);

    // p.insert_voxel([0, 0, 63], 1);                                        
    // p.insert_voxel([0, 2, 64], 1);
    // p.insert_voxel([0, 4, 65], 1);



    let p = p.flatten();

    //println!("Grid: {:?}\nChunk Data: {:?}", p.0, p.1);
    println!("Grid: {}", p.1.len());

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