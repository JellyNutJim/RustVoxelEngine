use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct ScalablePerlin {
    pub permutation: Vec<usize>,
    pub gradients: Vec<[f64; 2]>
}

#[allow(unused)]
impl ScalablePerlin {
    pub fn new(seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        let mut permutation: Vec<usize> = (0..256).collect();
        for i in 0..255 {
            let j = rng.random_range(i..256);
            permutation.swap(i, j);
        }

        let perm_double: Vec<usize> = permutation.iter()
            .chain(permutation.iter())
            .cloned()
            .collect();
        let permutation = perm_double;
        
        let mut gradients = Vec::with_capacity(256);
        for _ in 0..256 {
            let angle = rng.random_range(0.0..2.0 * PI);
            gradients.push([angle.cos(), angle.sin()]);
        }
        
        Self {
            permutation,
            gradients,
        }
    }
    
    fn fade(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }
    
    fn grad(&self, hash: usize, x: f64, y: f64) -> f64 {
        let gradient = self.gradients[hash & 255];
        gradient[0] * x + gradient[1] * y
    }
    
    pub fn noise(&self, x: f64, z: f64) -> f64 {

        let x_floor = x.floor() as isize;
        let z_floor = z.floor() as isize;
        
        let x = x - x_floor as f64;
        let z = z - z_floor as f64;
        
        let x_floor = x_floor & 255;
        let z_floor = z_floor & 255;
        
        let a = self.permutation[x_floor as usize] + z_floor as usize;
        let aa = self.permutation[a];
        let ab = self.permutation[a + 1];
        let b = self.permutation[(x_floor + 1) as usize] + z_floor as usize;
        let ba = self.permutation[b];
        let bb = self.permutation[b + 1];

        let u = Self::fade(x);
        let v = Self::fade(z);
        
        let x1 = self.grad(aa, x, z);
        let x2 = self.grad(ba, x - 1.0, z);
        let z1 = lerp(x1, x2, u);
        
        let x1 = self.grad(ab, x, z - 1.0);
        let x2 = self.grad(bb, x - 1.0, z - 1.0);
        let z2 = lerp(x1, x2, u);
        
        lerp(z1, z2, v)
    }

    pub fn get_noise_at_point(&self, x: f64, z: f64, scale: f64) -> f64 {
        self.noise(x * scale, z * scale)
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}