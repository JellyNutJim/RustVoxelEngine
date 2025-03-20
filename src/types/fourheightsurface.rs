
pub struct FourHeightSurface {
    quadrants: [u8; 4],
    meta: u32,
}

impl FourHeightSurface {
    pub fn new() -> Self {
        Self {
            quadrants: [0; 4],
            meta: 0,
        }
    }

    pub fn get_joined_quadrants(&self) -> u32 {
        let mut result: u32 = 0;
    
        for i in 0..4 {
            result |= (self.quadrants[i] as u32) << (i * 8);
        }
    
        result
    }
    
}