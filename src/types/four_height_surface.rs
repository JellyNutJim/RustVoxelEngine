#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FourHeightSurface {
    quadrants: [u8; 4], // High resolution voxel components
    voxel: u32, // Combined octants
    has_water: bool,
}

impl FourHeightSurface {

    pub fn from(v: [u8; 4]) -> Self {
        Self {
            quadrants: v,
            voxel: quadrants_to_u32(v),
            has_water: false,
        }
    }

    pub fn from_u32(v: [u32; 4]) -> Self {
        let v8 = [
            v[0] as u8,
            v[1] as u8,
            v[2] as u8,
            v[3] as u8,
        ];

        Self {
            quadrants: v8,
            voxel: quadrants_to_u32(v8),
            has_water: false,
        }
    }

    pub fn from_u32_water(v: [u32; 4]) -> Self {
        let v8 = [
            v[0] as u8,
            v[1] as u8,
            v[2] as u8,
            v[3] as u8,
        ];

        Self {
            quadrants: v8,
            voxel: quadrants_to_u32(v8),
            has_water: true,
        }
    }


    pub fn from_water_level(v: [u8; 4]) -> Self {
        Self {
            quadrants: v,
            voxel: quadrants_to_u32(v),
            has_water: true,
        }
    }

    pub fn from_f64(v: [f64; 4]) -> Self {
        let quadrants = [
            (v[0].fract() * 255.0) as u8,
            (v[1].fract() * 255.0) as u8,
            (v[2].fract() * 255.0) as u8,
            (v[3].fract() * 255.0) as u8
        ];

        Self {
            quadrants: quadrants,
            voxel: quadrants_to_u32(quadrants),
            has_water: false,
        }
    }

    pub fn from_f64_water(v: [f64; 4]) -> Self {
        let quadrants = [
            (v[0].fract() * 255.0) as u8,
            (v[1].fract() * 255.0) as u8,
            (v[2].fract() * 255.0) as u8,
            (v[3].fract() * 255.0) as u8
        ];

        Self {
            quadrants: quadrants,
            voxel: quadrants_to_u32(quadrants),
            has_water: true,
        }
    }
 
    // sets quadrant
    pub fn set_quadrant(&mut self, octant: usize, v_type: u8) {
        self.quadrants[octant] = v_type;
    }

    pub fn set_water(&mut self, has_water: bool) {
        self.has_water = has_water;
    }

    pub fn update_4_part_voxel(&mut self) -> u32 {
        self.voxel = quadrants_to_u32(self.quadrants);
        return self.voxel;
    }

    pub fn flatten(&self) -> Vec<u32> {
        if self.has_water {
            vec![0b0000_0101_0000_0000_0000_0000_0000_0000, self.voxel]
        } else {
            vec![0b0000_0100_0000_0000_0000_0000_0000_0000, self.voxel]
        }
    }
}

impl Default for FourHeightSurface {
    fn default() -> Self {
        Self {
            quadrants: [0; 4],
            voxel: 0,
            has_water: false,
        }
    }
}

// Shifts each 8-bit value into the appropriate position
fn quadrants_to_u32(o: [u8; 4]) -> u32 { 
    let mut result: u32 = 0;
    
    for i in 0..4 {
        let shifted_value = (o[i] as u32) << (i * 8);
        result |= shifted_value;
    }

    result
}