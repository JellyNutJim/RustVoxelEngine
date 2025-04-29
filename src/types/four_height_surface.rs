#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FourHeightSurface {
    octants: [u8; 8], // High resolution voxel components
    voxel: u32, // Combined octants
}

#[allow(unused)]
impl FourHeightSurface {
    pub fn new() -> Self{
        Self {
            octants: [
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            voxel:1,
        }
    }

    pub fn from(v: [u8; 8], t: u8) -> Self {
        Self {
            octants: v,
            voxel: t as u32,
        }
    }

    pub fn from_type(t: u8) -> Self {
        Self {
            octants: [t; 8],
            voxel: t as u32,
        }
    }

    pub fn from_u32(t: u32) -> Self {
        Self {
            octants: [0; 8],
            voxel: t, 
        }
    }

    pub fn from_octants(v: [u8; 8]) -> Self {
        Self {
            octants: v,
            voxel: octants_to_u32(v),
        }
    }

    pub fn from_quadrants(v: [u8; 4]) -> Self {
        let octants = [v[0], v[1], v[2], v[3], 0, 0, 0, 0];

        Self {
            octants: octants,
            voxel: quadrants_to_u32(octants),
        }
    }

    pub fn set_octant(&mut self, octant: usize, v_type: u8) {
        self.octants[octant] = v_type;
    }

    // Same as set octant, but inserts stone in the octant below, if there is an octant below
    pub fn set_surface_octant(&mut self, octant: usize, v_type: u8) {
        self.octants[octant] = v_type;
        if octant > 3 {
            self.octants[octant - 4] = 2;
        }
    }


    pub fn set_voxel(&mut self, v_type: u32) {
        self.voxel = v_type;
    }

    pub fn update_8_part_voxel(&mut self) -> u32 {
        self.voxel = octants_to_u32(self.octants);
        return self.voxel;
    }

    pub fn update_4_part_voxel(&mut self) -> u32 {
        self.voxel = quadrants_to_u32(self.octants);
        return self.voxel;
    }

    pub fn get_voxel(&self) -> u32 {
        self.voxel
    }

    pub fn flatten(&self) -> Vec<u32> {
        vec![0b0000_0001_0000_0000_0000_0000_0000_0000, self.voxel]
    }
}

impl Default for FourHeightSurface {
    fn default() -> Self {
        Self {
            octants: [0; 8],
            voxel: 0,
        }
    }
}

// Shifts each 4-bit value into the appropriate position
fn octants_to_u32(o: [u8; 8]) -> u32 { 
    let mut result: u32 = 0;
    
    for i in 0..8 {
        let shifted_value = (o[i] as u32) << (i * 4);
        result |= shifted_value;
    }

    result
}

// Shifts each 8-bit value into the appropriate position
fn quadrants_to_u32(o: [u8; 8]) -> u32 { 
    let mut result: u32 = 0;
    
    for i in 0..4 {
        let shifted_value = (o[i] as u32) << (i * 8);
        result |= shifted_value;
    }

    result
}