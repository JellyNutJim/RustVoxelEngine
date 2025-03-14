
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Voxel {
    octants: [u8; 8], // High resolution voxel components
    voxel: u32, // Combined octants
}

#[allow(unused)]
impl Voxel {
    pub fn new() -> Self{
        Self {
            octants: [
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            voxel: 0,
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
            voxel: u8s_to_u32([t; 8]),
        }
    }

    pub fn from_octants(v: [u8; 8]) -> Self {
        Self {
            octants: v,
            voxel: u8s_to_u32(v),
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

    pub fn update_voxel(&mut self) -> u32 {
        self.voxel = u8s_to_u32(self.octants);
        return self.voxel;
    }

    pub fn get_voxel(&self) -> u32 {
        self.voxel
    }
}

impl Default for Voxel {
    fn default() -> Self {
        Self {
            octants: [0; 8],
            voxel: 0,
        }
    }
}

fn u8s_to_u32(o: [u8; 8]) -> u32 { 
    let mut result: u32 = 0;
    
    // Shift each 4-bit value into the appropriate position
    for i in 0..8 {
        // Convert the U4 to u32, shift it to its position, and OR it into the result
        let shifted_value = (o[i] as u32) << (i * 4);
        result |= shifted_value;
    }

    result
}

