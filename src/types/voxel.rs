pub struct Voxel {
    octants: [u8; 8], // High resolution voxel components
    voxel: u32, // Low resolution voxel type
    low_res_type: u32,
}

impl Voxel {
    pub fn new() -> Self{
        Self {
            octants: [
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            voxel: 0,
            low_res_type: 0,
        }
    }

    pub fn from(v: [u8; 8], t: u32) -> Self {
        Self {
            octants: v,
            voxel: 0,
            low_res_type: t,
        }
    }

    pub fn from_voxel(t: u32) -> Self {
        Self {
            octants: [0; 8],
            voxel: 0,
            low_res_type: t,
        }
    }

    pub fn from_octants(v: [u8; 8]) -> Self {
        Self {
            octants: v,
            voxel: u8s_to_u32(v),
            low_res_type: 0,
        }
    }

    pub fn set_octant(&mut self, octant: usize, v_type: u8) {
        self.octants[octant] = v_type;
    }

    pub fn set_voxel(&mut self, v_type: u32) {
        self.voxel = v_type;
    }

    pub fn update_voxel(&mut self) -> u32 {
        let mut result: u32 = 0;
        
        self.voxel = u8s_to_u32(self.octants);
        return self.voxel;
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