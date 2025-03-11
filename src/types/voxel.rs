pub struct Voxel {
    octants: [u8; 8], // High resolution voxel components
    voxel: u32, // Low resolution voxel type
}

impl Voxel {
    pub fn new() -> Self{
        Self {
            octants: [
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            voxel: 0,
        }
    }

    pub fn from(v: [u8; 8], t: u32) -> Self {
        Self {
            octants: v,
            voxel: t,
        }
    }

    pub fn from_voxel(t: u32) -> Self {
        Self {
            octants: [0; 8],
            voxel: t,
        }
    }

    pub fn from_octants(v: [u8; 8]) -> Self {
        Self {
            octants: v,
            voxel: 0,
        }
    }

    pub fn set_octant(&mut self, octant: usize, v_type: u8) {
        self.octants[octant] = v_type;
    }

    pub fn set_voxel(&mut self, v_type: u32) {
        self.voxel = v_type;
    }

}