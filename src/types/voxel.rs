#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Voxel {
    voxel_type: u32,
}

#[allow(unused)]
impl Voxel {
    pub fn new() -> Self {
        Self {
            voxel_type: 1
        }
    }

    pub fn from(mut voxel_type: u32) -> Self {
        if voxel_type == 0 {
            voxel_type = 1;
        }

        Self {
            voxel_type
        }
    }

    
    pub fn flatten(&self) -> Vec<u32> {
        vec![self.voxel_type]
    }
}