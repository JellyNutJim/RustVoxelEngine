#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SteepFourHeightSurface {
    quadrants: [u32; 4], // High resolution voxel components
    rel_height: u32,
    flat: (u32, u32),
    has_water: bool,
}

// LEVEL 32 EQUALS BOTTOM OF CURRENT VOXEL HEIGHT

impl SteepFourHeightSurface {

    pub fn from(v: [u32; 4],  rh: u32) -> Self {
        Self {
            quadrants: v,
            rel_height: rh,
            flat: quad_to_flat(v),
            has_water: false,
        }
    }

    pub fn from_water_level(v: [u32; 4], rh: u32) -> Self {
        Self {
            quadrants: v,
            rel_height: rh,
            flat: quad_to_flat(v),
            has_water: true,
        }
    }

    pub fn set_water(&mut self, has_water: bool) {
        self.has_water = has_water;
    }

    pub fn flatten(&self) -> Vec<u32> {


        let mut g_type =  if self.has_water {
            0b0000_0011_0000_0000_0000_0000_0000_0000
        } else {
            0b0000_0010_0000_0000_0000_0000_0000_0000
        };

        g_type = g_type | (self.rel_height << 26);

        vec![
            g_type | self.flat.0,
            self.flat.1
        ]
    }
}

// Compacts four 14 bit u32s into two u32s
fn quad_to_flat(quads: [u32; 4]) -> (u32, u32) { 
    (
        ((quads[0]) << 10) | ((quads[1]) >> 4),
        (quads[3]) | ((quads[2]) << 14) | (((quads[1]) ) << 28)
    )
}