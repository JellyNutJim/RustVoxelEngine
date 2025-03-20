
pub struct Sphere {
    radius: u32,
    origin: [u8; 3]
}

impl Sphere {
    pub fn new() -> Self{
        Self {
            radius: 0,
            origin: [0, 0, 0]
        }
    }

    pub fn from(radius: u32, origin: [u8; 3]) -> Self{
        Self {
            radius,
            origin
        }
    }
}