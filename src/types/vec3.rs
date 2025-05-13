use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64
}



impl Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul for Vec3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, f: f64) -> Self {
        Self {
            x: self.x * f,
            y: self.y * f,
            z: self.z * f,
        }
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, f: f64) -> Self {
        Self {
            x: self.x / f,
            y: self.y / f,
            z: self.z / f,
        }
    }
}

impl Vec3 {
    pub fn new() -> Self {
        Vec3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn from(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    pub fn from_i32_3(origin: [i32; 3]) -> Self {
        Vec3 {x: origin[0] as f64, y: origin[1] as f64, z: origin[2] as f64}
    }

    pub fn floor(mut self) -> Self{
        self.x = self.x.floor();
        self.y = self.y.floor();
        self.z = self.z.floor();

        self
    }

    #[allow(dead_code)]
    pub fn dot(self, other: Self) -> f64 {
        {   self.x * other.x +
            self.y * other.y +
            self.z * other.z
        }
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x
        }
    }

    pub fn magnitude_squared(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn magnitude(self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn norm(self) -> Self {
        self / self.magnitude()
    }

}