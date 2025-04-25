use crate::Vec3;

pub struct CameraLocation {
    pub location: Vec3,
    pub old_loc: Vec3,
    pub h_angle: f64,
    pub v_angle: f64,
    pub direction: Vec3,
    pub sun_loc: Vec3,
}