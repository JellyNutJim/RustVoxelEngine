
use super::FourHeightSurface;
use super::Voxel;

pub trait get_geom {

}

pub enum Geometry {
    Voxel(Voxel),
    FourHeightSurface(FourHeightSurface),
    FourPointSurface(),
    Sphere(),
}



