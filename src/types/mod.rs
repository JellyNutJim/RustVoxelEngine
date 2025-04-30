mod vec3;
mod voxel;
mod octo_voxel;
mod sphere;
mod four_height_surface;
mod steep_four_height_surface;


pub use vec3::Vec3;
pub use voxel::Voxel;
pub use sphere::Sphere;
pub use four_height_surface::FourHeightSurface;
pub use octo_voxel::OctoVoxel;
pub use steep_four_height_surface::SteepFourHeightSurface;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum Geometry {
    Voxel(Voxel),
    OctoVoxel(OctoVoxel),
    Sphere(Sphere),
    FourHeightSurface(FourHeightSurface),
    SteepFourHeightSurface(SteepFourHeightSurface),
}

impl Geometry {
    pub fn flatten(&self) -> Vec<u32> {
        match self {
            Geometry::Voxel(voxel) => voxel.flatten(),
            Geometry::OctoVoxel(octo_voxel) => octo_voxel.flatten(),
            Geometry::Sphere(sphere) => sphere.flatten(),
            Geometry::FourHeightSurface(four_height) => four_height.flatten(),
            Geometry::SteepFourHeightSurface(steep_height) => steep_height.flatten(),
        }
    }
}