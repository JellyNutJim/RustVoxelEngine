use crate::{
    Octree,
    Voxel,
    FourHeightSurface,
    Geometry,
};

#[allow(unused)]
pub fn test() {
    let mut p = Octree::new([0, 0, 0]);
    let mut v = Voxel::from(3126736);
    //v.update_4_part_voxel();


    let mut f = FourHeightSurface::from_type(3);
    f.update_4_part_voxel();

    let g2 = Geometry::Voxel(v);
    let g1 = Geometry::FourHeightSurface(f);

    //p.insert_voxel([0, 0, 0], v, false);
    p.insert_subchunk([0,0,0], 0, g1, false);

    // let g2 = Geometry::FourHeightSurface(f);
    // p.insert_subchunk([32,0,0], 0, g1, false);
    // p.insert_subchunk([0,32,0], 0, g1, false);
    // p.insert_subchunk([0,0,32], 0, g1, false);

    // p.insert_subchunk([32,32,0], 0, g1, false);
    // p.insert_subchunk([32,0,32], 0, g1, false);
    // p.insert_subchunk([0,32,32], 0, g1, false);
    p.insert_subchunk([32,32,32], 0, g1, false);


    p.set_generation_level(2);
    let flat = p.flatten();

    println!("{:?}", flat.1);

}