use crate::{
    ShaderChunk,
    Voxel,
};


pub fn test() {
    let mut p = ShaderChunk::new([0, 0, 0]);
    let mut v = Voxel::from_type(3);
    v.update_4_part_voxel();

    //p.insert_voxel([0, 0, 0], v, false);
    p.insert_subchunk([0,0,0], 0, v, false);

    p.set_generation_level(5);
    let flat = p.flatten();

    println!("{:?}", flat.1);

}