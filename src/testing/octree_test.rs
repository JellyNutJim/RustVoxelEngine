use crate::{
    Octree,
    Voxel,
    FourHeightSurface,
    SteepFourHeightSurface,
    Geometry,
};

#[allow(unused)]
pub fn test() {
    let mut p = Octree::new([0, 0, 0]);
    let mut v = Voxel::from(3126736);
        //v.update_4_part_voxel();
        
    let x = SteepFourHeightSurface::from(
        [
            0b10_0000_0000_0011,
            0b10_0000_0000_1000,
            0b10_0000_0001_0000,
            0b10_0000_0000_0001,
        ]
    );

    let n1 = x.flatten()[0];
    let n2 = x.flatten()[1];

    println!("1: {:032b}", x.flatten()[0]);
    println!("2: {:032b}", x.flatten()[1]);
    //let mut f = FourHeightSurface::from_type(3);
    //f.update_4_part_voxel();

    let exisiing = (n1 >> 24); 

    let num1 = (n1 >> 10) & 0x3FFF;
    let num2 = (n1 & 0x3FF) << 4 | (n2 >> 28);
    let num3 = (n2 >> 14) & 0x3FFF;
    let num4 = n2 & 0x3FFF;


    println!("{} {} {} {} {}", exisiing, num1, num2, num3, num4);



    let g2 = Geometry::Voxel(v);
    //let g1 = Geometry::FourHeightSurface(f);

    //p.insert_voxel([0, 0, 0], v, false);
    p.insert_subchunk([32,0,0], 0, g2, false);


    p.set_generation_level(2);
    let flat = p.flatten();

    println!("{:?}", flat.1);

}