use crate::Voxel;
use crate::Geometry;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Octree {
    pos: [i32; 3], // position in 3d space
    generation_level: u32, // Level of detail to send to gpu
    max_generation_level: u32, // Highest level of terrain generated
    data: Octant,
}

#[repr(C)]
#[derive(Debug, Clone)]
enum Octant {
    Leaf(Geometry),
    Branch(Box<ChildOctants>, Geometry),
}

#[repr(C)]
#[derive(Debug, Clone)]
struct ChildOctants([Octant; 8]);


// Depth = Octants of Size
// 0 = 32
// 1 = 16
// 2 = 8
// 3 = 4
// 4 = 2 
// 5 = 1 aka voxels!

// Octants follow this format -> layer = y, up = x, right = z,
// layer 1: 1 3
//          0 2
//
// layer 2: 5 7
//          4 6


// Generation Levels
// same as depth, but 2 is the smallest as terrain lower than that wont be generated for now

// #[allow(unused)]
impl Octree {
    pub fn new(pos: [i32; 3]) -> Self {
        Self {
            pos,
            generation_level: 0,
            max_generation_level: 0,
            data: Octant::Leaf(Geometry::Voxel(Voxel::new() )),
        }
    }

    pub fn new_with_type(pos: [i32; 3], data: u32) -> Self {
        Self {
            pos,
            generation_level: 0,
            max_generation_level: 0,
            data: Octant::Leaf( Geometry::Voxel(Voxel::from(data) )),
        }
    }

    pub fn get_origin(&self) -> [i32; 3]{
        self.pos
    }
    
    // Convert position to be local to the current chunk
    pub fn local_pos_from_vec(self, point: [i32; 3]) -> [u32; 3] {
        [
            (self.pos[0].abs() - point[0].abs()) as u32,
            (self.pos[1].abs() - point[1].abs()) as u32,
            (self.pos[2].abs() - point[2].abs()) as u32
        ]
    }

    // Get the octant index at a given position and depth
    pub fn get_octant_at_depth(pos: [u32; 3], depth: u32) -> usize {
        
        let octant_size = 64 / (2u32.pow(depth));
        let mid = octant_size / 2 - 1;
        let pos = [pos[0] % octant_size, pos[1] % octant_size, pos[2] % octant_size];
        let mut octant: usize = 0;

        #[cfg(debug_assertions)]
        if depth > 6 { panic!("Chunk depth out of range") }
       
        if pos[0] > mid {
            octant += 1;
        }

        if pos[1] > mid {
            octant += 4;
        }

        if pos[2] > mid {
            octant += 2;
        }
        
        octant
    }

    pub fn set_generation_level(&mut self, level: u32) {
        self.generation_level = level;

        if level > self.max_generation_level {
            self.max_generation_level = level;
        }
    }

    pub fn get_generation_level(&self) -> u32 {
        self.generation_level
    }

    pub fn get_max_generation_level(&self) -> u32 {
        self.max_generation_level
    }

    // Insert a single voxel at the given position2 -> Voxels are just leaf nodes at a depth 4
    pub fn insert_geometry(&mut self, pos: [u32; 3], geometry: Geometry, ground_insert: bool) {
        self.insert_subchunk(pos, 5, geometry, ground_insert);
    }

    // Create a subchunk of a specific type at a specific depth
    pub fn insert_subchunk(&mut self, pos: [u32; 3], depth: u32, geometry: Geometry, ground_insert: bool) {
        #[cfg(debug_assertions)]
        if depth > 6 { panic!("Depth out of range") }

        let mut current = &mut self.data;       
        let mut curr_octant;

        for i in 0..=depth {
            // Create octants if they do not already exist

            if let Octant::Leaf(data) = current {
                *current = Octant::Branch( Box::new(
                    ChildOctants([ 
                        Octant::Leaf(Geometry::Voxel(Voxel::new())),
                        Octant::Leaf(Geometry::Voxel(Voxel::new())),
                        Octant::Leaf(Geometry::Voxel(Voxel::new())),
                        Octant::Leaf(Geometry::Voxel(Voxel::new())),
                        Octant::Leaf(Geometry::Voxel(Voxel::new())),
                        Octant::Leaf(Geometry::Voxel(Voxel::new())),
                        Octant::Leaf(Geometry::Voxel(Voxel::new())),
                        Octant::Leaf(Geometry::Voxel(Voxel::new())),
                    ])), data.clone()
                );
            }

            // Set current to the next octant the given position lies in
            if let Octant::Branch(ref mut octants, _geom) = current {
                curr_octant = Self::get_octant_at_depth(pos, i);
                if ground_insert == true && i == depth { 
                    if curr_octant > 3 {
                        octants.0[curr_octant - 4] = Octant::Leaf(Geometry::Voxel(Voxel::new()));
                    }
                }

                current = &mut octants.0[curr_octant];
            }
        }

        // Set current octants value to the given voxel type
        *current = Octant::Leaf(geometry);

    }

    // Convert entire octree to flat u32 vector
    pub fn flatten(&self) -> ([i32; 3], Vec<u32>){
        // Check if chunk has no depth
        if let Octant::Leaf(geom) = self.data {
            return (self.pos, geom.flatten())
        }

        let mut octant_vec: Vec<u32> = Vec::new();
        let curr_depth = 0;
        let max_depth = self.get_generation_level(); 

        // Otherwise recursively search octree
        if let Octant::Branch(ref octants, _geom) = self.data {
            octant_vec = Self::get_flattened_octant(&octants.0, curr_depth, max_depth);
        }

        (self.pos, octant_vec)
    }

    // currently only works for voxels (four height data)
    fn get_flattened_octant(octants: &[Octant; 8], curr_depth: u32, max_depth: u32) -> Vec<u32>{
        let mut octant_vec: Vec<Vec<u32>> = Vec::new(); 
        let mut result: Vec<u32> = Vec::new();
        let mut curr_len = 0;

        for octant in octants {
            match octant { 
                Octant::Leaf(voxel_data) => { 
                    octant_vec.push(voxel_data.flatten()); 
                }

                Octant::Branch(ref octants, geom) => {
                    if curr_depth == max_depth {
                        octant_vec.push(geom.flatten());
                        continue;
                    } 

                    octant_vec.push(Self::get_flattened_octant(&octants.0, curr_depth + 1, max_depth));
                }
            } 
        }

        // When at the lowest point, all octants are garunteed to be within at least 
        if curr_depth != 5 {
            for (i, ov) in octant_vec.iter_mut().enumerate() {
                result.insert(i, curr_len + 9);
                curr_len += ov.len() as u32;
                result.append(ov);
            }
    
            result.insert(0, 0);
            result
        }
        else {
            result.push(0);
            let mut total_len: u32 = 0;
            for (i, ov) in octant_vec.iter_mut().enumerate() {
                let shifted_value = curr_len << (i * 4);
                curr_len += ov.len() as u32;

                total_len |= shifted_value; 
                result.append(ov);
            }

            result.insert(1, total_len);
            result
        }
    }

}


    // Returns the voxel type at the given position, accounts for subchunks
    // pub fn get_voxel(&mut self, pos: [u32; 3]) -> u32{
    //     let mut current = &mut self.data;       

    //     for i in 0..=5 {
    //         // Create octants if they do not already exist
    //         match current {
    //             ChunkContent::Leaf(a) => { println!{"{i}"}; return a.0.get_voxel() }
    //             ChunkContent::Octants(ref mut octants, voxel_type) => {
    //                 current = &mut octants.0[Self::get_octant_at_depth(pos, i)];
    //             }
    //         };
    //     }

    //     panic!("No Voxel or subchunk could be found");
    // }