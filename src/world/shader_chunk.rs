#[repr(C)]
#[derive(Debug, Clone)]
pub struct ShaderChunk {
    pos: [i32; 3], // position in 3d space
    data: ChunkContent,
}

#[repr(C)]
#[derive(Debug, Clone)]
enum ChunkContent {
    Leaf(VoxelData),
    Octants(Box<OctantsData>),
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct VoxelData(u32);

#[repr(C)]
#[derive(Debug, Clone)]
struct OctantsData([ChunkContent; 8]);


// Depth = Octants of Size
// 0 = 32
// 1 = 16
// 2 = 8
// 3 = 4
// 4 = 2 aka Voxels!!!

// Octants follow this format -> layer = y, up = x, right = z,
// layer 1: 1 3
//          0 2
//
// layer 2: 5 7
//          4 6

impl ShaderChunk {
    pub fn new(pos: [i32; 3]) -> Self {
        Self {
            pos,
            data: ChunkContent::Leaf(VoxelData(0)),
        }
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

    // Insert a single voxel at the given position2 -> Voxels are just leaf nodes at a depth 4
    pub fn insert_voxel(&mut self, pos: [u32; 3], voxel_type: u32) {
        self.insert_subchunk(pos, 4, voxel_type);
    }

    // Create a subchunk of a specific type at a specific depth
    pub fn insert_subchunk(&mut self, pos: [u32; 3], depth: u32, voxel_type: u32) {
        #[cfg(debug_assertions)]
        if depth > 5 { panic!("Depth out of range") }

        let mut current = &mut self.data;       

        for i in 0..=depth {
            // Create octants if they do not already exist
            if let ChunkContent::Leaf(_) = current {
                *current = ChunkContent::Octants( Box::new(
                    OctantsData([ 
                        ChunkContent::Leaf(VoxelData(0)),
                        ChunkContent::Leaf(VoxelData(0)),
                        ChunkContent::Leaf(VoxelData(0)),
                        ChunkContent::Leaf(VoxelData(0)),
                        ChunkContent::Leaf(VoxelData(0)),
                        ChunkContent::Leaf(VoxelData(0)),
                        ChunkContent::Leaf(VoxelData(0)),
                        ChunkContent::Leaf(VoxelData(0)),
                    ]))
                );
            }

            // Set current to the next octant the given position lies in
            if let ChunkContent::Octants(ref mut octants) = current {
                current = &mut octants.0[Self::get_octant_at_depth(pos, i)];
            }
        }

        // Set current octants value to the given voxel type
        *current = ChunkContent::Leaf(VoxelData(voxel_type));
    }

    // Returns the voxel type at the given position, accounts for subchunks
    pub fn get_voxel(&mut self, pos: [u32; 3]) -> u32{
        let mut current = &mut self.data;       

        for i in 0..=5 {
            // Create octants if they do not already exist
            match current {
                ChunkContent::Leaf(a) => { println!{"{i}"}; return a.0 }
                ChunkContent::Octants(ref mut octants) => {
                    current = &mut octants.0[Self::get_octant_at_depth(pos, i)];
                }
            };
        }

        panic!("No Voxel or subchunk could be found");
    }

    // Format
    // build one branch of the octree at a time
    // From ground up

    // Recursive
    // Get octant,
    // Check if its a leaf
    // If it is, mark as such, and provide voxel_type [0, 0, 23]
    // Else,


    // [index1, index2, index3, index4, octal1, octal2, octal3, octal4]


    // Convert 
    pub fn flatten(&self) -> ([i32; 3], Vec<u32>){
        // Check if chunk has no depth
        if let ChunkContent::Leaf(voxel_data) = self.data {
            return (self.pos, vec![0, voxel_data.0])
        }

        let mut octant_vec: Vec<u32> = Vec::new();

        // Otherwise recursively search octree
        if let ChunkContent::Octants(ref octants) = self.data {
            octant_vec = Self::get_flattened_octant(&octants.0);
        }

        (self.pos, octant_vec)
    }

    fn get_flattened_octant(octants: &[ChunkContent; 8]) -> Vec<u32>{
        let mut octant_vec: Vec<Vec<u32>> = Vec::new(); 
        let mut result: Vec<u32> = Vec::new();
        let mut curr_len = 0;

        for octant in octants {
            match octant { 
                ChunkContent::Leaf(voxel_data) => { octant_vec.push(vec![0, voxel_data.0]); }
                ChunkContent::Octants(ref octants) => {
                    octant_vec.push(Self::get_flattened_octant(&octants.0));
                }
            } 
        }

        for (i, ov) in octant_vec.iter_mut().enumerate() {
            result.insert(i, curr_len + 9);
            curr_len += ov.len() as u32;
            result.append(ov);
        }

        result.insert(0, 1);

        result
    }
}