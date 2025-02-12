use std::i32::MAX;
use crate::world::ShaderChunk;

// Holds data to be placed in the voxel buffer
// Origin will always be smaller than the current position
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ShaderGrid {
    origin: [i32; 3],  // Origin of the current grid = the origin of the chunk with the lowest positional value 
    grid: Vec<u32>, // Contains indexes relating to chunk memory location
}

impl ShaderGrid {
    
    // Finds the smallest chunk origin, sets that as the grid origin
    fn get_origin(chunk_positions: &Vec<([i32; 3], u32)>) -> [i32; 3]{
        let mut origin = [0, 0, 0];
        let mut curr_min = MAX;

        for pos in chunk_positions {
            let min: i32 = pos.0.iter().sum();
            if min < curr_min {
                origin = pos.0;
                curr_min = min;
            }
        }

        origin
    }
    
    // Assumes position is within the bounds of the grid
    fn get_chunk_pos(&self, pos: &[i32; 3]) -> [u32; 3] {
        [
            (pos[0] / 64 - self.origin[0]) as u32, 
            (pos[1] / 64 - self.origin[1]) as u32, 
            (pos[2] / 64 - self.origin[2]) as u32
        ]
    }

    fn new(width: u32, chunk_positions: &Vec<([i32; 3], u32)> ) -> Self {
        let origin = Self::get_origin(chunk_positions);
        let grid = vec![];

        let mut s = Self {
            origin,
            grid
        };

        for (pos, i) in chunk_positions {
            let chunk_pos = s.get_chunk_pos(pos);
            let index = chunk_pos[0] + chunk_pos[1] + chunk_pos[2];
            s.grid.insert(index as usize, *i); 
        }

        s
    }

    fn from(chunks: &Vec<ShaderChunk>) -> (Self, Vec<u32>) {
        let mut pos_ind: Vec<([i32; 3], u32)> = Vec::new();
        let mut flat_data: Vec<u32> = Vec::new();
        let mut curr_index = 0;

        // Convert to function in grid
        for chunk in chunks {
            let mut flat = chunk.flatten();
            pos_ind.push((flat.0, curr_index));

            curr_index += flat.1.len() as u32;
            flat_data.append(&mut flat.1);
        }

        (ShaderGrid::new(2, &pos_ind), flat_data)
    }
}