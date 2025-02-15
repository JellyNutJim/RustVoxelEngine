use std::i32::MAX;
use crate::world::ShaderChunk;

use super::chunk;

// Holds data to be placed in the voxel buffer
// Origin will always be smaller than the current position
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ShaderGrid {
    origin: [i32; 3],  // Origin of the current grid = the origin of the chunk with the lowest positional value 
    width: u32,
    chunks: Vec<ShaderChunk>
}

impl ShaderGrid {
    
    // Finds the smallest chunk origin, sets that as the grid origin
    fn get_origin_from_flat(chunk_positions: &Vec<([i32; 3], u32)>) -> [i32; 3]{
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

    // Finds the smallest chunk origin, and sets that to the grid origin
    fn get_origin_from_chunks(shader_chunks: &Vec<ShaderChunk>) -> [i32; 3] {
        let mut origin = [0, 0, 0];
        let mut curr_min = MAX;

        for chunk in shader_chunks {
            let min: i32 = chunk.get_origin().iter().sum();
            if min < curr_min {
                origin = chunk.get_origin();
                curr_min = min;
            }
        }

        origin
    }
    
    // Assumes position is within the bounds of the grid
    pub fn get_chunk_pos(&self, pos: &[i32; 3]) -> [u32; 3] {
        [
            (pos[0] / 64 - self.origin[0]) as u32, 
            (pos[1] / 64 - self.origin[1]) as u32, 
            (pos[2] / 64 - self.origin[2]) as u32
        ]
    }

    // Create grid with given origin and size
    pub fn new(width: u32, origin: [i32; 3], ) -> Self {
        let mut s = Self {
            origin: origin,
            width: width,
            chunks: Vec::new(),
        };

        let width = width as i32;

        // Fill grid with empty chunks
        for x in 0..width {
            for y in 0..width {
                for z in 0..width {
                    s.chunks.push(ShaderChunk::new([(64 * x) + origin[0], (64 * y) + origin[1], (64 * z) + origin[2]]));
                }
            }
        }

        s
    }

    // Create grid with existing chunks and width
    pub fn from(chunks: Vec<ShaderChunk>, width: u32) -> Self {
        #[cfg(debug_assertions)]
        if chunks.len() > width.pow(3) as usize { panic!("Chunk depth out of range") }

        Self {
            origin: Self::get_origin_from_chunks(&chunks),
            width: width,
            chunks: chunks,
        }
    }

    pub fn flatten(&self) -> (Vec<i32>, Vec<u32>) {
        let mut flat_chunks: Vec<u32> = Vec::new();
        let mut flat_grid: Vec<i32> = vec![0; (self.width.pow(3)) as usize];

        // Accumulated index
        let mut curr_index: i32 = 0;

        // Convert to function in grid
        for chunk in &self.chunks {
            let mut flat = chunk.flatten();
            let chunk_pos = self.get_chunk_pos(&flat.0);

            let grid_index = chunk_pos[0] + chunk_pos[1] * self.width + chunk_pos[2] * self.width * self.width;

            flat_grid[grid_index as usize] = curr_index; 

            curr_index += flat.1.len() as i32;
            flat_chunks.append(&mut flat.1);
        }

        // Add origin to the flat grid
        flat_grid.splice(0..0, Vec::from(self.origin));
       

        (flat_grid, flat_chunks)
    }
}

