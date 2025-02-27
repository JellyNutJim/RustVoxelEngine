use std::i32::MAX;

use crate::world::ShaderChunk;

use super::PerlinNoise;

// Holds data to be placed in the voxel buffer
// Origin will always be smaller than the current position
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ShaderGrid {
    origin: [i32; 3],  // Origin of the current grid = the origin of the chunk with the lowest positional value 
    width: u32,
    grid: Vec<u32>, // Grid relating to Shaderchunk Structs
    chunks: Vec<ShaderChunk>,
    pub noise: PerlinNoise,

    flat_chunks: Vec<Vec<u32>>,
    seed: u64,
}

#[allow(unused)]
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

    pub fn insert_voxel(&mut self, pos: [i32; 3], voxel_type: u32) {

        let chunk_pos = self.get_chunk_pos(&pos);
        let chunk_index = chunk_pos[0] + chunk_pos[1] * self.width + chunk_pos[2] * (self.width * self.width);
        
        // Local voxel pos in chunk
        let pos = [
            (((pos[0] % 64) + 64) % 64) as u32,
            (((pos[1] % 64) + 64) % 64) as u32,
            (((pos[2] % 64) + 64) % 64) as u32
        ];

        //println!("{:?}", chunk_pos);

        self.chunks[self.grid[chunk_index as usize] as usize].insert_voxel(pos, voxel_type);
    }

    pub fn insert_subchunk(&mut self, pos: [i32; 3], voxel_type: u32, depth: u32) {

        let chunk_pos = self.get_chunk_pos(&pos);
        let chunk_index = chunk_pos[0] + chunk_pos[1] * self.width + chunk_pos[2] * (self.width * self.width);
        
        // Local voxel pos in chunk
        let pos = [
            (((pos[0] % 64) + 64) % 64) as u32,
            (((pos[1] % 64) + 64) % 64) as u32,
            (((pos[2] % 64) + 64) % 64) as u32
        ];

        //println!("{:?}", chunk_pos);

        self.chunks[self.grid[chunk_index as usize] as usize].insert_subchunk(pos, depth, voxel_type);
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
            (((pos[0] - (pos[0] & 63)) / 64) - self.origin[0] / 64) as u32, 
            (((pos[1] - (pos[1] & 63)) / 64) - self.origin[1] / 64) as u32, 
            (((pos[2] - (pos[2] & 63)) / 64) - self.origin[2] / 64) as u32
        ]
    }

    // Create grid with given origin and size
    pub fn new(width: u32, origin: [i32; 3], seed: u64 ) -> Self {
        let mut s = Self {
            origin: origin,
            width: width,
            grid: vec![0; (width.pow(3)) as usize],
            noise: PerlinNoise::new(seed),
            chunks: Vec::new(),
            flat_chunks: Vec::new(),
            seed: seed,
        };

        let width = width as i32;

        // 64 0 0 
        
        // Fill grid with empty chunks
        for x in 0..width {
            for y in 0..width {
                for z in 0..width {
                    s.chunks.push(ShaderChunk::new([(64 * x) + origin[0], (64 * y) + origin[1], (64 * z) + origin[2]]));
                }
            }
        }

        s.set_grid_from_chunks();
        s
    }

    // Create grid with existing chunks and width
    pub fn from(chunks: Vec<ShaderChunk>, width: u32, seed: u64) -> Self {
        #[cfg(debug_assertions)]
        if chunks.len() > width.pow(3) as usize { panic!("Chunk depth out of range") }

        let mut s = Self {
            origin: Self::get_origin_from_chunks(&chunks),
            width: width,
            grid: vec![0; (width.pow(3)) as usize],
            chunks: chunks,
            noise: PerlinNoise::new(seed),
            flat_chunks: Vec::new(),
            seed: seed,
        };

        s.set_grid_from_chunks();
        s
    }

    pub fn shift(&mut self, axis: usize, dir: i32) {

        if axis == 1 { panic!("Invalid shift axis") }; 

        let new_axis: i32;
        let remove_axis: i32;

        let alt_axis: usize;

        if axis == 0 {
            alt_axis = 2;
        } else {
            alt_axis = 0;
        }

        // Determine row to chanage, and update origin
        if dir == -1 {
            new_axis = self.origin[axis] - 64;
            remove_axis = self.origin[axis] + (self.width - 1) as i32 * 64;
            self.origin[axis] -= 64;
        }
        else {
            new_axis = self.origin[axis] + (self.width) as i32 * 64;
            remove_axis = self.origin[axis];
            self.origin[axis] += 64;
        }

        // println!("new {new_axis} remove {remove_axis}");

        // INSTEAD OF SERACHING THROUGH SHIZ I CAN JUST DO X Y * W Z * W * W WITH SPECIFIC X OR Z !!!

        // Replace uneeded chunks with new chunks
        for i in 0..self.chunks.len() {
            let mut origin = self.chunks[i].get_origin();

            if origin[axis] == remove_axis {
                origin[axis] = new_axis;
                self.chunks[i] = ShaderChunk::new(origin);

            }
        }

        self.set_grid_from_chunks();

        let mut update_chunk = self.origin;
        if dir == 1 {
            update_chunk[axis] += (self.width - 1) as i32 * 64;
        }


        for i in 0..self.width as usize {

            // Update vertical chunk
            let grid = self.noise.generate_grid_from_point(64, 64, 0.005, (update_chunk[2] as u32, update_chunk[0] as u32));

            for x in 0..64 {
                for z in 0..64 {
                    let y = grid[x as usize][z as usize] * 64.0;

                    let x_adjusted = x + update_chunk[0] as i32;
                    let z_adjusted = z + update_chunk[2]  as i32;

                    let y = (y + 768.0) as i32;
                    if y < 768 {
                        let mut temp = y + 1;
                        while temp < 769 {
                            self.insert_voxel([x_adjusted, temp, z_adjusted], 3);
                            temp += 1;
                        }
                    }

                    self.insert_voxel([x_adjusted, y, z_adjusted], 1);
                    self.insert_voxel([x_adjusted, y - 1, z_adjusted], 2);
                    self.insert_voxel([x_adjusted, y - 2, z_adjusted], 2);
                    self.insert_voxel([x_adjusted, y - 3, z_adjusted], 2);
                }
            }

            let c_ind = self.get_chunk_pos(&update_chunk);

            for j in 0..self.width {
                let grid_index = c_ind[0] as u32 + (c_ind[1] as u32 + j)  * self.width + c_ind[2] as u32 * self.width.pow(2);
                let index = self.grid[grid_index as usize] as usize;
                self.flat_chunks[index] = self.chunks[index].flatten().1;
            }

            //println!("{:?}", update_chunk);

            update_chunk[alt_axis] += 64;
        }
    }

    // Assumes grid has be predfined with the needed size
    fn set_grid_from_chunks(&mut self) {

        // Convert to function in grid
        for (i, chunk) in self.chunks.iter().enumerate() {
            let chunk_pos: [u32; 3] = self.get_chunk_pos(&chunk.get_origin());
            
            let grid_index = chunk_pos[0] + chunk_pos[1] * self.width + chunk_pos[2] * self.width.pow(2);

            self.grid[grid_index as usize] = i as u32; 

        }
    }

    pub fn flatten_world(&mut self) -> (Vec<i32>, Vec<u32>) {
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

            // Store data for later access
            self.flat_chunks.push(flat.1.clone());

            flat_chunks.append(&mut flat.1);
        }

        // Add origin to the flat grid
        flat_grid.splice(0..0, Vec::from(self.origin));

        (flat_grid, flat_chunks)
    }

    // Updates a given chunks stored flat data from a position within the shaderchunk
    pub fn update_flat_chunk_with_world_pos(&mut self, pos: [i32; 3]) {
        let chunk_pos = self.get_chunk_pos(&pos);
        let chunk_index = chunk_pos[0] + chunk_pos[1] * self.width + chunk_pos[2] * (self.width * self.width);
        self.update_flat_chunk_with_chunk_index(chunk_index as usize);
    }

    // Updates a given chunks stored flat data from a given index
    pub fn update_flat_chunk_with_chunk_index(&mut self, index: usize) {
        let i = self.grid[index] as usize;

        self.flat_chunks[i] = self.chunks[i].flatten().1;
    }

    // Returns flattened shader grid, does not do any flattening itself
    pub fn get_flat_world(&self) -> (Vec<i32>, Vec<u32>){
        let mut flat_chunks: Vec<u32> = Vec::new();
        let mut flat_grid: Vec<i32> = vec![0; (self.width.pow(3)) as usize];

        // Accumulated index
        let mut curr_index: i32 = 0;

        // Loop over flat chunk data to rebuild the 
        for i in 0..self.chunks.len() {

            flat_chunks.extend(&self.flat_chunks[i]);
            let c_pos = self.get_chunk_pos(&self.chunks[i].get_origin());

            let grid_index = c_pos[0] + c_pos[1] * self.width + c_pos[2] * self.width * self.width;
            flat_grid[grid_index as usize] = curr_index; 
            curr_index += self.flat_chunks[i].len() as i32;
        }

        // Add origin to the flat grid
        flat_grid.splice(0..0, Vec::from(self.origin));

        (flat_grid, flat_chunks)
    }

    pub fn get_flat_grid(&self) -> Vec<i32> {
        let mut flat_grid: Vec<i32> = vec![0; (self.width.pow(3)) as usize];

        // Accumulated index
        let mut curr_index: i32 = 0;

        // Loop over flat chunk data to rebuild the 
        for i in 0..self.chunks.len() {
            let c_pos = self.get_chunk_pos(&self.chunks[i].get_origin());

            let grid_index = c_pos[0] + c_pos[1] * self.width + c_pos[2] * self.width * self.width;
            flat_grid[grid_index as usize] = curr_index; 
            curr_index += self.flat_chunks[i].len() as i32;
        }

        // Add origin to the flat grid
        flat_grid.splice(0..0, Vec::from(self.origin));

        flat_grid
    }
}

