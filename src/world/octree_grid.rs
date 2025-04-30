use std::i32::MAX;

use crate::{types::Geometry, world::Octree};

use super::{
    chunk_generator::*,
    Voxel,
};


// Holds data to be placed in the voxel buffer
// Origin will always be smaller than the current position
#[repr(C)]
#[derive(Debug, Clone)]
pub struct OctreeGrid {

    // Meta Data
    pub origin: [i32; 3],  // Origin of the current grid = the origin of the chunk with the lowest positional value 
    pub width: u32,
    pub sea_level: u32,
    pub generator: GenPipeLine,
    pub spatial_map: Vec<u32>, // Grid to relate world location to memory location
    pub seed: u64,


    // World Data
    pub trees: Vec<Octree>,

    // Chunks as array of u32
    flat_chunks: Vec<Vec<u32>>,

    // Layer Sizes
    res_8_layer_width: u32,
    res_4_layer_width: u32,
    res_2_layer_width: u32,
    res_1_layer_width: u32,
}

// Layer sizes -> Will be dynamically determined at some point

#[allow(unused)]
impl OctreeGrid {
    // Create grid with given origin and size
    pub fn new(width: u32, origin: [i32; 3], seed: u64, sea_level: u32, res_8_layer_width: u32, res_4_layer_width: u32, res_2_layer_width: u32, res_1_layer_width: u32) -> Self {
        let mut s = Self {
            origin: origin,
            width: width,
            sea_level: sea_level,
            spatial_map: vec![0; (width.pow(3)) as usize],
            generator: GenPipeLine::new(seed, width as usize, sea_level as f64),
            trees: Vec::new(),
            flat_chunks: Vec::new(),
            seed: seed,

            res_8_layer_width,
            res_4_layer_width,
            res_2_layer_width,
            res_1_layer_width,
        };

        let width = width as i32;

        // 64 0 0 
        
        // Fill grid with empty chunks
        for x in 0..width {
            for y in 0..width {
                for z in 0..width {
                    s.trees.push(Octree::new([(64 * x) + origin[0], (64 * y) + origin[1], (64 * z) + origin[2]]));
                }
            }
        }

        s.set_grid_from_chunks();
        s
    }

    pub fn get_res_8_width(&self) -> u32{
        self.res_8_layer_width
    }

    pub fn get_res_4_width(&self) -> u32{
        self.res_4_layer_width
    }

    pub fn get_res_2_width(&self) -> u32{
        self.res_2_layer_width
    }

    pub fn get_res_1_width(&self) -> u32{
        self.res_1_layer_width
    }

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

    pub fn insert_geometry(&mut self, pos: [i32; 3], geometry: Geometry, ground_insert: bool) {

        let chunk_pos = self.get_chunk_pos(&pos);
        let chunk_index = chunk_pos[0] + chunk_pos[1] * self.width + chunk_pos[2] * (self.width * self.width);
        
        // Local voxel pos in chunk
        let pos = [
            (((pos[0] % 64) + 64) % 64) as u32,
            (((pos[1] % 64) + 64) % 64) as u32,
            (((pos[2] % 64) + 64) % 64) as u32
        ];

        self.trees[self.spatial_map[chunk_index as usize] as usize].insert_geometry(pos, geometry, ground_insert);
    }

    pub fn insert_subchunk(&mut self, pos: [i32; 3], geometry: Geometry, depth: u32, ground_insert: bool) {
        let chunk_pos = self.get_chunk_pos(&pos);
        let chunk_index = chunk_pos[0] + chunk_pos[1] * self.width + chunk_pos[2] * (self.width * self.width);
        
        // Local voxel pos in chunk
        let pos = [
            (((pos[0] % 64) + 64) % 64) as u32,
            (((pos[1] % 64) + 64) % 64) as u32,
            (((pos[2] % 64) + 64) % 64) as u32
        ];

        self.trees[self.spatial_map[chunk_index as usize] as usize].insert_subchunk(pos, depth, geometry, ground_insert);
    }

    pub fn insert_simple_voxel(&mut self, pos: [i32; 3], voxel_type: u32) {
        let chunk_pos = self.get_chunk_pos(&pos);
        let chunk_index = chunk_pos[0] + chunk_pos[1] * self.width + chunk_pos[2] * (self.width * self.width);
        let pos = [
            (((pos[0] % 64) + 64) % 64) as u32,
            (((pos[1] % 64) + 64) % 64) as u32,
            (((pos[2] % 64) + 64) % 64) as u32
        ];
        self.trees[self.spatial_map[chunk_index as usize] as usize].insert_geometry(pos, Geometry::Voxel(Voxel::from(voxel_type)), false);
    }

    pub fn insert_simple_subchunk(&mut self, pos: [i32; 3], voxel_type: u32, depth: u32) {
        let chunk_pos = self.get_chunk_pos(&pos);
        let chunk_index = chunk_pos[0] + chunk_pos[1] * self.width + chunk_pos[2] * (self.width * self.width);
        let pos = [
            (((pos[0] % 64) + 64) % 64) as u32,
            (((pos[1] % 64) + 64) % 64) as u32,
            (((pos[2] % 64) + 64) % 64) as u32
        ];
        self.trees[self.spatial_map[chunk_index as usize] as usize].insert_subchunk(pos, depth, Geometry::Voxel(Voxel::from(voxel_type)), false);
    }

    // Finds the smallest chunk origin, and sets that to the grid origin
    fn get_origin_from_chunks(shader_chunks: &Vec<Octree>) -> [i32; 3] {
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

    pub fn get_chunk_pos_u32(&self, pos: &[u32; 3]) -> [u32; 3] {
        [
            (((pos[0] - (pos[0] & 63)) / 64) - self.origin[0] as u32 / 64) as u32, 
            (((pos[1] - (pos[1] & 63)) / 64) - self.origin[1] as u32 / 64) as u32, 
            (((pos[2] - (pos[2] & 63)) / 64) - self.origin[2] as u32 / 64) as u32
        ]
    }

    // Assumes grid has be predfined with the needed size
    fn set_grid_from_chunks(&mut self) {

        // Convert to function in grid
        for (i, chunk) in self.trees.iter().enumerate() {
            let chunk_pos: [u32; 3] = self.get_chunk_pos(&chunk.get_origin());
            
            let grid_index = chunk_pos[0] + chunk_pos[1] * self.width + chunk_pos[2] * self.width.pow(2);

            self.spatial_map[grid_index as usize] = i as u32; 

        }
    }

    // Intial world flatten
    pub fn flatten_world(&mut self) -> (Vec<i32>, Vec<u32>) {
        let mut flat_chunks: Vec<u32> = Vec::new();
        let mut flat_grid: Vec<i32> = vec![0; (self.width.pow(3)) as usize];

        // Accumulated index
        let mut curr_index: i32 = 0;

        // Convert to function in grid
        for chunk in &self.trees {
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
        let i = self.spatial_map[index] as usize;

        self.flat_chunks[i] = self.trees[i].flatten().1;
    }

    // Returns flattened shader grid, does not do any flattening itself
    pub fn get_flat_world(&self) -> (Vec<i32>, Vec<u32>){
        let mut flat_chunks: Vec<u32> = Vec::new();
        let mut flat_grid: Vec<i32> = vec![0; (self.width.pow(3)) as usize];

        // Accumulated index
        let mut curr_index: i32 = 0;

        // Loop over flat chunk data to rebuild the 
        for i in 0..self.trees.len() {

            flat_chunks.extend(&self.flat_chunks[i]);
            let c_pos = self.get_chunk_pos(&self.trees[i].get_origin());

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
        for i in 0..self.trees.len() {
            let c_pos = self.get_chunk_pos(&self.trees[i].get_origin());

            let grid_index = c_pos[0] + c_pos[1] * self.width + c_pos[2] * self.width * self.width;
            flat_grid[grid_index as usize] = curr_index; 
            curr_index += self.flat_chunks[i].len() as i32;
        }

        // Add origin to the flat grid
        flat_grid.splice(0..0, Vec::from(self.origin));

        flat_grid
    }

    pub fn get_centre_point_in_space(&self) -> [u32; 3] {
        [ 
            (self.origin[0] + (((self.width as i32) / 2 + 1) * 64)) as u32,
            (self.origin[1] + (((self.width as i32) / 2 + 1) * 64)) as u32,
            (self.origin[2] + (((self.width as i32) / 2 + 1) * 64)) as u32,
        ]
    }

    pub fn get_x_z_midpoint_in_space(&self) -> [u32; 2] {
        [ 
            (self.origin[0] + (((self.width as i32) / 2 + 1) * 64)) as u32,
            (self.origin[2] + (((self.width as i32) / 2 + 1) * 64)) as u32,
        ]
    }


}




// GENERATION PIPELINE
impl OctreeGrid {
    pub fn shift(&mut self, axis: usize, dir: i32) {
        return;
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

        // INSTEAD OF SERACHING THROUGH I CAN JUST DO X Y * W Z * W * W WITH SPECIFIC X OR Z !!!

        // Replace uneeded chunks with new chunks
        for i in 0..self.trees.len() {
            let mut origin = self.trees[i].get_origin();

            if origin[axis] == remove_axis {
                origin[axis] = new_axis;
                self.trees[i] = Octree::new(origin);

            }
        }

        // Update grid to reflect new chunk positions
        self.set_grid_from_chunks();

        // Update height memory locations to match shifted chunks
        self.generator.shift_maps(axis, dir);

        // Determine row of chunks to populate with new terrain data
        let mut update_chunk = self.origin;
        if dir == 1 {
            update_chunk[axis] += (self.width - 1) as i32 * 64;
        }

        // let temp = [
        //     self.origin[0] as u32,
        //     self.origin[1] as u32,
        //     self.origin[2] as u32,
        // ];

        // Populate moved chunks with new data, loops through the row of chunks to be generated
        for _i in 0..self.width as usize {

            // Update Biome MaP for this chunk

            generate_res_8(self, update_chunk[0] as u32, update_chunk[2] as u32);

            let c_ind = self.get_chunk_pos(&update_chunk);

            // Update Flat Chunks
            for j in 0..self.width {
                let grid_index = c_ind[0] as u32 + (c_ind[1] as u32 + j)  * self.width + c_ind[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                self.flat_chunks[index] = self.trees[index].flatten().1;
            }

            // Move to next chunk
            update_chunk[alt_axis] += 64;
        }

        // Now the world has been shifted, multiresolution work can take place
        self.complete_biome_heightmap_gen(dir, axis, alt_axis);

        // Calculates the final per subvoxel heightmap data, inserts features
        self.merge_biome(dir, axis, alt_axis);

        // Simply inserts the detail generated at final adjust at a higher resolution
        self.insert_inner(dir, axis, alt_axis);


    }

    fn complete_biome_heightmap_gen(&mut self, dir: i32, axis: usize, alt_axis: usize) {
        let layer_width = self.get_res_4_width();
        let layer_half_width = layer_width / 2;

        let mid_xz = self.get_x_z_midpoint_in_space();
        let mid_chunk = [(mid_xz[0] / 64) * 64, (mid_xz[1] / 64) * 64];

        let mut update_chunk = [
            (mid_chunk[0] - layer_half_width * 64) as i32,
            self.origin[1],
            (mid_chunk[1] - layer_half_width * 64) as i32
        ];

        let mut delete_chunk = [
            update_chunk[0],
            update_chunk[1],
            update_chunk[2]
        ];

        if dir == 1 {
            update_chunk[axis] += ((layer_width - 1) * 64) as i32;
            delete_chunk[axis] += (-1 * 64) as i32;
        }
        else {
            delete_chunk[axis] += ((layer_width + 1) * 64) as i32;
        }

        for _i in 0..layer_width as usize {

            let update_chunk_pos   = self.get_chunk_pos(&update_chunk);
            let delete_chunk_pos = self.get_chunk_pos(&delete_chunk);

            // Temp solution of just deleting chunks at layer boundaries
            for j in 0..self.width {
                let grid_index = update_chunk_pos[0] as u32 + (update_chunk_pos[1] as u32 + j)  * self.width + update_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                self.trees[index] = Octree::new(self.trees[index].get_origin());

                self.trees[index].set_generation_level(3);

                let grid_index = delete_chunk_pos[0] as u32 + (delete_chunk_pos[1] as u32 + j)  * self.width + delete_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                self.trees[index] = Octree::new(self.trees[index].get_origin());

                self.trees[index].set_generation_level(2);
            }

            generate_res_4(self, update_chunk[0] as u32, update_chunk[2] as u32);

            // Chunks outside this border are reset
            generate_res_8(self, delete_chunk[0] as u32, delete_chunk[2] as u32);


            // Update Flat Chunk Column
            for j in 0..self.width {
                let grid_index = update_chunk_pos[0] as u32 + (update_chunk_pos[1] as u32 + j)  * self.width + update_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                self.flat_chunks[index] = self.trees[index].flatten().1;

                let grid_index = delete_chunk_pos[0] as u32 + (delete_chunk_pos[1] as u32 + j)  * self.width + delete_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                self.flat_chunks[index] = self.trees[index].flatten().1;
            }

            // Move to next chunk along axis
            update_chunk[alt_axis] += 64;
            delete_chunk[alt_axis] += 64;
        }


    }

    // Updates
    // L
    fn merge_biome(&mut self, dir: i32, axis: usize, alt_axis: usize) {
        let layer_width = self.get_res_2_width();
        let layer_half_width = layer_width / 2;

        let mid_xz = self.get_x_z_midpoint_in_space();
        let mid_chunk = [(mid_xz[0] / 64) * 64, (mid_xz[1] / 64) * 64];

        let mut update_chunk = [
            (mid_chunk[0] - layer_half_width * 64) as i32,
            self.origin[1],
            (mid_chunk[1] - layer_half_width * 64) as i32
        ];

        let mut delete_chunk = [
            update_chunk[0],
            update_chunk[1],
            update_chunk[2]
        ];

        if dir == 1 {
            update_chunk[axis] += ((layer_width - 1) * 64) as i32;
            delete_chunk[axis] += (-1 * 64) as i32;
        }
        else {
            delete_chunk[axis] += ((layer_width + 1) * 64) as i32;
        }

        for _i in 0..layer_width as usize {

            let update_chunk_pos   = self.get_chunk_pos(&update_chunk);
            let delete_chunk_pos = self.get_chunk_pos(&delete_chunk);

            let mut generate = false;

            // Temp solution of just deleting chunks at layer boundaries
            for j in 0..self.width {
                let grid_index = update_chunk_pos[0] as u32 + (update_chunk_pos[1] as u32 + j)  * self.width + update_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;

                let chunk = &mut self.trees[index];
                if chunk.get_max_generation_level() == 3 {
                    generate = true;
                    self.trees[index].set_generation_level(4);
                } 

                let grid_index = delete_chunk_pos[0] as u32 + (delete_chunk_pos[1] as u32 + j)  * self.width + delete_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                self.trees[index] = Octree::new(self.trees[index].get_origin());

                self.trees[index].set_generation_level(3);
            }

            if generate {
                generate_res_2(self, update_chunk[0] as u32, update_chunk[2] as u32);
            }

            generate_res_4(self, delete_chunk[0] as u32, delete_chunk[2] as u32);

            // Update Flat Chunk Column
            for j in 0..self.width {
                let grid_index = update_chunk_pos[0] as u32 + (update_chunk_pos[1] as u32 + j)  * self.width + update_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                self.flat_chunks[index] = self.trees[index].flatten().1;

                let grid_index = delete_chunk_pos[0] as u32 + (delete_chunk_pos[1] as u32 + j)  * self.width + delete_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                self.flat_chunks[index] = self.trees[index].flatten().1;
            }

            // Move to next chunk along axis
            update_chunk[alt_axis] += 64;
            delete_chunk[alt_axis] += 64;
        }
    }


    // Close Detail
    fn insert_inner(&mut self, dir: i32, axis: usize, alt_axis: usize) {
        let layer_width = self.get_res_1_width();
        let layer_half_width = layer_width / 2;

        let mid_xz = self.get_x_z_midpoint_in_space();
        let mid_chunk = [(mid_xz[0] / 64) * 64, (mid_xz[1] / 64) * 64];

        let mut update_chunk = [
            (mid_chunk[0] - layer_half_width * 64) as i32,
            self.origin[1],
            (mid_chunk[1] - layer_half_width * 64) as i32
        ];

        let mut delete_chunk = [
            update_chunk[0],
            update_chunk[1],
            update_chunk[2]
        ];

        if dir == 1 {
            update_chunk[axis] += ((layer_width - 1) * 64) as i32;
            delete_chunk[axis] += (-1 * 64) as i32;
        }
        else {
            delete_chunk[axis] += ((layer_width + 1) * 64) as i32;
        }
        
        // Populate selected chunks with new data
        // Replace old chunks with lower res data
        for _i in 0..layer_width as usize {

            let update_chunk_pos   = self.get_chunk_pos(&update_chunk);
            let delete_chunk_pos = self.get_chunk_pos(&delete_chunk);

            let mut generate = false;

            // Temp solution of just deleting chunks at layer boundaries
            for j in 0..self.width {
                let grid_index = update_chunk_pos[0] as u32 + (update_chunk_pos[1] as u32 + j)  * self.width + update_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                let chunk = &mut self.trees[index];

                if chunk.get_max_generation_level() == 4 {
                    generate = true;
                    self.trees[index].set_generation_level(5);
                } 

                let grid_index = delete_chunk_pos[0] as u32 + (delete_chunk_pos[1] as u32 + j)  * self.width + delete_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                self.trees[index].set_generation_level(4);
            }

            if generate {
                generate_res_1(self, update_chunk[0] as u32, update_chunk[2] as u32);
            }

            // Update Flat Chunk Column
            for j in 0..self.width {
                let grid_index = update_chunk_pos[0] as u32 + (update_chunk_pos[1] as u32 + j)  * self.width + update_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                self.flat_chunks[index] = self.trees[index].flatten().1;

                let grid_index = delete_chunk_pos[0] as u32 + (delete_chunk_pos[1] as u32 + j)  * self.width + delete_chunk_pos[2] as u32 * self.width.pow(2);
                let index = self.spatial_map[grid_index as usize] as usize;
                self.flat_chunks[index] = self.trees[index].flatten().1;
            }

            // Move to next chunk along axis
            update_chunk[alt_axis] += 64;
            delete_chunk[alt_axis] += 64;
        }

    }
}