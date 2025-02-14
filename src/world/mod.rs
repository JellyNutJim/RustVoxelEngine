// Shader Usage
mod shader_chunk;
mod shader_grid;

// Rust usage
mod chunk;

pub use shader_chunk::ShaderChunk;
pub use shader_grid::ShaderGrid;


pub fn get_world() -> (ShaderGrid, Vec<u32>){
    let mut chunks: Vec<ShaderChunk> = vec![
        ShaderChunk::new([0, 0, 0]),
        ShaderChunk::new([0, 64, 0]),
        ShaderChunk::new([0, 0, 64]),
        ShaderChunk::new([0, 64, 64]),
        ShaderChunk::new([64, 0, 0]),
        ShaderChunk::new([64, 64, 0]),
        ShaderChunk::new([64, 0, 64]),
        ShaderChunk::new([64, 64, 0]),
    ];

    chunks.get_mut(0).unwrap().insert_voxel([0,0,0], 1);
    chunks.get_mut(1).unwrap().insert_voxel([0,64,0], 1);
    chunks.get_mut(2).unwrap().insert_voxel([0,0,64], 1);
    chunks.get_mut(3).unwrap().insert_voxel([0,64,64], 1);
    
    //chunks.get_mut(1).unwrap().insert_voxel([0,64,0], 1);
    //chunks.get_mut(0).unwrap().insert_voxel([0,64,0], 1);
    //chunks.get_mut(0).unwrap().insert_voxel([0,64,0], 1);
    //chunks.get_mut(0).unwrap().insert_subchunk([0,0,0], 0, 1);
    //chunks.get_mut(0).unwrap().insert_voxel([0,0,0], 69420);

    //chunks.get_mut(7).unwrap().insert_subchunk(, 0, 1);

    ShaderGrid::from(&chunks)
}