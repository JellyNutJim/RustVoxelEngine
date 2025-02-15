// Shader Usage
mod shader_chunk;
mod shader_grid;

// Rust usage
mod chunk;

pub use shader_chunk::ShaderChunk;
pub use shader_grid::ShaderGrid;


pub fn get_flat_world() -> (Vec<i32>, Vec<u32>){
    let mut chunks: Vec<ShaderChunk> = vec![
        ShaderChunk::new([0, 0, 0]),
        ShaderChunk::new([0, 64, 0]),
        ShaderChunk::new([0, 0, 64]),
        ShaderChunk::new([0, 64, 64]),
        ShaderChunk::new([64, 0, 0]),
        ShaderChunk::new([64, 64, 0]),
        ShaderChunk::new([64, 0, 64]),
        ShaderChunk::new_with_type([64,64,64], 1)
    ];

    chunks.get_mut(0).unwrap().insert_voxel([1,1,1], 1);
    chunks.get_mut(0).unwrap().insert_subchunk([2,2,2], 4, 1);
    chunks.get_mut(0).unwrap().insert_subchunk([4,4,4], 3, 1);
    chunks.get_mut(0).unwrap().insert_subchunk([8,8,8], 2, 1);
    chunks.get_mut(0).unwrap().insert_subchunk([16,16,16], 1, 1);
    chunks.get_mut(0).unwrap().insert_subchunk([32,32,32], 0, 1);


    let p = ShaderGrid::from(chunks, 2).flatten();

    println!("Grid: {:?}\nChunk Data: {:?}", p.0, p.1);

    p
}