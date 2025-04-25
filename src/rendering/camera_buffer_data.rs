use vulkano::buffer::BufferContents;

#[repr(C)]
#[derive(BufferContents, Debug, Clone, Copy)]
#[repr(align(16))] 
pub struct CameraBufferData {
    pub origin: [f32; 4],
    pub pixel00_loc: [f32; 4],
    pub pixel_delta_u: [f32; 4],
    pub pixel_delta_v: [f32; 4],
    pub world_position: [f32; 4],
    pub sun_position: [f32; 4],
    pub time: [f32; 4],
}