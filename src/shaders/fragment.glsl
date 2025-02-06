#version 450
            
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform camera_subbuffer {
    vec3 camera_pos;
    vec3 look_at;
    vec3 up;
    float aspect_ratio;
    float fov;
} camera;

layout(set = 0, binding = 1) buffer VoxelBuffer {
    uint voxels[];  // Dynamic array
};


void main() {

    f_color = vec4(camera.camera_pos.x, uv.y, uv.x, 1.0);
}