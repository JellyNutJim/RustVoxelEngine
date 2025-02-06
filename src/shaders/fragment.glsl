#version 450
            
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform camera_subbuffer {
    vec3 origin;
    vec3 look_at;
    vec3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;

} camera;

layout(set = 0, binding = 0) uniform Resolution {
    vec2 screen_size;
} u_resolution;

layout(set = 0, binding = 1) buffer VoxelBuffer {
    uint voxels[32][32][32];
} v_buf;


void main() {
    vec2 pixel_coords = vec2(gl_FragCoord.x, 800 - gl_FragCoord.y);
    vec3 pixel_center = camera.pixel00_loc + (camera.pixel_delta_u * pixel_coords.x) + (camera.pixel_delta_v * pixel_coords.y);
    vec3 dir = pixel_center - camera.origin;

    // vec3 world_pos = floor(camera.origin);
    // vec3 step = vec3(
    // dir.x < 0.0 ? -1.0 : 1.0,
    // dir.y < 0.0 ? -1.0 : 1.0, 
    // dir.z < 0.0 ? -1.0 : 1.0
    // );

    // vec3 t_delta = abs(vec3(1.0) / dir);
    // vec3 t_max = vec3(
    // (step.x > 0.0 ? floor(camera.origin.x) + 1.0 : floor(camera.origin.x)) - camera.origin.x,
    // (step.y > 0.0 ? floor(camera.origin.y) + 1.0 : floor(camera.origin.y)) - camera.origin.y,
    // (step.z > 0.0 ? floor(camera.origin.z) + 1.0 : floor(camera.origin.z)) - camera.origin.z
    // ) / dir;

    // vec3 color;
    // int hit_axis = 0;
    // while(length(world_pos) < 1000.0) {
    //     if(v_buf.voxels[uint(abs(world_pos.x))][uint(abs(world_pos.y))][uint(abs(world_pos.z))] == 1u) {
    //         vec3 normal;

    //         if(hit_axis == 0) normal = vec3(-step.x, 0.0, 0.0);
    //         else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
    //         else normal = vec3(0.0, 0.0, -step.z);
            
    //         f_color = vec4((normal + vec3(1.0)) * 0.5, 1.0);
    //         return;
    //     }
        
    //     if(t_max.x < t_max.y) {
    //         if(t_max.x < t_max.z) {
    //             world_pos.x += step.x;
    //             t_max.x += t_delta.x;
    //             hit_axis = 0;
    //         } else {
    //             world_pos.z += step.z;
    //             t_max.z += t_delta.z;
    //             hit_axis = 2;
    //         }
    //     } 
    //     else {
    //         if(t_max.y < t_max.z) {
    //             world_pos.y += step.y;
    //             t_max.y += t_delta.y;
    //             hit_axis = 1;
    //         } else {
    //             world_pos.z += step.z;
    //             t_max.z += t_delta.z;
    //             hit_axis = 2;
    //         }
    //     }
    // }

    float a = (normalize(dir).y + 1.0) * 0.5;
    vec3 color = vec3(1.0) * (1.0 - a) + vec3(0.5, 0.7, 1.0) * (a);
    vec2 tasd = (gl_FragCoord.xy / 800) * 2.0 - 1.0;
    f_color = vec4(normalize(dir), 1.0);
}