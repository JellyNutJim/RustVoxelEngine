#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform camera_subbuffer {
    float data[15];
} c;


//  TEST IF 3D MATIX OF BIT SHIFTING IS FASTER FOR VOXEL CHECKING!
layout(set = 0, binding = 1) buffer VoxelBuffer {
    uint voxels[32][32][32];
} v_buf;

layout(set = 0, binding = 2, rgba8) uniform image2D storageImage;

float clamp(float num) {
    if (num < 0.0) {
        return 0.0;
    } 
    else if (num > 0.999) {
        return 0.999;
    }

    return num;
}


void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    // vec3 pixel_center = c.pixel00_loc + (c.pixel_delta_u * float(pixel_coords.x)) + (c.pixel_delta_v * float(pixel_coords.y));
    // vec3 dir = normalize(pixel_center - c.origin);

    vec4 test = vec4(0.0, 0.0, 1.0, 1.0); 
    vec4 output_colour = vec4(c.data[5], c.data[6], c.data[7], 1.0); 

    imageStore(storageImage, pixel_coords, output_colour);
    return;

    // vec3 world_pos = floor(c.origin);
    // vec3 t_delta = abs(vec3(1.0) / dir);

    // ivec3 step = ivec3(
    //     dir.x < 0.0 ? -1 : 1,
    //     dir.y < 0.0 ? -1 : 1, 
    //     dir.z < 0.0 ? -1 : 1
    // );

    // vec3 t_max = vec3(
    //     (step.x > 0 ?  world_pos.x + 1.0 : world_pos.x) - c.origin.x / dir.x,
    //     (step.y > 0 ?  world_pos.y + 1.0 : world_pos.y) - c.origin.y / dir.y,
    //     (step.z > 0 ?  world_pos.z + 1.0 : world_pos.z) - c.origin.z / dir.z
    // );

    // uint hit_axis = 0;

    // while(length(world_pos) < 30.0) {

    //     if (v_buf.voxels[uint(abs(world_pos.x))][uint(abs(world_pos.y))][uint(abs(world_pos.z))] == 1) {

    //         vec3 normal;

    //         if(hit_axis == 0) normal = vec3(-step.x, 0.0, 0.0);
    //         else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
    //         else normal = vec3(0.0, 0.0, -step.z);

    //         output_colour = vec4((normal + vec3(1.0)) * 0.5, 1.0);
    //         output_colour = vec4(0.0, 0.0, 0.0, 1.0);
    //         imageStore(storageImage, pixel_coords, output_colour);
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
    

    // float a = (normalize(dir).y + 1.0) * 0.5;
    // vec3 colour = vec3(1.0) * (1.0 - a) + vec3(0.5, 0.7, 1.0) * (a);
    // output_colour = vec4(clamp(colour.x), clamp(colour.y), clamp(colour.z), 1.0);

    // imageStore(storageImage, pixel_coords, output_colour);
}