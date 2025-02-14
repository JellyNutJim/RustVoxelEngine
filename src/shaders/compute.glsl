#version 460

layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform camera_subbuffer {
    vec3 origin;
    vec3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;

    vec3 world_pos_1;
    vec3 world_pos_2;
    vec3 world_pos_4;
    vec3 world_pos_8;
    vec3 world_pos_16;
    vec3 world_pos_32;
    vec3 world_pos_64;
} c;


//  TEST IF 3D MATIX OF BIT SHIFTING IS FASTER FOR VOXEL CHECKING!
layout(set = 0, binding = 1) buffer VoxelBuffer {
    uint voxels[131];
} v_buf;

layout(set = 0, binding = 2) buffer WorldBuffer {
    ivec3 origin;
    uint chunks[8];
} w_buf;

layout(set = 0, binding = 3, rgba8) uniform image2D storageImage;

// Input must be 64 chunk based -> CURRENTLY ASSUME CHUNK WIDTH OF 2
int get_current_chunk_index(vec3 curr_pos) {
    ivec3 t = (ivec3(curr_pos) / 64) - w_buf.origin;

    if (any(greaterThan(t, vec3(1)))) {
        return 100;
    }

    if (any(lessThan(t, vec3(0)))) {
        return 100;
    }

    return t.x + t.y * 2 + t.z * 4;
}

uint get_octant(vec3 pos, uint mid) {
    uint octant = 0;
    if (pos.x > mid) {
        octant += 1;
    }

    if (pos.y > mid) {
        octant += 4;
    }

    if (pos.z > mid) {
        octant += 2;
    }

    return octant;
}



uint get_depth(vec3 pos) {
    if (pos.x < 0 || pos.y < 0 || pos.z < 0) {
        return 100;
    }

    ivec3 realtive_chunk_location = ivec3(floor((pos) / 64)) - w_buf.origin;
    ivec3 chunk_location = ivec3((floor((pos) / 64)) * 64);

    if (any(greaterThan(realtive_chunk_location, vec3(1)))) {
        return 100;
    }

    if (any(lessThan(realtive_chunk_location, vec3(0)))) {
        return 100;
    }

    vec3 local_pos = mod(abs(pos), 64.0);

    uint index = realtive_chunk_location.x + realtive_chunk_location.y * 2 + realtive_chunk_location.z * 4;
    index = w_buf.chunks[index];

    if (v_buf.voxels[index] == 0) {
        return 0;
    }

    uint octant = get_octant(mod(local_pos, 64), 31);
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] == 0) {
        return 1;
    }

    octant = get_octant(mod(local_pos, 32), 15);
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] == 0) {
        return 2;
    }

    octant = get_octant(mod(local_pos, 16), 7);
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] == 0) {
        return 3;
    }

    octant = get_octant(mod(local_pos, 8), 3);
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] == 0) {
        return 4;
    }

    octant = get_octant(mod(local_pos, 4), 1);
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] == 0) {
        return 5;
    }

    octant = get_octant(mod(local_pos, 2), 0);
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index + 1] == 1) {
        return 7;
    }
    return 6;
}

void take_step(ivec3 step, vec3 t_delta, inout vec3 t_max, inout uint hit_axis, inout vec3 world_pos) {
    if(t_max.x < t_max.y) {
        if(t_max.x < t_max.z) {
            world_pos.x += step.x;
            t_max.x += t_delta.x;
            hit_axis = 0;
        } else {
            world_pos.z += step.z;
            t_max.z += t_delta.z;
            hit_axis = 2;
        }
    } 
    else {
        if(t_max.y < t_max.z) {
            world_pos.y += step.y;
            t_max.y += t_delta.y;
            hit_axis = 1;
        } else {
            world_pos.z += step.z;
            t_max.z += t_delta.z;
            hit_axis = 2;
        }
    }
}

vec4 get_colour(uint hit_axis, ivec3 step) {
    vec3 normal;

    if(hit_axis == 0) normal = vec3(-step.x, 0.0, 0.0);
    else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
    else normal = vec3(0.0, 0.0, -step.z);

    return vec4((normal + vec3(1.0)) * 0.5, 1.0);
}


void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    vec3 origin = c.origin;

    vec3 pixel_center = c.pixel00_loc + (c.pixel_delta_u * float(pixel_coords.x)) + (c.pixel_delta_v * float(pixel_coords.y));
    vec3 dir = pixel_center - origin;

    vec4 output_colour = vec4(1.0);
    vec3 world_pos = c.world_pos_1;

    vec3 t_delta = abs(vec3(1.0) / dir);
    ivec3 step;
    vec3 t_max;


    if (dir.x < 0.0) {
        step.x = -1;
        t_max.x = ((world_pos.x) - origin.x);
    }
    else {
        step.x = 1;
        t_max.x = ((world_pos.x + 1.0) - origin.x);
    }

    if (dir.y < 0.0) {
        step.y = -1;
        t_max.y = ((world_pos.y) - origin.y);
    }
    else {
        step.y = 1;
        t_max.y = ((world_pos.y + 1.0) - origin.y);
    }

    if (dir.z < 0.0) {
        step.z = -1;
        t_max.z = ((world_pos.z) - origin.z);
    }
    else {
        step.z = 1;
        t_max.z = ((world_pos.z + 1.0) - origin.z);
    }

    t_max /= dir;


    uint steps = 0;
    uint hit_axis = 0;

    steps = 0;

    while (steps < 300) {
        if (world_pos.x > 0 && world_pos.y > 0 && world_pos.z > 0) {
            steps += 1;
            take_step(step, t_delta, t_max, hit_axis, world_pos);
            continue;
        }

        //uint c_pos = get_current_chunk_index(world_pos_64);

        //if (c_pos > 7) { break; }

        //c_pos = w_buf.chunks[c_pos];

        // Go through chunks
        // Check if current chunk has octants

        uint current_depth = get_depth(world_pos);
        if (current_depth < 0) {
            break;
        }

        if (current_depth == 0) {
            imageStore(storageImage, pixel_coords, get_colour(hit_axis, step));
            return;
        }
        if (current_depth >= 1 && current_depth <= 7) {
            vec3 depth_color = mix(
                vec3(0.071, 0.173, 0.365),  // Dark blue
                vec3(0.937, 0.235, 0.251),  // Coral red
                float(current_depth) / 7.0
            );
            imageStore(storageImage, pixel_coords, vec4(depth_color, 1.0));
            return;
        }

        steps += 1;
        take_step(step, t_delta, t_max, hit_axis, world_pos);
    }


    float a = (normalize(dir).y + 1.0) * 0.5;
    vec3 colour = vec3(1.0) * (1.0 - a) + vec3(0.5, 0.7, 1.0) * (a);
    output_colour = vec4(colour, 1.0);
    imageStore(storageImage, pixel_coords, output_colour);
}