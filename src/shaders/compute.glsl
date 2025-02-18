#version 460

layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform camera_subbuffer {
    vec3 origin;
    vec3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 world_pos_1;
} c;


//  TEST IF 3D MATIX OF BIT SHIFTING IS FASTER FOR VOXEL CHECKING!
layout(set = 0, binding = 1) buffer VoxelBuffer {
    uint voxels[409293];
} v_buf;

layout(set = 0, binding = 2) buffer WorldBuffer {
    ivec3 origin;
    uint chunks[27];
} w_buf;

layout(set = 0, binding = 3, rgba8) uniform image2D storageImage;

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



uint get_depth(vec3 pos, inout int multiplier) {


    //ivec3 rel = ivec3(floor((pos) / 64)) - (w_buf.origin) / 64;

    ivec3 rel = ivec3(
        ((pos.x - (int(pos.x) & 63)) / 64) - (w_buf.origin.x / 64),
        ((pos.y - (int(pos.y) & 63)) / 64) - (w_buf.origin.y / 64),
        ((pos.z - (int(pos.z) & 63)) / 64) - (w_buf.origin.z / 64)
    );


    // Currently defining here -> Will switch this out for buffer input
    int WIDTH = 20;


    // Chunk if chunk is within current grid bounds
    if (any(greaterThan(rel, ivec3(WIDTH - 1)))) {
        return 100;
    }

    if (any(lessThan(rel, ivec3(0)))) {
        return 100;
    }

    uvec3 realitive_chunk_location = uvec3(rel);

    // Voxel position in the current chunk
    //vec3 voxel_pos = vec3(0.0);


    vec3 local_pos = mod(mod(pos, 64.0) + vec3(64.0), 64.0);
    //vec3 local_pos = mod(pos, 64.0);


    uint index = realitive_chunk_location.x + realitive_chunk_location.y * WIDTH + realitive_chunk_location.z * WIDTH * WIDTH;
    index = w_buf.chunks[index];

    if (v_buf.voxels[index] == 0) {
        multiplier =  64;
        return v_buf.voxels[index + 1];
    }

    uint octant = get_octant(mod(local_pos, 64), 31);
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] == 0) {
       multiplier = 32;
        return v_buf.voxels[index + 1];
    }

    octant = get_octant(mod(local_pos, 32), 15);
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] == 0) {
        multiplier = 16;
        return v_buf.voxels[index + 1];
    }

    octant = get_octant(mod(local_pos, 16), 7);
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] == 0) {
        multiplier =  8;
        return v_buf.voxels[index + 1];
    }

    octant = get_octant(mod(local_pos, 8), 3);
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] == 0) {
        multiplier = 4;
        return v_buf.voxels[index + 1];
    }

    octant = get_octant(mod(local_pos, 4), 1);
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] == 0) {
        multiplier = 2;
        return v_buf.voxels[index + 1];
    }

    octant = get_octant(mod(local_pos, 2), 0);
    index = index + v_buf.voxels[index + octant + 1];

    multiplier = 1;
    return v_buf.voxels[index + 1];
}

void take_step(ivec3 step, vec3 t_delta, inout vec3 t_max, inout uint hit_axis, inout vec3 world_pos, int multiplier, vec3 dir) {

    if (multiplier > 1) {
        float minT = 1e10;

        //vec3 origin = world_pos + fract(c.origin);


        // vec3 origin = c.origin;
        // vec3 relativePos = world_pos - floor(origin);
        // origin = relativePos;

        // vec3 origin = world_pos + vec3(
        //     (step.x > 0 ? fract(c.origin.x) : 1.0 - fract(c.origin.x)),
        //     (step.y > 0 ? fract(c.origin.y) : 1.0 - fract(c.origin.y)),
        //     (step.z > 0 ? fract(c.origin.z) : 1.0 - fract(c.origin.z))
        // );

        float t;

        if (hit_axis == 0) {
            t = t_max.x;
        }
        else if (hit_axis == 1) {
            t = t_max.y;
        }
        else {
            t = t_max.z;
        }


        vec3 origin = c.origin + dir * t;


        //vec3 origin = c.origin + world_pos;


        for (int i = 0; i < 3; i++) {
            if (dir[i] != 0.0) {
                float nextBoundary = dir[i] > 0.0 ? 
                    multiplier * (floor(origin[i] / multiplier) + 1.0) : 
                    multiplier * floor(origin[i] / multiplier);
                
                float t = (nextBoundary - origin[i]) / dir[i];
                if (t > 0.0 && t < minT) {
                    minT = t;
                    hit_axis = i;
                }
            } 
        }
        
        // Get position just before intersection
        vec3 pos = origin + dir * (minT - 0.001);
        vec3 temp = floor(pos);
        
        t_max += abs(temp - world_pos) * t_delta;
        world_pos = temp;
    }



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

vec4 get_colour(uint hit_axis, ivec3 step, vec3 c) {
    vec3 normal;

    if(hit_axis == 0) normal = vec3(-step.x, 0.0, 0.0);
    else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
    else normal = vec3(0.0, 0.0, -step.z);

    return vec4((normal + c) * 0.5, 1.0);
}

vec4 stone(uint hit_axis, ivec3 step) {
    vec3 normal;

    if(hit_axis == 0) normal = vec3(-step.x, 0.0, 0.0);
    else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
    else normal = vec3(0.0, 0.0, -step.z);

    return vec4((vec3(0.7, 0.71, 0.7) * 0.3 + normal * 0.1), 1.0);
}

vec4 grass(uint hit_axis, ivec3 step) {
        vec3 normal;

    if(hit_axis == 0) normal = vec3(0.0, 0.0, 0.0);
    else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
    else normal = vec3(0.0, 0.0, 0.0);

    return vec4((normal + vec3(0.0, 0.5, 0.1)) * 0.5, 1.0);
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

    int multiplier;

    while (steps < 640) {
        // Go through chunks

        uint current_depth = get_depth(world_pos, multiplier);

        if (current_depth == 100) { // Empty space or out of bounds
            multiplier = 1;
            steps += 1;
            take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir);
            continue;
        }

        if (current_depth == 1) {
            imageStore(storageImage, pixel_coords, grass(hit_axis, step) );
            return;
        }

        if (current_depth == 2) {
            imageStore(storageImage, pixel_coords, stone(hit_axis, step));
            return;
        }

        steps += 1;
        take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir);
    }


    float a = (normalize(dir).y + 1.0) * 0.5;
    vec3 colour = vec3(1.0) * (1.0 - a) + vec3(0.5, 0.7, 1.0) * (a);
    output_colour = vec4(colour, 1.0);
    imageStore(storageImage, pixel_coords, output_colour);
}





        // if (current_depth == 0) {
        //     imageStore(storageImage, pixel_coords, get_colour(hit_axis, step));
        //     return;
        // }
        // if (current_depth >= 1 && current_depth <= 7) {
        //     vec3 depth_color = mix(
        //         vec3(0.071, 0.173, 0.365),  // Dark blue
        //         vec3(0.937, 0.235, 0.251),  // Coral red
        //         float(current_depth) / 7.0
        //     );
        //     imageStore(storageImage, pixel_coords, vec4(depth_color, 1.0));
        //     return;
        // }