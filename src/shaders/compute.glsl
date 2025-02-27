#version 460

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform camera_subbuffer {
    vec3 origin;
    vec3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 world_pos_1;
    vec3 sun_loc;
} c;


//  TEST IF 3D MATIX OF BIT SHIFTING IS FASTER FOR VOXEL CHECKING!
layout(set = 0, binding = 1) readonly buffer VoxelBuffer {
    uint voxels[409293];
} v_buf;

layout(set = 0, binding = 2) readonly buffer WorldBuffer {
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
    uvec3 realitive_chunk_location = uvec3((floor((pos) / 64)) - (w_buf.origin) / 64);

    // Currently defining here -> Will switch this out for buffer input
    int WIDTH = 41;

    //vec3 local_pos = mod(mod(pos , 64.0) + vec3(64.0), 64.0);

    vec3 local_pos = mod(pos, 64.0);

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
    index = index + octant + 1;

    multiplier = 1;
    return v_buf.voxels[index];
}

void take_step(ivec3 step, vec3 t_delta, inout vec3 t_max, inout uint hit_axis, inout vec3 world_pos, int multiplier, vec3 dir, inout float curr_distance) {

    if (multiplier > 4) {
        float minT = 1e10;

        // Change so curr_distance adjustment happens here if previous hit dir was negative
        bool adjust = false;

        // Add to switch
        if (dir[hit_axis] < 0.0) {
            adjust = true;
            curr_distance += 0.0001;
        }
        
        // Remove to switch
        //vec3 origin = c.origin + dir * (curr_distance + 0.0001); //(curr_distance + 0.0001);
        vec3 origin = c.origin + dir * curr_distance;

        // After issues have bee fixed, the next boundary calculation could easily be replaced with hardcoded values for each multi

        // ADJUST SUCH THAT BEING AXIS ALIGNED AND IN A NEGATIVE DIRECTION E.G. AT 0.0 WOULD PLACE YOU IN THE NEXT NEGATIVE CHUNK SO 
        // I DONT HAVE TO ADJUST CURR_DISTANCE
        for (int i = 0; i < 3; i++) {
            // if (abs(dir[i]) < 0.0001) {
            //     continue;
            // }
            float nextBoundary = dir[i] > 0.0 ? 
                multiplier * (floor(origin[i] / multiplier) + 1.0) : 
                multiplier * floor(origin[i] / multiplier);
            
            float t = (nextBoundary - origin[i]) / dir[i];
            
            if (t < minT) {
                minT = t;
                hit_axis = uint(i);
            }
            
        }

        // if (adjust == true) {
        //     curr_distance -= 0.01;
        // }
        
        // Get position just before intersection
        curr_distance += minT;
        vec3 pos = c.origin + (dir) * curr_distance;
        
        vec3 temp = floor(pos);

        if (dir[hit_axis] < 0.0) {
            temp[hit_axis] += step[hit_axis];
        }

        
        t_max += (abs(temp - world_pos)) * t_delta;

        // Remove to switch
        //curr_distance += 0.0001;

        // if (dir[hit_axis] < 0.0) {
        //     curr_distance += 0.0001;
        // }
        //curr_distance = t_max[hit_axis] - t_delta[hit_axis];
        
        world_pos = temp;

        //curr_distance += 0.01;

        return;
    }

    if(t_max.x < t_max.y) {
        if(t_max.x < t_max.z) {
            world_pos.x += step.x;
            curr_distance = t_max.x;
            t_max.x += t_delta.x;
            hit_axis = 0;
        } else {
            world_pos.z += step.z;
            curr_distance = t_max.z;
            t_max.z += t_delta.z;
            hit_axis = 2;
        }
    } 
    else {
        if(t_max.y < t_max.z) {
            world_pos.y += step.y;
            curr_distance = t_max.y;
            t_max.y += t_delta.y;
            hit_axis = 1;
        } else {
            world_pos.z += step.z;
            curr_distance = t_max.z;
            t_max.z += t_delta.z;
            hit_axis = 2;
        }
    }
}

vec3 get_colour(uint hit_axis, ivec3 step, vec3 c) {
    vec3 normal;

    // if(hit_axis == 0) normal = vec3(-step.x, 0.0, 0.0);
    // else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
    // else normal = vec3(0.0, 0.0, -step.z);

    return vec3(0.1, 0.1, 0.1);
}

vec3 stone(uint hit_axis, ivec3 step) {
    vec3 normal;

    if(hit_axis == 0) normal = vec3(-step.x, 0.0, 0.0);
    else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
    else normal = vec3(0.0, 0.0, -step.z);

    return vec3(0.7, 0.71, 0.7) * 0.3 + normal * 0.1;
}

vec3 grass(uint hit_axis, ivec3 step) {
        vec3 normal;

    if(hit_axis == 0) normal = vec3(0.0, 0.0, 0.0);
    else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
    else normal = vec3(0.0, 0.0, 0.0);

    // if (hit_axis == 1) {
    //     return vec3(0.7, 0.73, 0.7) * 0.5;
    // }

    return normal + vec3(0.0, 0.4, 0.1) * 0.5;
}

bool get_intersect(ivec2 pixel_coords, vec3 world_pos, inout vec3 t_max, vec3 t_delta, ivec3 step, vec3 dir, inout vec3 hit_colour, inout float curr_distance) {

    uint steps = 0;
    uint hit_axis = 0;
    steps = 0;
    int multiplier = 1;
    int transparent_hits = 0;
    vec3 tansparent_mask = vec3(1.0);

    while((world_pos.x > w_buf.origin.x && world_pos.x < w_buf.origin.x + 64*41) && (world_pos.z > w_buf.origin.z && world_pos.z < w_buf.origin.z + 64*41) && steps < 500) {
        // Go through chunks

        if (world_pos.y > 64*41) {
            break;
        }

        uint voxel_type = get_depth(world_pos, multiplier);

        if (steps > 1000) {
            hit_colour = vec3(0.0, 0.0, 0.0);
            return true;
        }

        // Air
        if (voxel_type == 0) {  
            steps += 1;
            take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance);
            continue;
        }

        if (voxel_type == 1) {
            hit_colour = grass(hit_axis, step) * tansparent_mask;
            vec3 hit_pos = c.origin + dir * curr_distance;
            if (fract(hit_pos.x) > 0.5) {
                hit_colour *= 0.9;
            }

            if (fract(hit_pos.y) < 0.5) {
                hit_colour *= 0.9;
            }

            if (fract(hit_pos.z) < 0.5) {
                hit_colour *= 0.9;
            }


            return true;
        }

        if (voxel_type == 2) {
            hit_colour = stone(hit_axis, step) * tansparent_mask;
            return true;
        }

        if (voxel_type == 3) {
            hit_colour = vec3(0.31, 0.239, 0.9);
            return true; 
        }
    }
    return false;
}

void apply_shadow(vec3 world_pos, vec3 t_max, vec3 t_delta, ivec3 step, vec3 dir, inout vec3 hit_colour, inout float curr_distance) {

    uint steps = 0;
    uint hit_axis = 0;
    int multiplier;

    take_step(step, t_delta, t_max, hit_axis, world_pos, 1, dir, curr_distance);
    //world_pos += step;

    while(steps < 10) {
        // Go through chunks

        uint voxel_type = get_depth(world_pos, multiplier);

        if (voxel_type != 0) {
            hit_colour *= 0.5;
            return;
        }

        steps += 1;
        take_step(step, t_delta, t_max, hit_axis, world_pos, 1, dir, curr_distance);
    }

    return;
}

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    vec3 origin = c.origin;

    vec3 pixel_center = c.pixel00_loc + (c.pixel_delta_u * float(pixel_coords.x)) + (c.pixel_delta_v * float(pixel_coords.y));
    vec3 dir = pixel_center - origin;

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

    vec3 hit_colour;
    float curr_distance = 0;

    bool hit = get_intersect(pixel_coords, world_pos, t_max, t_delta, step, dir, hit_colour, curr_distance);

    if (hit == true) {
        // Get lighting
        vec3 hit_pos = c.origin + dir * curr_distance;
        world_pos = floor(hit_pos);
        dir = normalize((c.sun_loc - hit_pos));

        t_delta = abs(vec3(1.0)/dir);

        if (dir.x < 0.0) {
            step.x = -1;
            t_max.x = ((world_pos.x) - hit_pos.x);
        }
        else {
            step.x = 1;
            t_max.x = ((world_pos.x + 1.0) - hit_pos.x);
        }

        if (dir.y < 0.0) {
            step.y = -1;
            t_max.y = ((world_pos.y) - hit_pos.y);
        }
        else {
            step.y = 1;
            t_max.y = ((world_pos.y + 1.0) - hit_pos.y);
        }

        if (dir.z < 0.0) {
            step.z = -1;
            t_max.z = ((world_pos.z) - hit_pos.z);
        }
        else {
            step.z = 1;
            t_max.z = ((world_pos.z + 1.0) - hit_pos.z);
        }

        t_max /= dir;

        apply_shadow(world_pos, t_max, t_delta, step, dir, hit_colour, curr_distance);

        imageStore(storageImage, pixel_coords, vec4(hit_colour, 1.0));

        return;
    }

    // Apply sun
    // Use sphere ray interception from weekend ray tracing, maybe apply normal/multi ray

    float radius = 500;

    vec3 oc = c.sun_loc - c.origin;
    float a = dot(dir, dir);
    float b = -2.0 * dot(dir, oc);
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;

    if (discriminant >= 0) {
        imageStore(storageImage, pixel_coords, vec4(1.0, 1.0, 0.0, 1.0));
        return;
    }

    float k = (normalize(dir).y + 1.0) * 0.5;
    vec3 colour = vec3(1.0) * (1.0 - k) + vec3(0.5, 0.7, 1.0) * (k);
    vec4 output_colour = vec4(colour, 1.0);
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