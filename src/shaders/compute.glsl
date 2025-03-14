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

layout(set = 0, binding = 3) readonly buffer NoiseBuffer {
    float perm[512];
    float grad[512];
} n_buf;

layout(set = 0, binding = 4, rgba8) uniform image2D storageImage;

const int WIDTH = 321;
const bool DETAIL = false;

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

        vec3 origin = c.origin + dir * curr_distance;

        const float adjust = 0.001 * multiplier / 64.0;

        for (int i = 0; i < 3; i++) {
            if (dir[i] == 0) {
                continue;
            }

            float current_chunk = floor(origin[i] / multiplier);

            float target_chunk = dir[i] > 0.0 ? 
                current_chunk + 1.0 : 
                current_chunk - (mod(origin[i], float(multiplier)) <= adjust ? 1.0 : 0.0);

            float boundary = target_chunk * multiplier;
            
            float t = (boundary - origin[i]) / dir[i];
            
            if ( t > 1e-6 && t < minT) {
                minT = t;
                hit_axis = uint(i);
            }
            
        }

        curr_distance += minT;

        if (dir.y <= 0.0) {
            // Apply a tiny backstep only for non-upward rays or non-y-axis hits
            curr_distance -= 1e-4;
        }
        
        vec3 pos = c.origin + (dir) * curr_distance;
        
        vec3 temp = floor(pos);

        if (dir[hit_axis] < 0.0) {
            temp[hit_axis] += step[hit_axis];
        }

        
        t_max += (abs(temp - world_pos)) * t_delta;
        
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
    vec3 tansparent_mask = vec3(0.0);

    float accumculated_curve = 0.0;
    float curve = 0.01;

    int curr_chunk = 0;

    while((world_pos.x > w_buf.origin.x && world_pos.x < w_buf.origin.x + 64*WIDTH) && (world_pos.z > w_buf.origin.z && world_pos.z < w_buf.origin.z + 64*WIDTH)) {
        // Go through chunks

        if (world_pos.y > 64*41) {
            break;
        }

        uint voxel_type = get_depth(world_pos, multiplier);

        // if (floor(curr_distance / 64) * 64 == 960) {
        //     world_pos.y += 1;
        // }

        if (steps > 2000) {
            hit_colour = vec3(0.0, 0.0, 0.0);
            return true;
            break;
        }

        if (DETAIL == false) {
            voxel_type = voxel_type & 0xFu;
        }


        // Air
        if (voxel_type == 0) {  
            steps += 1;
            take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance);
            continue;
        } else {
            if (voxel_type == 1) {
                hit_colour = grass(hit_axis, step);

                if (transparent_hits > 0) {
                    hit_colour = (hit_colour + tansparent_mask) / 2;
                }


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

            if (voxel_type == 4) {
                hit_colour = vec3(0.969, 0.953, 0.0);

                if (transparent_hits > 0) {
                    hit_colour = (hit_colour * 0.1 + tansparent_mask * 0.9);
                    return true;
                }


                vec3 hit_pos = c.origin + dir * curr_distance;
                if (fract(hit_pos.x) > 0.5) {
                    hit_colour *= 0.96;
                }

                if (fract(hit_pos.y) < 0.5) {
                    hit_colour *= 0.96;
                }

                if (fract(hit_pos.z) < 0.5) {
                    hit_colour *= 0.96;
                }

                return true;
            }

            if (voxel_type == 2) {
                hit_colour = stone(hit_axis, step);

                if (transparent_hits > 0) {
                    hit_colour = (hit_colour + tansparent_mask) / 2;
                }

                return true;
            }

            if (voxel_type == 3) {
                transparent_hits = 1;
                tansparent_mask = vec3(0.2, 0.5, 0.91);
                steps += 1;

                vec3 hit_pos = c.origin + dir * curr_distance;
                if (fract(hit_pos.x) > 0.5) {
                    tansparent_mask *= 0.96;
                }

                if (fract(hit_pos.y) < 0.5) {
                    tansparent_mask *= 0.96;
                }

                if (fract(hit_pos.z) < 0.5) {
                    tansparent_mask *= 0.96;
                }


                take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance);
                continue;
            }

            hit_colour = stone(hit_axis, step);

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

    while(steps < 30) {
        // Go through chunks

        uint voxel_type = get_depth(world_pos, multiplier);

        if (voxel_type != 0 && voxel_type != 3) {
            hit_colour *= 0.5;
            return;
        }

        steps += 1;
        take_step(step, t_delta, t_max, hit_axis, world_pos, 1, dir, curr_distance);
    }

    return;
}

float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

float fade(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float grad(uint hash, float x, float z) {
    uint index = (hash & 255) * 2;
    return n_buf.grad[index] * x + n_buf.grad[index + 1] * z;
}

float get_perlin_noise(float x, float z) { 
    int x_floor = int(floor(x));
    int z_floor = int(floor(z));

    x = x - x_floor;
    z = z - z_floor;

    x_floor = x_floor & 255;
    z_floor = z_floor & 255;

    uint a  = uint(n_buf.perm[x_floor] + z_floor); 
    uint aa = uint(n_buf.perm[a]);
    uint ab = uint(n_buf.perm[a + 1]);
    
    uint b  = uint(n_buf.perm[x_floor + 1] + z_floor);
    uint ba = uint(n_buf.perm[b]);
    uint bb = uint(n_buf.perm[b + 1]); 

    float u = fade(x);
    float v = fade(z);

    float x1 = grad(aa, x, z);
    float x2 = grad(ba, x - 1, z);
    float z1 = lerp(x1, x2, u);

    x1 = grad(ab, x, z - 1.0);
    x2 = grad(bb, x - 1, z - 1);
    float z2 = lerp(x1, x2, u);


    return lerp(z1, z2, v);
} 

// Triangle intercept
bool intersection_test(vec3 origin, vec3 dir, vec3 v0, vec3 v1, vec3 v2) {

    vec3 v0v1 = v1 - v0;
    vec3 v0v2 = v2 - v0;
    vec3 pvec = cross(dir, v0v2);
    float det = dot(v0v1, pvec);

    if (abs(det) < 0.0001) { return false; }

    float invDet = 1 / det;

    vec3 tvec = origin - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) { return false; }

    vec3 qvec = cross(tvec, v0v1);
    float v = dot(dir, qvec) * invDet;
    if (v < 0 || u + v > 1) { return false; }

    float t;
    vec3 barycentricCoords;
    
    t = dot(v0v2, qvec) * invDet;
    
    // Ray intersection
    if (t > 0.0001) {
        barycentricCoords = vec3(1.0 - u - v, u, v);
        return true;
    }


    return false;
}


void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    vec3 origin = c.origin;

    vec3 pixel_center = c.pixel00_loc + (c.pixel_delta_u * float(pixel_coords.x)) + (c.pixel_delta_v * float(pixel_coords.y));
    vec3 dir = pixel_center - origin;


    vec3 v0 = vec3(17738.0, 830.0, 10580.0);
    vec3 v1 = vec3(17729.0, 831.0, 10621.0);
    vec3 v2 = vec3(17720.0, 820.0, 10550.0);

    if (intersection_test(origin, dir, v0, v1, v2) == true) {
        imageStore(storageImage, pixel_coords, vec4(1.0, 0.984, 0.0, 1.0));
        return;
    }

    vec3 world_pos = c.world_pos_1;

    vec3 t_delta;
    ivec3 step;
    vec3 t_max;

    const float limit = 1e-10;
    
    t_delta.x = (abs(dir.x) < limit) ? 1e30 : abs(1.0 / dir.x);
    t_delta.y = (abs(dir.y) < limit) ? 1e30 : abs(1.0 / dir.y);
    t_delta.z = (abs(dir.z) < limit) ? 1e30 : abs(1.0 / dir.z);

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

    // Check for world intersection
    if (origin.y > 768 && dir.y < 0) {
        // float y_hit_dis = (768 - origin.y) / dir.y;
        // vec3 hp = origin + dir * y_hit_dis;

        vec3 planet_loc = vec3(origin.x, -900000.0, origin.y);
        float radius = 900768.0;

        vec3 oc = planet_loc - c.origin;
        float a = dot(dir, dir);
        float b = -2.0 * dot(dir, oc);
        float c = dot(oc, oc) - radius*radius;
        float discriminant = b*b - 4*a*c;

        if (discriminant >= 0) {
            float y_hit_dis = (768 - origin.y) / dir.y; 
            vec3 hp = origin + dir * y_hit_dis;
            float scale = 0.00003;
            float nx = hp.x * scale;
            float nz = hp.z * scale;

            float y = get_perlin_noise(nx, nz) * 16;
            float stepped_y = floor(y) * 16;

            y *= 16;

            if (y > 16.0  && y < 16.3) {
                imageStore(storageImage, pixel_coords, vec4(1.0, 0.984, 0.0, 1.0));
                return;
            }

            if (stepped_y == 16) {
                imageStore(storageImage, pixel_coords, vec4(0.0, 1.0, 0.1, 1.0));
                return;
            }

            if (y > 32.0  && y < 33.5) {
                imageStore(storageImage, pixel_coords, vec4(0.0, 0.6, 0.0, 1.0));
                return;
            }

            if (stepped_y > 16.0) {
                imageStore(storageImage, pixel_coords, vec4(0.0, 1.0, 0.1, 1.0));
                return;
            }

            imageStore(storageImage, pixel_coords, vec4(0.31, 0.239, 0.9, 1.0));
            return;   
        }
    }


    float radius = 500;

    vec3 oc = c.sun_loc - c.origin;
    float a = dot(dir, dir);
    float b = -2.0 * dot(dir, oc);
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;

    if (discriminant > 0) {
        imageStore(storageImage, pixel_coords, vec4(1.0, 1.0, 0.0, 1.0));
        return;
    }
    
    // Calculate sky color (your existing sky gradient)
    float k = (normalize(dir).y + 1.0) * 0.5;
    vec3 colour = vec3(1.0) * (1.0 - k) + vec3(0.5, 0.7, 1.0) * (k);
    vec4 output_colour = vec4(colour, 1.0);
    imageStore(storageImage, pixel_coords, output_colour);
    
}


// In voxel search and render

// Sub voxel render

// Sub triangle render


