#version 460
// Initial Beam

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform camera_subbuffer {
    vec3 origin;
    vec3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 world_pos_1;
    vec3 sun_loc;
    vec3 time;
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

layout(set = 0, binding = 4) buffer RayDistanceBuffer {
    float ray_distances[7680][4320];
} r_buf;

layout(set = 0, binding = 5) buffer StatBuffer {
    uint march_total;
    uint hit_total;
    uint miss_total;
} stat_buf;

layout(set = 0, binding = 6, rgba8) uniform image2D storageImage;

#include "triangle.glsl"

const int WIDTH = 321;
const bool DETAIL = false;
const bool RENDER_OUT_OF_WORLD_FEATURES = false;

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
    return vec3(0.1, 0.1, 0.1);
}

vec3 stone() {
    return vec3(0.7, 0.71, 0.7) * 0.3;
}

vec3 grass(uint hit_axis, ivec3 step, vec3 hit_pos) {
        vec3 normal;

    if(hit_axis == 0) normal = vec3(0.0, 0.0, 0.0);
    else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
    else normal = vec3(0.0, 0.0, 0.0);

    vec3 hit_colour = normal + vec3(0.0, 0.4, 0.1) * 0.5;

    if (fract(hit_pos.x) > 0.5) {
        hit_colour *= 0.9;
    }

    if (fract(hit_pos.y) < 0.5) {
        hit_colour *= 0.9;
    }

    if (fract(hit_pos.z) < 0.5) {
        hit_colour *= 0.9;
    }

    return hit_colour;
}

vec3 grass2(vec3 hit_pos) {

    vec3 hit_colour = vec3(0.2, 0.94, 0.2);

    if (fract(hit_pos.x) > 0.5) {
        hit_colour *= 0.94;
    }

    if (fract(hit_pos.z) > 0.5) {
        hit_colour *= 0.94;
    }

    if (fract(hit_pos.y) < 0.5) {
        hit_colour *= 0.98;
    }


    return hit_colour;
}

vec3 sand(vec3 hit_pos) {
    vec3 hit_colour = vec3(0.969, 0.953, 0.0);


    if (fract(hit_pos.x) > 0.5) {
        hit_colour *= 0.96;
    }

    if (fract(hit_pos.y) < 0.5) {
        hit_colour *= 0.96;
    }

    if (fract(hit_pos.z) < 0.5) {
        hit_colour *= 0.96;
    }

    return hit_colour;
}

// Triangle intercept
bool intersection_test(vec3 origin, vec3 dir, vec3 v0, vec3 v1, vec3 v2, inout float t) {

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

    vec3 barycentricCoords;
    
    t = dot(v0v2, qvec) * invDet;
    
    // Ray intersection
    if (t > 0.0001) {
        barycentricCoords = vec3(1.0 - u - v, u, v);
        return true;
    }


    return false;
}

bool get_intersect(ivec2 pixel_coords, vec3 world_pos, inout vec3 t_max, vec3 t_delta, ivec3 step, vec3 dir, inout vec3 hit_colour, inout float curr_distance, inout float transparent_distance, inout uint steps) {

    uint hit_axis = 0;
    int multiplier = 1;
    int transparent_hits = 0;
    vec3 tansparent_mask = vec3(0.0);

    float accumculated_curve = 0.0;
    float curve = 0.01;
    float dis = 0.0;

    int curr_chunk = 0;

    while((world_pos.x > w_buf.origin.x && world_pos.x < w_buf.origin.x + 64*WIDTH) && (world_pos.z > w_buf.origin.z && world_pos.z < w_buf.origin.z + 64*WIDTH)) {
        // Go through chunks

        if (world_pos.y > 64*41 || world_pos.y < 0) {
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

        if (transparent_hits > 0) {
            dis = curr_distance - transparent_distance;
            if (dis > 50) { 
                hit_colour = tansparent_mask;
                return true;
            }
        }

        //View octant boundaries
        // if (curr_distance < 500.0 && curr_distance > 2.0) {

        //     vec3 position = c.origin + dir * curr_distance;
        //     vec3 octant_relative = mod(position, multiplier);
            
        //     // bit jank for now but its fine

        //     float threshold = 0.1;
        //     bool near_x = octant_relative.x < threshold || octant_relative.x > multiplier - threshold;
        //     bool near_y = octant_relative.y < threshold || octant_relative.y > multiplier - threshold;
        //     bool near_z = octant_relative.z < threshold || octant_relative.z > multiplier - threshold;
            
        //     int boundary_count = 0;
        //     if (near_x) boundary_count++;
        //     if (near_y) boundary_count++;
        //     if (near_z) boundary_count++;
            
        //     if (boundary_count >= 2) {
        //         float r = 0.0;
        //         float g = 0.0;
        //         float b = 0.0;

        //         if (multiplier == 64) {
        //             r = 1.0;
        //         }
        //         if (multiplier == 32) {
        //             g = 1.0;
        //         }
        //         if (multiplier == 16) {
        //             b = 1.0;
        //         }
        //         hit_colour = vec3(r, g, b);
        //         return true;
        //     }
        // }

        // For now, assume all voxels are surface voxels
        if ( (multiplier == 1 || multiplier == 2 || multiplier == 4) && voxel_type != 858993459) { //world_pos == vec3(54732, 830, 10561) // multiplier == 1
            if (voxel_type == 0) {
                steps += 1;
                take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance);
                continue;
            }

            // if (hit_axis == 1 && step.y == 1) {
            //     hit_colour = stone();
            //     return true;
            // }

            uint voxel = voxel_type;

            uint height_0 = voxel_type & 0xFFu;                 
            uint height_1 = (voxel_type & 0xFF00u) >> 8u;      
            uint height_2 = (voxel_type & 0xFF0000u) >> 16u;   
            uint height_3 = (voxel_type & 0xFF000000u) >> 24u;  


            // Temp fix to deal with missing voxels
            if (height_0 == 0) { 
                height_0 = 1;
            }

            if (height_1 == 0) { 
                height_1 = 1;
            }

            if (height_2 == 0) { 
                height_2 = 1;
            }

            if (height_3 == 0) { 
                height_3 = 1;
            }

            float h0 = (float(height_0 - 1) / 253);
            float h1 = (float(height_1 - 1) / 253);
            float h2 = (float(height_2 - 1) / 253);
            float h3 = (float(height_3 - 1) / 253);

            // Check if side has been hit
            vec3 rel_pos = fract(c.origin + dir * curr_distance);

            // x
            // if (hit_axis == 0) {
            //     if (step.x == 1) {
            //         float m = h2 - h0;
            //         float y_on_line = m * (rel_pos.z) + h0;

            //         if (y_on_line > rel_pos.y) {
            //             hit_colour = stone();
            //             return true;
            //         }
            //     } else {
            //         float m = h3 - h1;
            //         float y_on_line = m * (rel_pos.z) + h1;

            //         if (y_on_line > rel_pos.y) {
            //             hit_colour = stone();
            //             return true;
            //         }
            //     }
            // }

            // // z
            // if (hit_axis == 2) {
            //      if (step.z == 1) {
            //         float m = h1 - h0;
            //         float y_on_line = m * (rel_pos.x) + h0;

            //         if (y_on_line > rel_pos.y) {
            //             hit_colour = stone();
            //             return true;
            //         }
            //     } else {
            //         float m = h3 - h2;
            //         float y_on_line = m * (rel_pos.x) + h2;

            //         if (y_on_line > rel_pos.y) {
            //             hit_colour = stone();
            //             return true;
            //         }
            //     }
            // }


            float scale = float(multiplier);

            // Dodgey temp fix until stepping is sorted out to what it should be
            vec3 scaleed_pos = floor(world_pos / scale) * scale;

            vec3 v0 = scaleed_pos + vec3(0,     h0 * scale, 0);
            vec3 v1 = scaleed_pos + vec3(scale, h1 * scale, 0);
            vec3 v2 = scaleed_pos + vec3(0,     h2 * scale, scale);
            vec3 v3 = scaleed_pos + vec3(scale, h3 * scale, scale);

            float t = 0.0;

            if (intersection_test(c.origin, dir, v0, v1, v2, t) == true) {
                vec3 hit_pos = c.origin + dir * t;

                curr_distance = t;

                if (hit_pos.y > 835) {
                    hit_colour = grass2(hit_pos) * pow((hit_pos.y / 891), 3);
                }
                else if (hit_pos.y < 831) {
                    hit_colour = sand(hit_pos) * pow((hit_pos.y / 891), 3);
                }
                else {
                    float ratio = (hit_pos.y - 831) / 4;

                    hit_colour = sand(hit_pos) * (1 - ratio) + (grass2(hit_pos) * pow((hit_pos.y / 891), 3)) * ratio;  //((sand(hit_pos) * (1 - ratio)) + (grass2(hit_pos) * ratio)) / 2;
                }

                if (transparent_hits > 0) {
                    if (dis > 50) { 
                        hit_colour = tansparent_mask;
                        return true;
                    }

                    float t_per = (dis / 164);

                    hit_colour = (hit_colour * (0.3 - t_per) + tansparent_mask * (0.7 + t_per));
                }

                return true;
            }

            
            if (intersection_test(c.origin, dir, v1, v2, v3, t) == true) {
                vec3 hit_pos = c.origin + dir * t;

                curr_distance = t;

                if (hit_pos.y > 835) {
                    hit_colour = grass2(hit_pos) * pow((hit_pos.y / 891), 3);
                }
                else if (hit_pos.y < 831) {
                    hit_colour = sand(hit_pos) * pow((hit_pos.y / 891), 3);
                }
                else {
                    float ratio = (hit_pos.y - 831) / 4;

                    hit_colour = sand(hit_pos) * (1 - ratio) + (grass2(hit_pos) * pow((hit_pos.y / 891), 3)) * ratio;  //((sand(hit_pos) * (1 - ratio)) + (grass2(hit_pos) * ratio)) / 2;
                }

                if (transparent_hits > 0) {
                    if (dis > 50) { 
                        hit_colour = tansparent_mask;
                        return true;
                    }

                    float t_per = (dis / 164);

                    hit_colour = (hit_colour * (0.3 - t_per) + tansparent_mask * (0.7 + t_per));
                }

                return true;
            }


            voxel_type = 4;

            steps += 1;
            take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance);
            continue;
        }


        // if detail == false
        //voxel_type = voxel_type & 0xFu;

        // Air
        if (voxel_type == 0) {  
            steps += 1;
            take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance);
            continue;
        } else {
            if (voxel_type == 1) {
                hit_colour = grass(hit_axis, step, c.origin + dir * curr_distance);

                if (transparent_hits > 0) {
                    hit_colour = (hit_colour + tansparent_mask) / 2;
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
                hit_colour = stone();

                if (transparent_hits > 0) {
                    hit_colour = (hit_colour + tansparent_mask) / 2;
                }

                return true;
            }

            if (voxel_type == 858993459) {
                if (transparent_hits == 0) {
                    transparent_hits = 1;
                    transparent_distance = curr_distance;
                    tansparent_mask = vec3(0.2, 0.5, 1.0);
                }

                steps += 1;
                take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance);
                continue;
            }

            hit_colour = stone();

            return true;

        }

    }
    return false;
}

void apply_shadow(vec3 world_pos, vec3 ray_origin, vec3 t_max, vec3 t_delta, ivec3 step, vec3 dir, inout vec3 hit_colour, inout float curr_distance) {

    uint steps = 0;
    uint hit_axis = 0;
    int multiplier;

    //world_pos += step;

    while(steps < 30) {
        // Go through chunks

        uint voxel_type = get_depth(world_pos, multiplier);

        if ( multiplier == 1 ) { //world_pos == vec3(54732, 830, 10561) // multiplier == 1

            if (voxel_type == 0) {
                steps += 1;
                take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance);
                continue;
            }

            uint voxel = voxel_type;

            uint n0 = voxel_type & 0xFu;                 // Bottom back left 
            uint n1 = (voxel_type & 0xF0u) >> 4u;        // Bottom front left
            uint n2 = (voxel_type & 0xF00u) >> 8u;       // Bottom back right
            uint n3 = (voxel_type & 0xF000u) >> 12u;     // Bottom front right

            uint n4 = (voxel_type & 0xF0000u) >> 16u;    // Top back left
            uint n5 = (voxel_type & 0xF00000u) >> 20u;   // Top front left
            uint n6 = (voxel_type & 0xF000000u) >> 24u;  // Top back right
            uint n7 = (voxel_type & 0xF0000000u) >> 28u; // Top front right
            
            // For 
            uint height_0 = n4 != 0u ? n4 + 15 : n0;
            uint height_1 = n5 != 0u ? n5 + 15 : n1;
            uint height_2 = n6 != 0u ? n6 + 15 : n2;
            uint height_3 = n7 != 0u ? n7 + 15 : n3;

            // Temp fix to deal with missing voxels
            if (height_0 == 0) { 
                height_0 = 1;
            }

            if (height_1 == 0) { 
                height_1 = 1;
            }

            if (height_2 == 0) { 
                height_2 = 1;
            }

            if (height_3 == 0) { 
                height_3 = 1;
            }

            vec3 v0 = world_pos + vec3(0, (float(height_0 - 1) / 29), 0);
            vec3 v1 = world_pos + vec3(1, (float(height_1 - 1) / 29), 0);
            vec3 v2 = world_pos + vec3(0, (float(height_2 - 1) / 29), 1);
            vec3 v3 = world_pos + vec3(1, (float(height_3 - 1) / 29), 1);

            float t = 0.0;

            if (intersection_test(ray_origin, dir, v0, v1, v2, t) == true) {
                hit_colour *= 0.9;
                return;
            }
            
            if (intersection_test(ray_origin, dir, v1, v2, v3, t) == true) {
                hit_colour *= 0.9;
                return;
            }

            steps += 1;
            take_step(step, t_delta, t_max, hit_axis, world_pos, 1, dir, curr_distance);
            continue;
        }

        steps += 1;
        take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance);

        // if (voxel_type != 0 && voxel_type != 3) {
        //     hit_colour *= 0.5;
        //     return;
        // }
    }

    return;
}

void main() {

    // Effectively every other pixel on the screen
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy) * 2;

    // if (pixel_coords.x == 0 && pixel_coords.y == 0) {
    //     stat_buf.march_total = 0;
    //     stat_buf.hit_total = 0;
    // }

    vec3 origin = c.origin;
    vec3 pixel_center = c.pixel00_loc + (c.pixel_delta_u * float(pixel_coords.x)) + (c.pixel_delta_v * float(pixel_coords.y));
    vec3 dir = normalize(pixel_center - origin);
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
    float curr_distance = 0.0;
    float transparent_distance = 0.0;
    uint steps = 0;

    bool hit = get_intersect(pixel_coords, world_pos, t_max, t_delta, step, dir, hit_colour, curr_distance, transparent_distance, steps);

    atomicAdd(stat_buf.march_total, steps);

    if (hit == true) {
        atomicAdd(stat_buf.hit_total, 1);

        // Get lighting
        vec3 hit_pos = c.origin + dir * (curr_distance - 0.001);
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

        //apply_shadow(world_pos, hit_pos, t_max, t_delta, step, dir, hit_colour, curr_distance);
        
        // If transparent geometry was hit, use that distance instead
        if (transparent_distance != 0) { curr_distance = transparent_distance; }

        
        // Save distance traveled
        r_buf.ray_distances[pixel_coords.x][pixel_coords.y] = curr_distance;

        imageStore(storageImage, pixel_coords, vec4(hit_colour, 1.0));
        return;
    }

    // infinity represents a world miss
    r_buf.ray_distances[pixel_coords.x][pixel_coords.y] = 1.0/0.0;
    atomicAdd(stat_buf.miss_total, 1);

    if ( RENDER_OUT_OF_WORLD_FEATURES == false ) {
        float k = (normalize(dir).y + 1.0) * 0.5;
        vec3 colour = vec3(1.0) * (1.0 - k) + vec3(0.5, 0.7, 1.0) * (k);
        vec4 output_colour = vec4(colour, 1.0);
        imageStore(storageImage, pixel_coords, output_colour);
        return;
    }

    // Check for world intersection
    if (origin.y > 816 && dir.y < 0) {
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
            float y_hit_dis = (816 - origin.y) / dir.y; 
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
    
    // Calculate sky colour 
    float k = (normalize(dir).y + 1.0) * 0.5;
    vec3 colour = vec3(1.0) * (1.0 - k) + vec3(0.5, 0.7, 1.0) * (k);
    vec4 output_colour = vec4(colour, 1.0);
    imageStore(storageImage, pixel_coords, output_colour);
    
}