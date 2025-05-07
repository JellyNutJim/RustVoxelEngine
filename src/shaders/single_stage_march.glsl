#version 460
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

layout(set = 0, binding = 6) readonly buffer OctantMapBuffer {
    uint octant_map[1572864];
} o_buf;

layout(set = 0, binding = 7, rgba8) uniform image2D storageImage;

layout(set = 0, binding = 8) uniform sampler2D grassTexture;

#include "march.glsl"

const int WIDTH = 321;
const int WIDTH_SQUARED = WIDTH * WIDTH;
const int VOXEL_WIDTH = WIDTH * 64;
const bool DETAIL = false;
const bool RENDER_OUT_OF_WORLD_FEATURES = false;

// Multiplier levels
const float mul_64 = 1.0/64.0;
const float mul_32 = 1.0/32.0;
const float mul_16 = 1.0/16.0;
const float mul_8 = 1.0/8.0;
const float mul_4 = 1.0/4.0;
const float mul_2 = 1.0/2.0;

// Geometry Types
const uint four_height_surface_code = 4 << 24;
const uint four_height_water_code = 5 << 24;

const uint steep_four_height_surface_code = 2 << 24;
const uint steep_four_height_water_code = 3 << 24;

const uint sphere = 2 << 24;

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

uvec3 get_geom(uint type, uint index) {
    // Quick return for air
    if (type == 1) {
        return uvec3(0, 0, 0);
    }

    // Get Geometry Type
    uint geom_type = type & 0xFF000000u;

    // Voxel
    if (geom_type == 0) {
        return uvec3(0, type, 0);
    } 
    // Four Height Surface
    else if (geom_type == four_height_surface_code) {
        return uvec3(4, v_buf.voxels[index + 1], 0);
    }
    else if (geom_type == four_height_water_code) {
        return uvec3(5, v_buf.voxels[index + 1], 0);
    } 
    else if ((geom_type & steep_four_height_surface_code) == steep_four_height_surface_code) {

        if ((geom_type & steep_four_height_water_code) == steep_four_height_water_code) {
            return uvec3(3, v_buf.voxels[index], v_buf.voxels[index + 1]);
        }

        return uvec3(2, v_buf.voxels[index], v_buf.voxels[index + 1]);
    }

    return uvec3(0,0,0);
}

uvec3 get_depth(vec3 pos, inout int multiplier, inout float multiplier_div) {
    vec3 chunk_loc = floor(pos / 64);
    uvec3 rel_chunk_loc = uvec3(chunk_loc);

    uint index = rel_chunk_loc.x + (rel_chunk_loc.y * WIDTH) + (rel_chunk_loc.z * WIDTH_SQUARED);

    index = w_buf.chunks[index];
    uint current = v_buf.voxels[index];

    // Initial check for air
    if (current != 0) {
        multiplier = 64;
        multiplier_div = mul_64;
        return get_geom(v_buf.voxels[index], index);
    }

    uvec3 chunk_loc_u = uvec3(chunk_loc) << 6;

    // Get octant map
    uvec3 lpos = uvec3(pos - chunk_loc_u);
    uint octant_map = o_buf.octant_map[lpos.x + (lpos.y << 6) + (lpos.z << 12)];
    
    uint octant = octant_map & 7;
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] != 0) {
       multiplier = 32;
       multiplier_div = mul_32;
       return get_geom(v_buf.voxels[index], index);
    }

    octant = (octant_map >> 3) & 7;
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] != 0) {
        multiplier = 16;
        multiplier_div = mul_16;
        return get_geom(v_buf.voxels[index], index);
    }

    octant = (octant_map >> 6) & 7;;
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] != 0) {
        multiplier = 8;
        multiplier_div = mul_8;
        return get_geom(v_buf.voxels[index], index);
    }

    octant = (octant_map >> 9) & 7;
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] != 0) {
        multiplier = 4;
        multiplier_div = mul_4;
        return get_geom(v_buf.voxels[index], index);
    }

    octant = (octant_map >> 12) & 7;
    index = index + v_buf.voxels[index + octant + 1];

    if (v_buf.voxels[index] != 0) {
        multiplier = 2;
        multiplier_div = mul_2;
        return get_geom(v_buf.voxels[index], index);
    }

    octant = (octant_map >> 15) & 7;
    index = index + 2 + ((v_buf.voxels[index + 1] >> (octant * 4u)) & 0xFu) ;

    multiplier = 1;
    multiplier_div = 1;
    return get_geom(v_buf.voxels[index], index);
}

vec3 grass2(vec3 hit_pos) {
    vec2 uv = hit_pos.xz * 0.01;
    
    return texture(grassTexture, uv).rgb * 10.0;
}

bool get_intersect(ivec2 pixel_coords, vec3 world_pos, inout vec3 t_max, vec3 t_delta, ivec3 step, vec3 dir, inout vec3 hit_colour, inout float curr_distance, inout uint steps) {

    uint hit_axis = 0;
    int multiplier = 1;
    float multiplier_div = 1;

    int transparent_hits = 0;
    float transparent_distance = 0.0;
    vec3 tansparent_mask = vec3(0.0);
    float dis = 0.0;

    vec3 true_origin = c.origin;
    vec3 world_origin = vec3(0,0,0);

    float y_max = 41 << 6;

    vec3 world_max = world_origin + VOXEL_WIDTH;
    world_max.y = y_max;


    while(isInWorld(world_pos, world_origin, world_max)) {
        // Go through chunks

        if (steps > 2000) {
            hit_colour = vec3(0, 1, 0);
            return true;
        }

        if (transparent_hits > 0) {
            dis = curr_distance - transparent_distance;
            if (dis > 50) { 
                hit_colour = tansparent_mask;
                return true;
            }
        }

        uvec3 voxel_type = get_depth(world_pos, multiplier, multiplier_div);

        // Voxels
        if ( voxel_type.x == 0 ) {
            if (voxel_type.y == 0 ) {
                steps += 1;
                take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance, true_origin, multiplier_div);
                continue;
            } else if (voxel_type.y == 3) {

                if (transparent_hits == 0) {
                    transparent_hits = 1;
                    transparent_distance = curr_distance;
                    tansparent_mask = vec3(0.2, 0.5, 1.0);
                }
                
                steps += 1;
                take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance, true_origin, multiplier_div);
                continue;

            } else if (voxel_type.y == 2) {
                vec3 hit_pos = c.origin + dir * curr_distance;
                hit_colour = grass(hit_pos);
                return true;
            } 
            else {
                hit_colour = stone();
                return true;
            }
        }

        // Water on 4 height surface
        if (voxel_type.x == 5) {
            voxel_type.x = 4;
            if (hit_axis == 1 && step.y == -1) {
                if (transparent_hits == 0) {
                    transparent_hits = 1;
                    transparent_distance = curr_distance;
                    tansparent_mask = vec3(0.2, 0.5, 1.0);
                }
            } 
        }
\
        // Four Height Surface
        if ( voxel_type.x == 4 ) {
            uint heights = voxel_type.y;

            uint height_0 = heights & 0xFFu;                 // Bottom back left
            uint height_1 = (heights & 0xFF00u) >> 8u;       // Bottom front left
            uint height_2 = (heights & 0xFF0000u) >> 16u;    // Bottom back right
            uint height_3 = (heights & 0xFF000000u) >> 24u;  // Bottom front right

            float h0 = (float(height_0) / 255);
            float h1 = (float(height_1) / 255);
            float h2 = (float(height_2) / 255);
            float h3 = (float(height_3) / 255);

            float scale = float(multiplier);

            // Dodgey temp fix until stepping is sorted out to what it should be
            vec3 octant = floor(world_pos / scale) * scale;

            // Traingles verticies
            vec3 v0 = octant + vec3(0,     h0 * scale, 0);
            vec3 v1 = octant + vec3(scale, h1 * scale, 0);
            vec3 v2 = octant + vec3(0,     h2 * scale, scale);
            vec3 v3 = octant + vec3(scale, h3 * scale, scale);

            float t = 0.0;
            if (intersection_test(c.origin, dir, v0, v1, v2, t) == true || intersection_test(c.origin, dir, v1, v2, v3, t) == true) {
                vec3 hit_pos = c.origin + dir * t;

                curr_distance = t;

                hit_colour = get_surface_colour(hit_pos, transparent_hits, tansparent_mask, dis);

                return true;
            }
        }


        if (voxel_type.x == 3) {
            voxel_type.x = 2;

            if (hit_axis == 1 && step.y == -1) {
                if (transparent_hits == 0) {
                    transparent_hits = 1;
                    transparent_distance = curr_distance;
                    tansparent_mask = vec3(0.2, 0.5, 1.0);
                }
            } 
        }

        if (voxel_type.x == 2) {

            uint n1 = voxel_type.y;
            uint n2 = voxel_type.z;

            uint rel_pos = (n1 >> 26);

            float height_0 = float ( (n1 >> 10) & 0x3FFF );    
            float height_1 = float ( (n1 & 0x3FF) << 4 | (n2 >> 28) );
            float height_2 = float ( (n2 >> 14) & 0x3FFF );
            float height_3 = float ( n2 & 0x3FFF );

            // get relative height within voxel
            float h0 = (height_0 / 255.0) - rel_pos;
            float h1 = (height_1 / 255.0) - rel_pos;
            float h2 = (height_2 / 255.0) - rel_pos;
            float h3 = (height_3 / 255.0) - rel_pos;

            float scale = float(multiplier);
            vec3 octant = floor(world_pos / scale) * scale;

            // Traingles verticies
            vec3 v0 = octant + vec3(0,     h0 * scale, 0);
            vec3 v1 = octant + vec3(scale, h1 * scale, 0);
            vec3 v2 = octant + vec3(0,     h2 * scale, scale);
            vec3 v3 = octant + vec3(scale, h3 * scale, scale);

            float t = 0.0;
            if (intersection_test(c.origin, dir, v0, v1, v2, t) == true || intersection_test(c.origin, dir, v1, v2, v3, t) == true) {
                vec3 hit_pos = c.origin + dir * t;

                if (point_in_octant(hit_pos, world_pos, scale) == false) {
                    if (multiplier != 8) {
                        steps += 1;
                        take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance, true_origin, multiplier_div);
                        continue;
                    } 
                } 

                curr_distance = t;

                hit_colour = get_surface_colour(hit_pos, transparent_hits, tansparent_mask, dis);

                return true;
            }
        }

        steps += 1;
        take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance, true_origin, multiplier_div);
        continue;
    }
    return false;
}

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_dimensions = imageSize(storageImage);

    // Setup ray direction
    vec3 origin = c.origin;
    vec3 pixel_center = c.pixel00_loc + (c.pixel_delta_u * float(pixel_coords.x)) + (c.pixel_delta_v * float(pixel_coords.y));
    vec3 dir = normalize(pixel_center - origin);

    bool hit = false;

    vec3 world_pos = c.world_pos_1;

    vec3 t_delta;
    ivec3 step;
    vec3 t_max;
    float curr_distance;

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
    curr_distance = 0.0;
    uint steps = 0;

    hit = get_intersect(pixel_coords, world_pos, t_max, t_delta, step, dir, hit_colour, curr_distance, steps);

    //atomicAdd(stat_buf.march_total, steps);

    if (hit == true) {

        imageStore(storageImage, pixel_coords, vec4(hit_colour, 1.0));
        return;
    }
    
    //atomicAdd(stat_buf.miss_total, 1);

    if ( RENDER_OUT_OF_WORLD_FEATURES == false ) {
        float k = (normalize(dir).y + 1.0) * 0.5;
        vec3 colour = vec3(1.0) * (1.0 - k) + vec3(0.5, 0.7, 1.0) * (k);
        vec4 output_colour = vec4(colour, 1.0);
        imageStore(storageImage, pixel_coords, output_colour);
        return;
    }

    // Check for world intersection
    if (origin.y > 816 && dir.y < 0) {

        vec3 planet_loc = vec3(origin.x, -900000.0, origin.z);
        float radius = 900768.0;

        vec3 oc = planet_loc - c.origin;
        float a = dot(dir, dir);
        float b = -2.0 * dot(dir, oc);
        float c = dot(oc, oc) - radius*radius;
        float discriminant = b*b - 4*a*c;

        if (discriminant >= 0) {
            vec3 rel_planet_loc = vec3(origin.x + w_buf.origin.x, origin.y, origin.z + w_buf.origin.z);

            float y_hit_dis = (816 - origin.y) / dir.y; 
            vec3 hp = rel_planet_loc + dir * y_hit_dis;
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


// testing
// float scale = float(multiplier);
        // float minT = 1e10;

        // vec3 origin = c.origin + dir * curr_distance;
        // vec3 rel_origin = origin - (floor(origin / scale) * scale); // switch for mod
        // vec3 current_chunk = floor(origin / scale) * scale;

        // float t;
        
        // hit_axis = 0;

        // for (int i = 0; i < 3; i++) {
        //     if (dir[i] == 0) {
        //         continue;
        //     }

        //     if (step[i] == 1) {
        //         t = (scale - rel_origin[i]) * t_delta[i];
        //     }
        //     else {
        //         if (rel_origin[i] < 1e-5) {
        //             t = (scale) * t_delta[i];
        //         } else{
        //             t = (rel_origin[i]) * t_delta[i];
        //         }
        //     } 
            
        //     if (t < minT) {
        //         minT = t;
        //         hit_axis = uint(i);
        //     }
        // }

        // current_chunk[hit_axis] += step[hit_axis] * scale;

        // vec3 hit_pos = vec3(0,0,0);

        // t = abs((current_chunk[hit_axis] - c.origin[hit_axis]) / dir[hit_axis]);

        // uint nextIndex = (hit_axis + 1) % 3;
        // uint lastIndex = (hit_axis + 2) % 3;
        
        // hit_pos[hit_axis]  = current_chunk[hit_axis];
        // hit_pos[nextIndex] = c.origin[nextIndex] + t * dir[nextIndex];
        // hit_pos[lastIndex] = c.origin[lastIndex] + t * dir[lastIndex];

        // curr_distance = t;

        // vec3 new_pos = floor(hit_pos);

        // if (step[hit_axis] == -1) {
        //     new_pos[hit_axis] += step[hit_axis];
        // }

        // t_max += (abs(new_pos - world_pos)) * t_delta;

        // ///new_pos[hit_axis] += step[hit_axis];
        // world_pos = new_pos;

    //       if (multiplier > 4) {
    //     float scale = float(multiplier);

    //     vec3 origin = c.origin + dir * curr_distance;
    //     vec3 current_chunk = floor(origin / scale) * scale;
    //     float t;
        
    //     vec3 f_max = vec3(0,0,0);

    //     if (step[0] == -1) {
    //         f_max.x = ((current_chunk.x) - origin.x);
    //     }
    //     else {
    //         f_max.x = ((current_chunk.x + scale) - origin.x);
    //     }

    //     if (step[1] == -1) {
    //         f_max.y = ((current_chunk.y) - origin.y);
    //     }
    //     else {
    //         f_max.y = ((current_chunk.y + scale) - origin.y);
    //     }

    //     if (step[2] == -1) {
    //         f_max.z = ((current_chunk.z) - origin.z);
    //     }
    //     else {
    //         f_max.z = ((current_chunk.z + scale) - origin.z);
    //     }

    //     f_max /= dir;

    //     float f_min;

    //     if (f_max.x < f_max.y) {
    //         if (f_max.x < f_max.z) {
    //             hit_axis = 0;
    //             f_min = f_max.x;
    //         } else {
    //             hit_axis = 2;
    //             f_min = f_max.z;
    //         }
    //     } 
    //     else  {
    //         if (f_max.y < f_max.z) {
    //             hit_axis = 1;
    //             f_min = f_max.y;
    //         } else {
    //             hit_axis = 2;
    //             f_min = f_max.z;
    //         }
    //     }


    //     curr_distance += f_min;
        
    //     vec3 hit_pos = c.origin + (dir) * curr_distance;
        

    //     vec3 new_pos = floor(hit_pos);

    //     if (step[hit_axis] == -1) {
    //         new_pos[hit_axis] += step[hit_axis];
    //     }

    //     t_max += (abs(new_pos - world_pos)) * t_delta;


        
    //     world_pos = new_pos;

    //     return;
    // }