struct Triangle {
    vec3 v0;
    vec3 v1;
    vec3 v2;
};

Triangle createTriangle(vec3 vertex0, vec3 vertex1, vec3 vertex2) {
    Triangle tri;
    tri.v0 = vertex0;
    tri.v1 = vertex1;
    tri.v2 = vertex2;
    return tri;
}

// Triangle intercept https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
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

void take_step(ivec3 step, vec3 t_delta, inout vec3 t_max, inout uint hit_axis, inout vec3 world_pos, int multiplier, vec3 dir, inout float curr_distance, vec3 true_origin, float multiplier_div) {
    if (multiplier > 4) {
        float minT = 1e10;

        vec3 origin = true_origin + dir * curr_distance;

        const float adjust = 0.001 * float(multiplier); // 0.0001
        const float neg_ajust = float(multiplier) - adjust;
        uint edge_count = 0;
        bool pre_near_edge;
        bool is_near_edge;

        for (uint i = 0; i < 3; i++) {
            if (dir[i] == 0) {
                continue;
            }

            pre_near_edge = false;

            // Way to keep track of current chunk so this calculation can be avoided?
            float current_chunk = floor(origin[i] * multiplier_div) * multiplier;

            if (step[i] == 1) {
                current_chunk += multiplier;
            }
            else {

            }
            
            float t = abs((origin[i] - current_chunk) * t_delta[i]); 
            
            if ( t < minT) {
                minT = t;
                hit_axis = i;
                //is_near_edge = pre_near_edge;
            }
        }

        // if (edge_count >= 2 && hit_axis == 1) {
        //     world_pos[hit_axis] += step[hit_axis];
        //     t_max[hit_axis] += t_delta[hit_axis];
        //     curr_distance = t_max[hit_axis];
        //     return;
        // } 

        if (step[hit_axis] == -1.0) {
            minT += adjust;
        }

        curr_distance += minT;

        vec3 temp = floor(c.origin + (dir) * curr_distance);


        // if (is_near_edge == false && edge_count < 2 && step[hit_axis] == -1.0) {
        //     temp[hit_axis] += step[hit_axis];
        // }

        if (is_near_edge == false )
        

        t_max += (abs(temp - world_pos)) * t_delta;
        
        world_pos = temp;
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

bool point_in_octant(vec3 point, vec3 corner, float scale ) {
    return point.x >= corner.x && point.x < corner.x + scale &&
           point.y >= corner.y && point.y < corner.y + scale &&
           point.z >= corner.z && point.z < corner.z + scale;
}

bool isInWorld(vec3 pos, vec3 world_min, vec3 world_max) {
    if (pos.x < world_min.x || pos.x > world_max.x)
        return false;
        
    if (pos.z < world_min.z || pos.z > world_max.z)
        return false;
        
    if (pos.y < world_min.y || pos.y > world_max.y)
        return false;

    return true;
}

vec3 get_colour(uint hit_axis, ivec3 step, vec3 c) {
    vec3 normal;
    return vec3(0.1, 0.1, 0.1);
}

vec3 stone() {
    return vec3(0.7, 0.71, 0.7) * 0.3;
}

vec3 old_grass(uint hit_axis, ivec3 step, vec3 hit_pos) {
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

vec3 grass(vec3 hit_pos) {

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

    // if (transparent_hits == 1) {
    //     return vec3(0,0,0);
    // }
    // if (multiplier == 1) {
    //     return vec3(1,0,0);
    // }
    // if (multiplier == 2) {
    //     return vec3(0,1,0);
    // }
    // if (multiplier == 4) {
    //     return vec3(0,0,1);
    // }
    // if (multiplier == 8) {
    //     return vec3(0,1,1);
    // }


vec3 get_surface_colour(vec3 hit_pos, int transparent_hits, vec3 transparent_mask, float dis, int multiplier) {

    vec3 hit_colour = vec3(0,0,0);

    if (hit_pos.y > 10308) {
        hit_colour = grass(hit_pos) * pow(((hit_pos.y - 9000) / 1389), 5);
    }
    else if (hit_pos.y < 10304) {
        hit_colour = sand(hit_pos) * pow(((hit_pos.y - 9000) / 1389), 5);
    }
    else {
        float ratio = (hit_pos.y - 10304) / 4;

        hit_colour = sand(hit_pos) * (1 - ratio) + (grass(hit_pos) * pow(((hit_pos.y - 9000) / 1389), 5) ) * ratio;  
    }

    if (transparent_hits > 0) {
        if (dis > 50) { 
            return transparent_mask;
        }

        float t_per = (dis / 164);

        hit_colour = (hit_colour * (0.3 - t_per) + transparent_mask * (0.7 + t_per));
    }

    return hit_colour;
}

            // Check if side has been hit
            //vec3 rel_pos = fract(c.origin + dir * curr_distance);

            // // x
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



            // void apply_shadow(vec3 world_pos, vec3 ray_origin, vec3 t_max, vec3 t_delta, ivec3 step, vec3 dir, inout vec3 hit_colour, inout float curr_distance) {

//     uint steps = 0;
//     uint hit_axis = 0;
//     int multiplier;

//     //world_pos += step;

//     while(steps < 30) {
//         // Go through chunks

//         uint voxel_type = get_depth(world_pos, multiplier);

//         if ( multiplier == 1 ) { //world_pos == vec3(54732, 830, 10561) // multiplier == 1

//             if (voxel_type == 0) {
//                 steps += 1;
//                 take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance);
//                 continue;
//             }

//             uint voxel = voxel_type;

//             uint n0 = voxel_type & 0xFu;                 // Bottom back left 
//             uint n1 = (voxel_type & 0xF0u) >> 4u;        // Bottom front left
//             uint n2 = (voxel_type & 0xF00u) >> 8u;       // Bottom back right
//             uint n3 = (voxel_type & 0xF000u) >> 12u;     // Bottom front right

//             uint n4 = (voxel_type & 0xF0000u) >> 16u;    // Top back left
//             uint n5 = (voxel_type & 0xF00000u) >> 20u;   // Top front left
//             uint n6 = (voxel_type & 0xF000000u) >> 24u;  // Top back right
//             uint n7 = (voxel_type & 0xF0000000u) >> 28u; // Top front right
            
//             // For 
//             uint height_0 = n4 != 0u ? n4 + 15 : n0;
//             uint height_1 = n5 != 0u ? n5 + 15 : n1;
//             uint height_2 = n6 != 0u ? n6 + 15 : n2;
//             uint height_3 = n7 != 0u ? n7 + 15 : n3;

//             // Temp fix to deal with missing voxels
//             if (height_0 == 0) { 
//                 height_0 = 1;
//             }

//             if (height_1 == 0) { 
//                 height_1 = 1;
//             }

//             if (height_2 == 0) { 
//                 height_2 = 1;
//             }

//             if (height_3 == 0) { 
//                 height_3 = 1;
//             }

//             vec3 v0 = world_pos + vec3(0, (float(height_0 - 1) / 29), 0);
//             vec3 v1 = world_pos + vec3(1, (float(height_1 - 1) / 29), 0);
//             vec3 v2 = world_pos + vec3(0, (float(height_2 - 1) / 29), 1);
//             vec3 v3 = world_pos + vec3(1, (float(height_3 - 1) / 29), 1);

//             float t = 0.0;

//             if (intersection_test(ray_origin, dir, v0, v1, v2, t) == true) {
//                 hit_colour *= 0.9;
//                 return;
//             }
            
//             if (intersection_test(ray_origin, dir, v1, v2, v3, t) == true) {
//                 hit_colour *= 0.9;
//                 return;
//             }

//             steps += 1;
//             take_step(step, t_delta, t_max, hit_axis, world_pos, 1, dir, curr_distance);
//             continue;
//         }

//         steps += 1;
//         take_step(step, t_delta, t_max, hit_axis, world_pos, multiplier, dir, curr_distance);

//         // if (voxel_type != 0 && voxel_type != 3) {
//         //     hit_colour *= 0.5;
//         //     return;
//         // }
//     }

//     return;
// }


        // View octant boundaries
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
