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