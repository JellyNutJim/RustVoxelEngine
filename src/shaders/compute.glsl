#version 460

layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform camera_subbuffer {
    vec3 origin;
    vec3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 world_pos_1;
    vec3 world_pos_2;
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

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    vec3 origin = c.origin;
    vec3 pixel00_loc = c.pixel00_loc;
    vec3 pixel_delta_u = c.pixel_delta_u;
    vec3 pixel_delta_v = c.pixel_delta_v;

    vec3 pixel_center = pixel00_loc + (pixel_delta_u * float(pixel_coords.x)) + (pixel_delta_v * float(pixel_coords.y));
    vec3 dir = pixel_center - origin;

    vec4 output_colour = vec4(1.0);

    //imageStore(storageImage, pixel_coords, vec4(dir,1.0));

    //vec3 world_pos = floor(origin);
    vec3 world_pos = c.world_pos_1;

    vec3 t_delta = abs(vec3(1.0) / dir);

    ivec3 step = ivec3(
        dir.x < 0.0 ? -1 : 1,
        dir.y < 0.0 ? -1 : 1, 
        dir.z < 0.0 ? -1 : 1
    );

    vec3 t_max_1 = vec3(
        (step.x > 0 ?  world_pos.x + 1.0 : world_pos.x) - origin.x,
        (step.y > 0 ?  world_pos.y + 1.0 : world_pos.y) - origin.y,
        (step.z > 0 ?  world_pos.z + 1.0 : world_pos.z) - origin.z
    ) / dir;

    vec3 t_max = t_max_1;

    t_delta *= 2.0;
    step *= 2;  

    vec3 temp = c.world_pos_2;
    vec3 t_max_2 = vec3(
        (step.x > 0 ? temp.x + 2.0 : temp.x) - origin.x,
        (step.y > 0 ? temp.y + 2.0 : temp.y) - origin.y,
        (step.z > 0 ? temp.z + 2.0 : temp.z) - origin.z
    ) / dir;

    t_max = t_max_2;
    uint steps = 0;

    uint hit_axis = 0;
    while(steps < 32) {

        if ((int(world_pos.x) & 8) == 0 && 
            (int(world_pos.y) & 8) == 0 && 
            (int(world_pos.z) & 8) == 0) {

            vec3 normal;

            if(hit_axis == 0) normal = vec3(-step.x, 0.0, 0.0);
            else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
            else normal = vec3(0.0, 0.0, -step.z);

            output_colour = vec4((normal + vec3(1.0)) * 0.5, 1.0);
            imageStore(storageImage, pixel_coords, output_colour);
            return;
        }
        steps += 1;
        take_step(step, t_delta, t_max, hit_axis, world_pos);
    }

    steps = 0;

    // t_delta *= 2.0;
    // step *= 2;  

    //t_delta /= 2.0;
    //step /= 2;  

    //t_max = t_max_1 + (t_max - t_max_2);


    while(steps < 32) {

        if ((int(world_pos.x) & 8) == 0 && 
            (int(world_pos.y) & 8) == 0 && 
            (int(world_pos.z) & 8) == 0) {

            vec3 normal;

            if(hit_axis == 0) normal = vec3(-step.x, 0.0, 0.0);
            else if(hit_axis == 1) normal = vec3(0.0, -step.y, 0.0);
            else normal = vec3(0.0, 0.0, -step.z);

            output_colour = vec4((normal + vec3(1.0)) * 0.5, 1.0);
            imageStore(storageImage, pixel_coords, output_colour);
            return;
        }
        steps += 1;
        take_step(step, t_delta, t_max, hit_axis, world_pos);
    }
    

    float a = (normalize(dir).y + 1.0) * 0.5;
    vec3 colour = vec3(1.0) * (1.0 - a) + vec3(0.5, 0.7, 1.0) * (a);
    output_colour = vec4(clamp(colour.x), clamp(colour.y), clamp(colour.z), 1.0);
    imageStore(storageImage, pixel_coords, output_colour);
}