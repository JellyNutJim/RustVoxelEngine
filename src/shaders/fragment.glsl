#version 450
            
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform camera_subbuffer {
    vec3 q;
} camera;


void main() {
    // Use UV coordinates for your pixel-based rendering
    // You can add your own logic here based on pixel position
    f_color = vec4(uv.x, uv.y, 0.0, 1.0);
}