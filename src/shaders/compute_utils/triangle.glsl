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
