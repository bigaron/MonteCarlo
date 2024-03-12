#version 460 core
out vec4 ourColor;

struct VertexData{
    vec4 pos;
    vec4 col;
};

layout(std430, binding = 0) readonly buffer vertexAttr{
    VertexData vertices[];
};

layout(std140, binding = 0) uniform Matrices{
    vec4 resolution;
};

layout(location = 0) in vec2 aTexCoords;
out vec2 TexCoords;

void main(){
    TexCoords = aTexCoords;
    ourColor = vec4(aTexCoords, .6, 1.0);
    gl_Position = vec4(vertices[gl_VertexID].pos.x / resolution.x * 2 - 1, vertices[gl_VertexID].pos.y / resolution.y * 2 - 1, 0., 1.);
}