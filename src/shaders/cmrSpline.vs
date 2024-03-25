//Catmullrom Spline 
#version 460 core

layout(std140, binding = 0) uniform Matrices{
    vec4 resolution;
};

layout(location = 0) in vec4 pos;

void main(){
    gl_Position = vec4(pos.x / resolution.x * 2 - 1, pos.y / resolution.y * 2 - 1, pos.zw);
}