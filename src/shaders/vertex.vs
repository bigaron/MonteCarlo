#version 450 core
out vec4 ourColor;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout(std140, binding = 0) uniform Matrices{
    mat4 projection;
    mat4 view;
};

uniform mat4 model;

void main(){
    ourColor = vec4(aColor, 1.0f);
    gl_Position = projection * view * model * vec4(aPos, 1.0f);
}