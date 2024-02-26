#version 460 core
out vec4 ourColor;

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout(std140, binding = 0) uniform Matrices{
    mat4 projection;
    mat4 view;
};

uniform int fragQuad;

void main(){
    if(fragQuad == 2){    
        ourColor = vec4(.7, 0.2, 0.4, 0.0);
    }
    if(fragQuad == 1) {
        ourColor = vec4(aColor, 1.0f);
        //ourColor = vec4(.4, 0.4, 0.4, 1.0);
    }

    gl_Position = vec4(aPos, 0.f, 1.f);
}