#version 460 core

#define M_PI 3.1415926535897932384626433832795
const float inf = 100000000.f;
const int maxWalk = 200;

out vec4 FragColor;

in vec4 ourColor;
in vec2 TexCoords;

uniform sampler2D tex;

void main(){
    vec3 texCol = texture(tex, TexCoords).rgb;
    FragColor = vec4(texCol, 1.0);
}