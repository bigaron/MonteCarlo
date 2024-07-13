#version 460 core

out vec4 FragColor;

in vec4 ourColor;
in vec2 TexCoords;

uniform sampler2D tex;

void main(){
    vec4 texCol = texture(tex, TexCoords);
    //FragColor = texCol
    FragColor = vec4(texCol.xyz / texCol.w, 1);
}