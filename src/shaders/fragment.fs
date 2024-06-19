#version 460 core

out vec4 FragColor;

in vec4 ourColor;
in vec2 TexCoords;

uniform sampler2D tex;

void main(){
    //vec3 texCol = texture(tex, TexCoords).rgb;
    vec4 finalText = texture(tex, TexCoords);
    FragColor = vec4(finalText.xyz / finalText.w, 1.0f);
    //FragColor = finalText;
}