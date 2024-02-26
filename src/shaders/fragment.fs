#version 460 core

#define M_PI 3.1415926535897932384626433832795
const float inf = 100000000.f;

out vec4 FragColor;

in vec4 ourColor;

struct VertexData{
    vec4 pos;
    vec4 value;
};

//Struct HAS to take up N * sizeof(vec4) space
struct MonteCarloParams{
    float vertexN;
    float eps;
    float sampleN;
    float padding2;
};

layout(std430, binding = 1) readonly buffer ssbo1{
    MonteCarloParams params;
    VertexData boundaryPoints[];
};

layout(std140, binding = 0) uniform Matrices{
    vec4 resolution;
};

//Result ranges from [0-1]
//Source: https://en.wikipedia.org/wiki/Lehmer_random_number_generator
float rand(int seed){
    int a = 16807, m = 2147483647;
    float m_f = 2147483647;
    float ret = float((a * seed) % m);
    return ret / m_f;
}

float distanceFromLine(vec4 p, vec4 x_o, vec4 x_1){
    vec4 v = (x_1 - x_o) / length(x_1 - x_o);
    vec4 r = p - x_o;
    vec4 a = dot(r,  v) * v;
    return length(r - a);
}

float closestPoint(vec4 x_o){
    float dist = inf;
    for(int i = 0; i < params.vertexN - 1; i++){
        float distFromCurrentWall = distanceFromLine(x_o, boundaryPoints[i].pos, boundaryPoints[i+1].pos);
        if(distFromCurrentWall < dist) dist = distFromCurrentWall;
    }
    return dist;
}

vec4 getBoundaryValue(vec4 x_o){
    float dist = inf;
    int boundIndx = 0;
    vec4 ret;
    for(int i = 0; i < params.vertexN - 1; ++i) {
        float currentDist = distanceFromLine(x_o, boundaryPoints[i].pos, boundaryPoints[i+1].pos);
        if(currentDist < dist) {
            dist = currentDist;
            ret = boundaryPoints[i].value;
        }
    }

    return ret;
}

vec4 MonteCarloEstim(vec4 currentOrigin){
    float radius = closestPoint(currentOrigin);
    while(radius > params.eps) {  
        int seed = int(currentOrigin.x * 2000000); 
        float th = rand(seed) * 2 *  M_PI;
        currentOrigin = vec4(currentOrigin.x + radius * cos(th), currentOrigin.y + radius * sin(th), currentOrigin.zw);
        radius = closestPoint(currentOrigin);
    }

    return getBoundaryValue(currentOrigin);    
}

void main(){
    vec4 sum = vec4(0.0f);
    for(int i = 0; i < params.sampleN; ++i){
        sum += MonteCarloEstim(gl_FragCoord);
    }
    sum /= params.sampleN;
    sum.w = 1.;

    FragColor = sum;
}