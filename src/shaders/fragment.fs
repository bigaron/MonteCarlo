#version 460 core

#define M_PI 3.1415926535897932384626433832795
const float inf = 100000000.f;


out vec4 FragColor;

in vec4 ourColor;

//Struct HAS to take up N * sizeof(vec4) space
struct MonteCarloParams{
    vec4 boundaryValue[4];
    vec4 points[8];
    float pointN;
    float eps;
    float sampleN;
    float padding2;
};

layout(std430, binding = 1) readonly buffer ssbo1{
    MonteCarloParams params;
};

//Result ranges from [0-1]
//Source: https://en.wikipedia.org/wiki/Lehmer_random_number_generator
float rand(int seed){
    int a = 16807, m = 2147483647;
    float m_f = 2147483647;
    float ret = float((a * seed) % m);
    return ret / m_f;
}

float closestPoint(vec4 x_o){
    float dist = inf;
    for(int i = 0; i < params.pointN; i += 2){
        vec4 v = (params.points[i+1] - params.points[i]) / length(params.points[i+1] - params.points[i]);
        vec4 p = x_o - params.points[i];
        vec4 a = dot(p,  v) * v;
        float distFromCurrentWall = length(p - a);
        if(distFromCurrentWall < dist) dist = distFromCurrentWall;
    }
    return dist;
}

vec4 getBoundaryValue(vec4 x_o){
    float dist = inf;
    int boundIndx = 0;
    vec4 ret;
    for(int i = 0; i < params.pointN; ++i) {
        if(i != 0 && i % 2 == 0) boundIndx++;
        float currentDist = length(x_o - params.points[i]);
        if(currentDist < dist) {
            dist = currentDist;
            ret = params.boundaryValue[boundIndx];
        }
    }

    return ret;
}

vec4 MonteCarloEstim(vec4 currentOrigin){
    float closest = closestPoint(currentOrigin);
    while(closest > params.eps) {  
        int seed = int(currentOrigin.x * 2000000); 
        float th = rand(seed) * 2 *  M_PI;
        currentOrigin = vec4(currentOrigin.x + closest * cos(th), currentOrigin.y + closest * sin(th), currentOrigin.zw);
        closest = closestPoint(currentOrigin);
    }

    return getBoundaryValue(currentOrigin);    
}

void main(){
    vec4 pos = vec4((gl_FragCoord.x / 1280.0f * 2) - 1, (gl_FragCoord.y / 720.0f * 2) - 1, gl_FragCoord.zw);
    vec4 sum = vec4(0.0f), sum2 = vec4(1.f);
    sum += sum2;
    for(int i = 0; i < params.sampleN; ++i){
        sum += MonteCarloEstim(pos);
    }
    sum /= params.sampleN;
    sum.w = 1.;

    FragColor = sum;

}