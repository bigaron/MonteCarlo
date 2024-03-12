#version 460 core

#define M_PI 3.1415926535897932384626433832795
const float inf = 100000000.f;
const int maxWalk = 200;

out vec4 FragColor;

in vec4 ourColor;
in vec2 TexCoords;

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


uniform sampler2D tex;

//==============================================================
//Rand function from Visual Studio
//Actual source: https://www.shadertoy.com/view/WdXfzl
int seed = 1;
int rand(void) { seed = seed*0x343fd+0x269ec3; return (seed>>16)&32767; }
float frand(void) { return float(rand())/32767.0; }
void  srand( ivec2 p, int frame ){
    int n = frame;
    n = (n<<13)^n; n=n*(n*n*15731+789221)+1376312589; // by Hugo Elias
    n += p.y;
    n = (n<<13)^n; n=n*(n*n*15731+789221)+1376312589;
    n += p.x;
    n = (n<<13)^n; n=n*(n*n*15731+789221)+1376312589;
    seed = n;
}
//==============================================================
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
    int it = 0;
    float radius = closestPoint(currentOrigin);
    int seed = int(currentOrigin.x * 2000000); 
    while(it < maxWalk && radius > params.eps) {  
        it++;
        float th = frand() * 2 *  M_PI;
        currentOrigin = vec4(currentOrigin.x + radius * cos(th), currentOrigin.y + radius * sin(th), currentOrigin.zw);
        radius = closestPoint(currentOrigin);
    }
    if(it == maxWalk) return vec4(0.,0.,0.,1.);
    return getBoundaryValue(currentOrigin);    
}

void main(){
    //vec2 texCoord = vec2((gl_FragCoord.x - 340) / 600, (gl_FragCoord.y - 60) / 600);
    vec3 texCol = texture(tex, TexCoords).rgb;
    FragColor = vec4(texCol, 1.0);
}