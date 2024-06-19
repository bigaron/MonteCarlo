#include <vector>
#include <string>

#include "glew.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"

#include "Shader.h"
#include "MonteCarloParams.h"
#include "VertexAttrib.h"

#include <cuda_runtime.h>
#include <cudaSrc/helper_cuda.h>
#include <cuda_gl_interop.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_motion_callback(GLFWwindow* window, int button, int action, int mods);
std::vector<glm::vec4> calculateBezierCurve(const std::vector<glm::vec4>& controlPoints, float timeStep);
VertexAttrib copyValuesToVertexAttrib(VertexAttrib src, const glm::vec4& pos, const glm::vec4& col) {
    src.pos = pos;
    src.col = col;
    return src;
}

cudaVertexAttrib copyValuesToCudaVertex(cudaVertexAttrib src, const glm::vec4& pos, const glm::vec4& col){
    src.col = make_float4(col.x, col.y, col.z, col.w);
    src.pos = make_float4(pos.x, pos.y, pos.z, pos.w);
    return src;
}


const int screenWidth = 1280, screenHeight = 736;
const int maxPoints = 100;
const glm::vec4 white(1.0, 1.0, 1.0, 1.0);
const glm::vec4 black(16.0f/255, 24.0f/255, 32.0f/255, 1.0f);
const glm::vec4 yellow(254.0f/255, 231.0f/255, 21.0f/255, 1.0f);
const glm::vec4 red(1.0f, 0.0f, 0.0f, 1.0f);
const glm::vec4 blue(0.0f, 0.0f, 1.0f, 1.0f);

extern "C" void setupConstants(const MonteCarloParameters & simulationParams);
extern "C" void computeMonteCarloWithTexture(cudaSurfaceObject_t surface, cudaVertexAttrib * boundPointer, dim3 grid, dim3 block, int imgW, int imgH, int iteration);


void initForCuda(GLuint* pbo, struct cudaGraphicsResource** pbo_resource);


GLFWwindow* contextSetup();

int main() {
    GLFWwindow* window = contextSetup();

    Shader shader;
    shader.GraphicsShader("src/shaders/vertex.vs", "src/shaders/fragment.fs");
    shader.ComputeShader("src/shaders/monteCarlo.comp");

    std::vector<glm::vec4> cps0 = { glm::vec4(120, 300 ,0 ,1), glm::vec4(250, 290, 0 , 1), glm::vec4(700, 360, 0 , 1), glm::vec4(600, 600, 0, 1) };
    std::vector<glm::vec4> cps1 = { glm::vec4(460, 270, 0, 1), glm::vec4(640, 360, 0, 1), glm::vec4(960, 450, 0, 1), glm::vec4(1220, 360, 0, 1) };
    std::vector<glm::vec4> controlPoints;
    controlPoints = calculateBezierCurve(cps0, 0.01f);
    size_t siz = controlPoints.size();
    std::vector<glm::vec4> newC = calculateBezierCurve(cps1, 0.01f);
    for (auto& val : newC) controlPoints.push_back(val);
    std::vector<glm::vec4> points = { glm::vec4(screenWidth, 0, 0., 1.), glm::vec4(screenWidth, screenHeight, 0., 1.), glm::vec4(0, screenHeight, 0., 1.), glm::vec4(0, 0, 0., 1.) };
    std::vector<glm::vec4> boundary = { glm::vec4(0.f, 0.f, .0f, 1.f), glm::vec4(0.f, 1.f, .0f, 1.f), glm::vec4(1.f, .0f, 1.0f, 1.f), glm::vec4(1.f, 1.f, 1.0f, 1.f) };
    std::vector<VertexAttrib> vtxs, bounds;
    std::vector<cudaVertexAttrib> cudaBounds;
    VertexAttrib vertices = VertexAttrib();
    cudaVertexAttrib cudaVerx = cudaVertexAttrib();
    for (int i = 0; i < points.size(); ++i) {
        vertices = copyValuesToVertexAttrib(vertices, points[i], boundary[i]);
        vtxs.push_back(vertices);
    }
    int cntr = 0;
    for (int i = 0; i < controlPoints.size(); ++i) {
        if (cntr == 19) cntr = 0;
        glm::vec4 colour;
        if (i < siz) {
            colour = cntr > 10 ? red : white;
        }
        else colour = cntr > 10 ? yellow : blue;
        vertices = copyValuesToVertexAttrib(vertices, controlPoints[i], colour);
        bounds.push_back(vertices);
        cudaVerx = copyValuesToCudaVertex(cudaVerx, controlPoints[i], colour);
        cudaBounds.push_back(cudaVerx);
        cntr++;
    }
    /*cudaBounds.clear();
    cudaVerx.col = make_float4(1.0, 0.0, 0.0, 1.0);
    cudaVerx.pos = make_float4(700, 308, 1, 1);
    cudaBounds.push_back(cudaVerx);

    cudaVerx.col = make_float4(0.0, 0.0, 1.0, 1.0);
    cudaVerx.pos = make_float4(700, 428, 1, 1);
    cudaBounds.push_back(cudaVerx);

    cudaVerx.col = make_float4(0.0, 1.0, 0.0, 1.0);
    cudaVerx.pos = make_float4(580, 428, 1, 1);
    cudaBounds.push_back(cudaVerx);

    cudaVerx.col = make_float4(0.0, 0.0, 0.0, 1.0);
    cudaVerx.pos = make_float4(580, 308, 1, 1);
    cudaBounds.push_back(cudaVerx);*/

    //bounds.push_back(copyValuesToVertexAttrib(vertices, controlPoints[0], white));

    MonteCarloParameters mcParms;
    mcParms.eps = 1.5f;
    mcParms.sampleN = 8;
    mcParms.vertexN = (float)cudaBounds.size();
    mcParms.maxWalkN = 40;

    GLint maxInvocation;
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxInvocation);
    std::cout << "MAX INVOCATIONS PER WORKGROUP: " << maxInvocation << std::endl;

    unsigned int vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glUseProgram(shader.graphicsID);
    glBindVertexArray(vao);

    GLuint boundSSBO;
    glCreateBuffers(1, &boundSSBO);
    glNamedBufferStorage(boundSSBO, sizeof(MonteCarloParameters) + sizeof(VertexAttrib) * bounds.size(), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, boundSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(MonteCarloParameters), (const void*)&mcParms);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(MonteCarloParameters), sizeof(VertexAttrib) * bounds.size(), (const void*)bounds.data());

    unsigned int windowContextUBO;
    glCreateBuffers(1, &windowContextUBO);
    glm::vec4 resVec(screenWidth, screenHeight, 0., 0.);
    glNamedBufferStorage(windowContextUBO, sizeof(glm::vec4), (const void*)&resVec, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, windowContextUBO);

    unsigned int vertxAttrSSBO;
    glCreateBuffers(1, &vertxAttrSSBO);
    glNamedBufferStorage(vertxAttrSSBO, sizeof(VertexAttrib) * vtxs.size(), (const void*)vtxs.data(), GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertxAttrSSBO);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, windowContextUBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, boundSSBO);

    GLuint texLoc = shader.getUniformLocation("tex", shader.graphicsID);
    glUniform1i(texLoc, 0);
    float tex[] = {
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        0.0, 0.0
    };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tex), tex, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);




    GLuint texture;
    glGenTextures(1, &texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screenWidth, screenHeight, 0, GL_RGBA, GL_FLOAT, nullptr);

    glBindTexture(GL_TEXTURE_2D, 0);

    GLuint pbo;
    struct cudaGraphicsResource *d_Resource;
    //initForCuda(&pbo, &d_Resource);
    setupConstants(mcParms);

    //PIXEL BUFFER OBJECT SETUP
    GLsizeiptr pboSize = screenHeight * screenWidth * sizeof(GLubyte) * 4;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, pboSize, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    //Register texture to CUDA
    struct cudaGraphicsResource* imageCUDAPointer;
    checkCudaErrors(cudaGraphicsGLRegisterImage(&imageCUDAPointer, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    checkCudaErrors(cudaGraphicsMapResources(1, &imageCUDAPointer, nullptr));

    //Get pointer to the texture resource
    struct cudaArray* imageCUDAArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&imageCUDAArray, imageCUDAPointer, 0, 0));
    cudaResourceDesc wdsc;
    wdsc.resType = cudaResourceTypeArray;
    wdsc.res.array.array = imageCUDAArray;

    cudaSurfaceObject_t writeSurface;
    checkCudaErrors(cudaCreateSurfaceObject(&writeSurface, &wdsc));
    //Binding cuda array to surface reference


    //glUseProgram(shader.computeID);
    GLuint offsetUBO;
    glCreateBuffers(1, &offsetUBO);
    glm::vec4 offs(0.0, 0.0, 0.0, 0.0);
    glNamedBufferStorage(offsetUBO, sizeof(glm::vec4), (const void*)&offs, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, offsetUBO);
    

    //===========================CUDA VARIABLES===========================
    float4* resourcePointer;
    cudaVertexAttrib* structPointer;
    dim3 blocks(32, 32, 1);
    dim3 grid(screenWidth / blocks.x, screenHeight / blocks.y, 1);
    size_t bytes;
    //===========================CUDA VARIABLES===========================


    //checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&resourcePointer, &bytes, d_Resource));
    checkCudaErrors(cudaMalloc((void**)&structPointer, cudaBounds.size() * sizeof(cudaVertexAttrib)));
    checkCudaErrors(cudaMemcpy(structPointer, cudaBounds.data(), cudaBounds.size() * sizeof(cudaVertexAttrib), cudaMemcpyHostToDevice));
    
    
    computeMonteCarloWithTexture(writeSurface, structPointer, grid, blocks, screenWidth, screenHeight, 1);
    //Wait for texture to stop rendering
    cudaDeviceSynchronize();
    checkCudaErrors(cudaDestroySurfaceObject(writeSurface));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &imageCUDAPointer));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    unsigned int iter = 1;
    while(!glfwWindowShouldClose(window)){
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //if (iter < 2) {
        //    //computeMonteCarlo(resourcePointer, structPointer, grid, blocks, screenWidth, screenHeight, iter);
        //    //Wait for texture to stop rendering
        //    //cudaDeviceSynchronize();
        //    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        //    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screenWidth, screenHeight, 0, GL_RGBA, GL_FLOAT, nullptr);
        //    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        //    glActiveTexture(GL_TEXTURE0);
        //    iter++;
        //}
        
        glUseProgram(shader.graphicsID);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, (int)vtxs.size());
        glfwPollEvents();
        glfwSwapBuffers(window);
    }
    
    //checkCudaErrors(cudaGraphicsUnmapResources(1, &d_Resource, 0));

    glDeleteBuffers(1, &vertxAttrSSBO);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    glfwTerminate();
    //cudaFree(d_Resource);
    //checkCudaErrors(cudaFree(structPointer));
    return 0;
}

std::vector<glm::vec4> calculateBezierCurve(const std::vector<glm::vec4>& controlPoints, float timeStep){
    std::vector<glm::vec4> points;

    for(auto t = 0.0f; t < 1.0f; t += timeStep){
        glm::vec4 point;
        float oneMinT = 1.0f-t;
        point = powf(oneMinT, 3) * controlPoints[0] + 3 * powf(oneMinT, 2) * t * controlPoints[1] + 3 * oneMinT * t * t * controlPoints[2] + powf(t, 3) * controlPoints[3];
        points.push_back(point);
    }

    return points;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height){
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void mouse_motion_callback(GLFWwindow* window, int button, int action, int mods){
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        
    }
}

GLFWwindow* contextSetup(){
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "MonteCarlo", nullptr, nullptr);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (window == nullptr){
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);
    GLenum err = glewInit();
    if(GLEW_OK != err) {
        std::cout << glewGetErrorString(err) << std::endl;
        exit(-1);
    }
    glViewport(0, 0, screenWidth, screenHeight);
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl << "GPU: "<< glGetString(GL_RENDERER) << std::endl;
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_motion_callback);

    return window;
}

void initForCuda(GLuint* pbo, struct cudaGraphicsResource** pbo_resource) {
    unsigned int sizeByte = screenWidth * screenHeight * 4 * sizeof(float);

    glGenBuffers(1, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeByte, nullptr, GL_STREAM_DRAW);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsMapFlagsNone));

    checkCudaErrors(cudaGraphicsMapResources(1, pbo_resource, 0));
}