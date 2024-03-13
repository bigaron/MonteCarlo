#include <vector>
#include <string>

#include "glew.h"
#include "GLFW/glfw3.h"

#include "Shader.h"
#include "Camera.h"
#include "MonteCarloParams.h"
#include "VertexAttrib.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

const int screenWidth = 1280, screenHeight = 720;

int main(){
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "MonteCarlo", nullptr, nullptr);
    if (window == nullptr){
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if(GLenum err = glewInit(); GLEW_OK != err) {
        std::cout << glewGetErrorString(err) << std::endl;
        return -1;
    }
    glViewport(0, 0, screenWidth, screenHeight);
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl << "GPU: "<< glGetString(GL_RENDERER) << std::endl;
    glEnable(GL_DEPTH_TEST);

    Shader shader;
    shader.GraphicsShader("../src/shaders/vertex.vs", "../src/shaders/fragment.fs");
    shader.ComputeShader("../src/shaders/monteCarlo.comp");

    int halfW = screenWidth / 2, halfH = screenHeight / 2;
    const int side = 320;
    std::vector<glm::vec4> points = {glm::vec4(halfW + side, halfH - side, 0., 1.), glm::vec4(halfW + side, halfH + side, 0., 1.), glm::vec4(halfW - side, halfH + side, 0., 1.), glm::vec4(halfW - side, halfH - side, 0., 1.)};
    std::vector<glm::vec4> boundary = {glm::vec4(0.f, 0.f, .0f, 1.f), glm::vec4(0.f, 1.f, .0f, 1.f), glm::vec4( 1.f, .0f, 1.0f, 1.f), glm::vec4( 1.f, 1.f, 1.0f, 1.f)};
    std::vector<VertexAttrib> vtxs, bounds;
    VertexAttrib vertices;
    for(int i = 0; i < points.size(); ++i) {
        vertices = copyValuesToVertexAttrib(vertices, points[i], boundary[i]);
        vtxs.push_back(vertices);
        bounds.push_back(vertices);
    }
    bounds.push_back(copyValuesToVertexAttrib(vertices, points[0], boundary[0]));

    MonteCarloParameters mcParms;
    mcParms.eps = 0.1f;
    mcParms.sampleN = 1000;
    mcParms.vertexN = 5;

    GLint localNum[3];
    GLint maxInvocation;
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxInvocation);
    for(int i = 0; i < 3; ++i) glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &localNum[i]);
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
    glNamedBufferStorage(vertxAttrSSBO, sizeof(VertexAttrib) * vtxs.size(), (const void*) vtxs.data(), GL_DYNAMIC_STORAGE_BIT);
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 2*side, 2*side, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glUseProgram(shader.computeID);
    GLuint offsetUBO;
    glCreateBuffers(1, &offsetUBO);
    glm::vec4 offs(screenWidth/2 - side, screenHeight/2 - side, 0.0, 0.0);  
    glNamedBufferStorage(offsetUBO, sizeof(glm::vec4), (const void*)&offs, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, offsetUBO);
    glDispatchCompute((GLuint) 2*side / 32, (GLuint) 2*side / 32, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glUseProgram(shader.graphicsID);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_FAN, 0, (int)vtxs.size());
    glfwSwapBuffers(window);
    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();
    }
    
    glDeleteBuffers(1, &vertxAttrSSBO);
    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height){
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
