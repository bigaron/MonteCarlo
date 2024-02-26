#include <vector>
#include <string>

#include "glew.h"
#include "GLFW/glfw3.h"

#include "Shader.h"
#include "Camera.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

int screenWidth = 1280, screenHeight = 720;
Camera camera((float)screenWidth / 2, (float)screenHeight / 2);

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
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); 

    if(GLenum err = glewInit(); GLEW_OK != err) {
        std::cout << glewGetErrorString(err) << std::endl;
        return -1;
    }

    glViewport(0, 0, screenWidth, screenHeight);
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl << "GPU: "<< glGetString(GL_RENDERER) << std::endl;

    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

    Shader shader("../src/shaders/vertex.vs", "../src/shaders/fragment.fs");
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    float verts[] = {
        .5f, -.49f,  .0f, 0.f, 1.f,
        .5f, .49f,   .0f, 0.f, 1.f,
        .49f, .5f,   .0f, 1.f, 0.f,
        -.49f, .5f,  .0f, 1.f, 0.f,
        -.5f, .49f,  1.f, 0.f, 0.f,
        -.5f, -.49f, 1.f, 0.f, 0.f,
        -.49f, -.5f, 1.f, 1.f, .0f,
        .49f, -.5f,  1.f, 1.f, .0f,
    };

    unsigned int indx[] = {0,1,3,1,2,3};

    float backgroundRect[] = {
        -.5f, -.5f,
        .5f, -.5f,
        .5f, .5f,
        -.5f, .5f
    };

    struct boundarySturct{
        glm::vec4 boundaryValue[4];
        glm::vec4 points[8];
        float pointN;   
        float eps;
        float sampleN;
        float padding2;
        //4byte + 4byte + 4*4*4 byte + 8*4*4 byte
        // 8byte + 64byte + 128byte = 
    };


    std::vector<glm::vec4> points = {glm::vec4(.5f, -.49f, .0f, 1.f), glm::vec4(.5f, .49f, 0.f, 1.f), glm::vec4(.49f, .5f, 0.f, 1.f), glm::vec4(-.49f, .5f,  .0f, 1.f), glm::vec4(-.5f, .49f, 0.f, 1.f),
        glm::vec4(-.5f, -.49f, 0.f, 1.f), glm::vec4(-.49f, -.5f, .0f, 1.f), glm::vec4(.49f, -.5f, 0.f, 1.f)};
    std::vector<glm::vec4> boundary = {glm::vec4(.0f, 0.f, 1.f, 1.f), glm::vec4(.0f, 1.f, 0.f, 1.f), glm::vec4(1.f, 0.f, 0.f, 1.f), glm::vec4( 1.f, 1.f, .0f, 1.f)};

    boundarySturct bound;
    bound.pointN = 8;
    bound.eps = .52f;
    bound.sampleN = 10;
    std::memcpy(bound.boundaryValue, boundary.data(), sizeof(glm::vec4) * 4);
    std::memcpy(bound.points, points.data(), sizeof(glm::vec4) * 8);

    unsigned int vao, vbo, ubo, ebo, bbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &ubo);
    glGenBuffers(1,  &bbo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo, 0, 2 * sizeof(glm::mat4));
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,  5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indx), indx, GL_STATIC_DRAW);

    GLuint ssbo;
    glCreateBuffers(1, &ssbo);
    glNamedBufferStorage(ssbo, sizeof(boundarySturct), (const void*)&bound, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo);


    glUseProgram(shader.ID);
    unsigned int fragDrawLoc;
    try{
        fragDrawLoc = shader.getUniformLocation("fragQuad");
    }catch(...){
        char cal;
        std::cout << "Enter in any key to close the program" << std::endl;
        std::cin >> cal;
        return 0;
    }

    glm::mat4 proj;
    glm::mat4 view = camera.getLookAt();
    glm::mat4 model;

    float deltaTime, lastFrame = .0f;
    
    auto currentFrame = (float)glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    camera.processInput(window, deltaTime);
    view = camera.getLookAt();
    proj = camera.getProjMtx();

    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), glm::value_ptr(proj)); 
    glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(view));

    int frag = 1;

    glBindBuffer(GL_ARRAY_BUFFER, bbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(backgroundRect), backgroundRect, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    frag = 2;
    glUniform1i(fragDrawLoc, frag);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    frag = 1;
    glUniform1i(fragDrawLoc, frag);
   // glDrawArrays(GL_LINES, 0, 8);

    glfwSwapBuffers(window);
    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();
    }


    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glfwTerminate();
    
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos){
    //camera.mouseLook(window, xpos, ypos);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
    camera.mouseScroll(window, xoffset, yoffset);
}   
