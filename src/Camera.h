#ifndef CAMERA_H
#define CAMERA_H

#include "glm.hpp"
#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"

#include <cmath>


class Camera{
    float yaw = -90.f, pitch = .0f, lastX, lastY, fov = 45.f;
    int width, height;
    glm::vec3 cameraPos, cameraFront, cameraUp;
    glm::vec3 direction;
    bool firstMouse = true; 
public:
    Camera(int screenW, int screenH, glm::vec3 cameraPos = glm::vec3(.0f, .0f, 3.f),
        glm::vec3 cameraFront = glm::vec3(.0f, .0f, -1.f), glm::vec3 cameraUp = glm::vec3(.0f, 1.f, .0f)): 
        lastX((float)screenW /2.0f), lastY((float)screenH / 2.0f), width(screenW), height(screenH), cameraPos(cameraPos), cameraFront(cameraFront), cameraUp(cameraUp){
            direction = glm::vec3(cos(glm::radians(yaw)) * cos(glm::radians(pitch)), 
                sin(glm::radians(pitch)),sin(glm::radians(yaw)) * cos(glm::radians(pitch)));
    }

    void mouseScroll(GLFWwindow* window, double xoffset, double yoffset){
        fov -= (float)yoffset;
        if (fov < 1.0f)
            fov = 1.0f;
        if (fov > 45.0f)
            fov = 45.0f; 
    }   

    glm::mat4 getLookAt(){
        return glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    }

    glm::mat4 getProjMtx(){
        return glm::perspective(glm::radians(fov), (float)width/(float)height, .1f, 100.f);
    }

    void mouseLook(GLFWwindow* window, double xpos, double ypos){
        if(firstMouse){
            lastX = (float)xpos;
            lastY = (float)ypos;
            firstMouse = false;
        }
        float xOff = (float)xpos - lastX;
        float yOff = -1* ((float)ypos - lastY);
        lastX = (float)xpos;
        lastY = (float)ypos;
        
        const float sens = .1f;
        xOff *= sens;
        yOff *= sens;

        yaw += xOff;
        pitch += yOff; 
        
        if(pitch > 89.0f)
            pitch =  89.0f;
        if(pitch < -89.0f)
            pitch = -89.0f;

        direction = glm::vec3(cos(glm::radians(yaw)) * cos(glm::radians(pitch)), sin(glm::radians(pitch)),sin(glm::radians(yaw)) * cos(glm::radians(pitch)));
        cameraFront = glm::normalize(direction);
    }

    void processInput(GLFWwindow* window, float deltaTime){
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        const float cameraSpeed = 2.5f * deltaTime; 
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            cameraPos += cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            cameraPos -= cameraSpeed * cameraFront;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    }
};
#endif