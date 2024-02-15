#ifndef SHADER_H
#define SHADER_H

#include "glew.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader{
public: 
    unsigned int ID;

    Shader(const char* vtxShdrPath, const char* fgmtShdrPath){
        std::string vtxCode, fgmntCode;
        std::ifstream vShaderFile, fShaderFile;
        vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        try{
            vShaderFile.open(vtxShdrPath);
            fShaderFile.open(fgmtShdrPath);
            std::stringstream vShdrStream, fShdrStream;
            vShdrStream << vShaderFile.rdbuf();
            fShdrStream << fShaderFile.rdbuf();
            vShaderFile.close();
            fShaderFile.close();
            vtxCode = vShdrStream.str();
            fgmntCode = fShdrStream.str();
        }catch(std::ifstream::failure& e){
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ( " << e.what() << " )" << std::endl;
        }
        const char* vShaderCode = vtxCode.c_str();
        const char* fShaderCode = fgmntCode.c_str();
        unsigned int vertex, fragment;
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, nullptr);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");

        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, nullptr);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");

        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");

        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }

private:
    void checkCompileErrors(unsigned int shader, const std::string& type) const{
        int success;
        char infoLog[1024];
        if (type != "PROGRAM")
        {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success)
            {
                glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else
        {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success)
            {
                glGetProgramInfoLog(shader, 1024, nullptr, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
    }
};

#endif