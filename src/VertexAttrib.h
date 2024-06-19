#ifndef VERTEXATTRIB_H
#define VERTEXATTRIB_H

#include "glm/glm.hpp"
#include <cstring>
#include <cuda_runtime.h>


struct VertexAttrib{
    glm::vec4 pos;
    glm::vec4 col;
};

struct cudaVertexAttrib {
    float4 pos;
    float4 col;
};
#endif