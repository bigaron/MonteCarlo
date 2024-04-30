#ifndef VERTEXATTRIB_H
#define VERTEXATTRIB_H

#include "glm/glm.hpp"
#include <cstring>

struct VertexAttrib{
    glm::vec4 pos;
    glm::vec4 col;
};

VertexAttrib copyValuesToVertexAttrib(VertexAttrib src, const glm::vec4& pos,const glm::vec4& col){
    src.pos = pos;
    src.col = col;
    return src;
}
#endif