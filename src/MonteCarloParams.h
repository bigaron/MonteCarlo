#ifndef MONTECARLOPARAMS_H
#define MONTECARLOPARAMS_H

#include "glm/glm.hpp"

struct MonteCarloParameters{
    float vertexN;
    float eps;
    float sampleN;
    unsigned int maxWalkN;
};

struct BoundaryTopology{
    float pointN;   
    float padding1;
    float padding2;
    float padding3;

    glm::vec4 points[];
};

struct BoundaryValues{
    float boundaryN;
    float padding1;
    float padding2;
    float padding3;

    glm::vec4 boundValues[];
};

#endif