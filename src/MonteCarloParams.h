#ifndef MONTECARLOPARAMS_H
#define MONTECARLOPARAMS_H

#include "glm.hpp"

struct MonteCarloParameters{
    float vertexN;
    float eps;
    float sampleN;
    float padding2;
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