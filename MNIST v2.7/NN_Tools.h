#ifndef NN_TOOLS_H_INCLUDED
#define NN_TOOLS_H_INCLUDED
#include <iostream>
#include "Matrix.h"
#include "Activations.h"
typedef matrix<float> Matrix;
////////////////////////////////////////////////////////
struct layer
{
    float neurons;
    ActivationType activation;
    float numOfLinearNodes;

    void put(float n, ActivationType activ)
    {
        neurons=n;
        activation=activ;
    }
};
////////////////////////////////////////////////////////
std::string CharGen(std::string name, int i);
////////////////////////////////////////////////////////
void AccuracyTest(Matrix* Y, Matrix* Y_hat, string devOrtest);
#endif // NN_TOOLS_H_INCLUDED
