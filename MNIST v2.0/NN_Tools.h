#ifndef NN_TOOLS_H_INCLUDED
#define NN_TOOLS_H_INCLUDED
#include <iostream>
////////////////////////////////////////////////////////
struct layer
{
    float neurons;
    std::string activation;
    float numOfLinearNodes;

    void put(float n, std::string activ)
    {
        neurons=n;
        activation=activ;
    }
};
////////////////////////////////////////////////////////
std::string CharGen(std::string name, int i);
////////////////////////////////////////////////////////

#endif // NN_TOOLS_H_INCLUDED
