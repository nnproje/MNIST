#pragma once
#include "Matrix.h"
#ifndef ACTIVATIONS_H_INCLUDED
#define ACTIVATIONS_H_INCLUDED
typedef matrix<float> Matrix;
////////////////////////////////////////////
Matrix softmax(Matrix& z);
////////////////////////////////////////////
Matrix sigmoid(Matrix& z);
///////////////////////////////////////////
Matrix mytanh(Matrix& z);
////////////////////////////////////////
Matrix relu(Matrix& z);
//////////////////////////////////////////
Matrix leakyRelu(Matrix& z);
//////////////////////////////////////////
Matrix Linear(Matrix& z);
//////////////////////////////////////////
Matrix satLinear(Matrix& z);
//////////////////////////////////////////
Matrix satLinear2(Matrix& z, float maxErr);
//////////////////////////////////////////
Matrix satLinear3(Matrix& z,float maxErr);
//////////////////////////////////////////
//////////////////////////////////////////
Matrix dsoftmax(Matrix& z);
//////////////////////////////////////////
Matrix dsigmoid(Matrix& z);
/////////////////////////////////////////
Matrix dtanh(Matrix& z);
/////////////////////////////////////////
Matrix drelu(Matrix& z);
//////////////////////////////////////////
Matrix dleakyRelu(Matrix& z);
////////////////////////////////////////
Matrix dLinear(Matrix& z);
/////////////////////////////////////////
Matrix dsatLinear(Matrix& z);
//////////////////////////////////////////////////
Matrix dsatLinear2(Matrix& z,float maxErr);
//////////////////////////////////////////////////
Matrix dsatLinear3(Matrix& z, float maxErr);
//////////////////////////////////////////////////
#endif // ACTIVATIONS_H_INCLUDED