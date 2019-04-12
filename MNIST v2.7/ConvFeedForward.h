#pragma once
#include "VectVolume.h"
typedef matrix<float> Matrix;
/**********************************************************************************/
/**********************************************************************************/
VectVolume to_VectorOfVolume(Matrix* A, int nh, int nw, int nc, int m);
Volume	to_2D(Matrix* X);
Matrix* to_1D(Volume& X_2D);
Matrix* to_FC(VectVolume A);
Matrix* pad(Matrix* img, int p, float value);
Matrix* convolve(Volume& Aprev, Volume& filter, int s);
void maxPool(Volume& Aprev, Volume& A, int f, int s);
void avgPool(Volume& Aprev, Volume& A, int f, int s);
/**********************************************************************************/
/**********************************************************************************/
