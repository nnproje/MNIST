#pragma once
#include "Volume.h"
typedef matrix<float> Matrix;
/**********************************************************************************/
/**********************************************************************************/
Volume	to_2D(Matrix* X);
Matrix* to_1D(Volume& X_2D);
Matrix* pad(Matrix* img, int p, float value);
Matrix* convolve(Volume& Aprev, Volume& filter, int s);
vector<Volume> convLayer(vector<Volume>& Aprev, vector<Volume>& filters, Volume b);
void maxPool(Volume& Aprev, Volume& A, int f, int s);
void avgPool(Volume& Aprev, Volume& A, int f, int s);
vector<Volume> poolLayer(vector<Volume>& Aprev, int f, string mode);
/**********************************************************************************/
/**********************************************************************************/
