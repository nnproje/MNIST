#pragma once
#ifndef DATASET_H_INCLUDED
#define DATASET_H_INCLUDED
#include <fstream>
#include <vector>
#include "Volume.h"
#include "Matrix.h"
#include "ConvFeedForward.h"
typedef matrix<float> Matrix;
///////////////////////////////////////////////////////////////////////////////
//////////////////////////GET DATASET FROM HARD DISK///////////////////////////
///////////////////////////////////////////////////////////////////////////////
int LittleEndian(uint32_t ii);
void get_dataset(Matrix* X,Matrix* Y,const char*Xdir,const char*Ydir,int EXAMPLES);
void Shuffle(Matrix* X, Matrix* Y);
int get_dataset_2D(Volume& X, Matrix* Y,const char*Xdir, const char*Ydir, int EXAMPLES);
void Shuffle(Volume& X, Matrix* Y);
void SWAP(Matrix* MAT, int i, int k);
void SWAP(Volume& Vol, int i, int k);

void DevSet(Matrix* X, Matrix* Y, Matrix* X_dev, Matrix* Y_dev, int DEV);
void normalize(Matrix& X, Matrix& X_test, Matrix& Y, Matrix& Y_test);
void DELETE(Volume& X);
///////////////////////////////////////////////////////////////////////////////
//////////////////////////ELASTIC DISTORTION///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
Matrix* gausianFilter(int x, int y, float sigma);
Matrix* gaussianBlur(Matrix* img, int filterSize, float sigma);
float norm_L1(Matrix* x);
Matrix* elasticDistortion(Matrix* img, int filterSize, float sigma, float alpha);
void enlarge2D(Volume& X, Matrix& Y, int enlargeFact);
Matrix* enlarge1D(Volume& X, Matrix& Y, int enlargeFact);
///////////////////////////////////////////////////////////////////////////////
/////////////////////////END END END END END///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#endif // DATASET_H_INCLUDED
