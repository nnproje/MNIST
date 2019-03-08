#ifndef CONVBACKPROB_H_INCLUDED
#define CONVBACKPROB_H_INCLUDED
#include "ConvFeedForward.h"
typedef matrix<float> Matrix;
//////////////////////////////////////////////////////////*********************************\////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////|******POOL BACKPROPAGATION*******|////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////\*********************************/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix create_mask_from_window(Matrix & x);
Matrix distribute_value(float dz,int nh,int nw);
vector<Volume> pool_backward(vector<Volume>& dA,vector<Volume> & Aprev,int f,int stride, string mode);
void updateparameters (float alpha,int iteration,vector<Volume>& W, Volume& b,vector<Volume>& dW, Volume &db);
//////////////////////////////////////////////////////////*********************************\////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////|***CONVOLUTION BACKPROPAGATION***|////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////\*********************************/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ConvBackward(vector<Volume>& ACprev, vector<Volume>& dAC, vector<Volume>& filters, int s);
vector<Volume> FullConvolution(vector<Volume>& RotatedFilters, vector<Volume>& dAC);
void PadAllVolumes(vector<Volume>& Original, vector<Volume>& Result, int pad, int value);
void PadVolume(Volume& Original, Volume& Result, int pad, int value);
void RotateAllVolumes(vector<Volume>& filters, vector<Volume>& RotatedFilters);
void RotateVolume(Volume& filter, Volume& Result);
vector<Volume> FilterGrades(vector<Volume>& ACprev, vector<Volume>& dAC);


#endif // CONVBACKPROB_H_INCLUDED
