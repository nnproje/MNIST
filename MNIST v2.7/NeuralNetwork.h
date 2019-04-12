#pragma once
#ifndef CNN_HEADER
#define CNN_HEADER
#include "Dictionary.h"
#include "Matrix.h"
#include "Volume.h"
#include "VectVolume.h"
#include "NN_Tools.h"
#include "Activations.h"
#include "ConvFeedForward.h"
#include "ConvBackProb.h"
typedef matrix<float> Matrix;
typedef Dictionary<string, Matrix*> Mat_Dictionary;
typedef Dictionary<string, Volume> Vol_Dictionary;
typedef Dictionary<string, VectVolume > VectVol_Dictionary;
enum TypeOfNet {FC,LENET1};
enum Optimizer {ADAM,GRADIENT_DESCENT};
enum ErrorType {SQAURE_ERROR,CROSS_ENTROPY};
enum Mode {TRAIN,DEV,TEST,MAX,AVG};
//*************************************************************************************************/
//4D discription for a 4D element A(m,nc,nh,nw):
//1- A is a vector of volumes that has a size m
//2- A[i] is a volume with nc channels, it represents the activations of some layer for the ith example
//3- A[i][j] is a pointer to the jth Matrix with nh hight and nw width in the ith example
//4- A[i][0] represents the first channel in the volume, take them from top to down
//input:
//Aprev(m,nc_prev,nh_prev,nw_prev) , filters(numOfFilters,nc,f,f) , b(numOfFilters,1)
//output:
//A(m,numOfFilters,nh,nw)
//*************************************************************************************************/
class NeuralNetwork
{
private:
	VectVol_Dictionary Conv_Weights;     // Dictionary containing weights of convolution layers
	VectVol_Dictionary Conv_Cache;	     // Dictionary containing temporaral values of internal activations of convolution layers
	VectVol_Dictionary Conv_Grades;      // Dictionary containing gradients of weights and biases of convolution layers
	VectVol_Dictionary ADAM_dWC;			 // Dictionary containing ADAM dW gradients of conv layers
	Mat_Dictionary Conv_biases;          //	Dictionary containing biases of convolution layers
	Mat_Dictionary Conv_dbiases;         //	Dictionary containing biases of convolution layers
	Mat_Dictionary ADAM_dbC;				 // Dictionary containing ADAM db gradients of conv layers 
	TypeOfNet NetType;					 // Type of architecture used in the network
	ErrorType ErrType;					 // Type of cost function or performance index used
	Optimizer optimizer;			     // Type of algorithm used
	bool momentum;						 // Indicates whether momentum is used or not
	bool isLastepoch;					 // Label for the last epoch
private:
	Mat_Dictionary FC_Parameters;		 // Dictionary containing weights and biases of fully connected layers
	Mat_Dictionary FC_Cache;		     // Dictionary containing temporaral values of internal activations of fully connected layers
	Mat_Dictionary FC_Grades;			 // Dictionary containing gradients of weights and biases of fully connected layers
	Mat_Dictionary FC_ADAM;
	Matrix** D;							 // Dropout Matrices in fully connected layers
	layer* layers;						 // Array of layers holding the number of neurons at each fully connected layer and its activation
	int numOfLayers;					 // Number of fully connected layers

public:
	NeuralNetwork(TypeOfNet TN);
	NeuralNetwork(TypeOfNet TN, layer* Layers, int L);
	void train(Matrix* X ,Matrix* Y, Matrix* X_div, Matrix* Y_div, float alpha, int numOfEpochs, int minibatchSize, Optimizer Op,int Numprint, ErrorType ET,float lambda,bool batchNorm,bool dropout,float* keep_prob);
	void test(Matrix* X_test, Matrix* Y_test, string path, Mode devOrtest, bool batchNorm, bool dropout, float* keep_prob);
private:
	void init_LeNet1();
	void train_LeNet1(Matrix* X, Matrix* Y, Matrix* X_div, Matrix* Y_div, float alpha, int numOfEpochs, int minibatchSize, Optimizer Op,int Numprint, ErrorType ET,float lambda,bool batchNorm,bool dropout,float* keep_prob);
	Matrix* test_LeNet1(Matrix* X_test, Matrix* Y_test,bool batchNorm, bool dropout, float* keep_prob); //forward m examples and return the predictions in Y_hat; //forward m examples through LeNet1 and return the predictions in Y_hat
private:
    void init_FC();
    void train_FC(Matrix* X, Matrix* Y, Matrix* X_div, Matrix* Y_div,float alpha, int numOfEpochs, int minibatchSize, Optimizer op,int Numprint, ErrorType ET,float lambda,bool batchNorm,bool dropout,float* keep_prob);
	Matrix* test_FC(Matrix* X_test, Matrix* Y_test,bool batchNorm, bool dropout, float* keep_prob); //forward m examples and return the predictions in Y_hat; //forward m examples through LeNet1 and return the predictions in Y_hat
private:
	Matrix* FC_FeedForward(bool batchNorm, bool dropout, float* keep_prob, Mode mode);
	void FC_BackProp(Matrix* X, Matrix* Y, Matrix* Y_hat, float alpha,int iteration,float lambda, bool batchNorm, bool dropout, float* keep_prob);
	void FC_CalGrads(Matrix* X, Matrix* Y, Matrix* Y_hat,float lambda, bool batchNorm, bool dropout, float* keep_prob);
	void FC_UpdateParameters(float alpha,int iteration,bool batchNorm);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void poolLayer(int stride,int f, Mode mode,int A_index);
	void convLayer(int stride, int A_index, ActivationType activation);
    void ConvBackwardOptimized(int stride,int A_index, ActivationType activation);
    void pool_backward(int f,int stride, Mode mode,int A_index);
    void updateparameters (float alpha,int iteration,int W_index);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
};
#endif // !CNN_HEADER

