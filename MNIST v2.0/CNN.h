#pragma once
#ifndef CNN_HEADER
#define CNN_HEADER
#include "Activations.h"
#include "CharGenerate.h"
#include "Dictionary.h"
#include "layer.h"
#include "ConvFeedForward.h"

typedef Dictionary<string, matrix<float> > FC_Dictionary;
typedef Dictionary<string, matrix<float> > Conv_Dictionary;
typedef matrix<float> Matrix;

class CNN
{
private:
	Conv_Dictionary Conv_Parameters;	 //	Dictionary containing weights and biases of convolution layers
	Conv_Dictionary Conv_Cache;			 // Dictionary containing temporaral values of internal activations of convolution layers
	Conv_Dictionary Conv_Grades;         // Dictionary containing gradients of weights and biases of convolution layers 
	string ConvNetType;
	string ErrorType;					 // Type of cost function or performance index used
	string optimizer;					 // Type of algorithm used
	bool momentum;						 // Indicates whether momentum is used or not
	bool isLastepoch;					 // Label for the last epoch
private:
	FC_Dictionary FC_Parameters;		 // Dictionary containing weights and biases of fully connected layers
	FC_Dictionary FC_Cache;				 // Dictionary containing temporaral values of internal activations of fully connected layers
	FC_Dictionary FC_Grades;			 // Dictionary containing gradients of weights and biases of fully connected layers 
	Matrix** D;							 // Dropout Matrices in fully connected layers
	layer* layers;						 // Array of layers holding the number of neurons at each fully connected layer and its activation
	int numOfLayers;					 // Number of fully connected layers

public:
	CNN(string TypeOfConvNet);
	void train();
	void test(Matrix& X_test, Matrix& Y_test, string path, string devOrtest, bool batchNorm, bool dropout, float* keep_prob);

private:
	void init_LeNet1();
	void train_LeNet1(); 
	void test_LeNet1(); //forward m examples through LeNet1 and return the predictions in Y_hat

private:
	Matrix FC_FeedForward(layer* layers, int L, bool batchNorm, bool dropout, float* keep_prob, string mode);
	void FC_BackProp(const Matrix& X, const Matrix& Y, const Matrix& Y_hat, float& alpha, layer* layers, int L, int iteration, int m, float lambda, bool batchNorm, bool dropout, float* keep_prob);
	void FC_CalGrads(const Matrix& X, const Matrix& Y, const Matrix& Y_hat, layer* layers, int L, float lambda, bool batchNorm, bool dropout, float* keep_prob);
	void FC_UpdateParameters(float& alpha, layer* layers, int L, int iteration, int m, bool batchNorm);


};
#endif // !CNN_HEADER

