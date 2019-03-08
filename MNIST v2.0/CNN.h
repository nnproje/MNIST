#pragma once
#ifndef CNN_HEADER
#define CNN_HEADER
#include "Dictionary.h"
#include "Matrix.h"
#include "Volume.h"
#include "NN_Tools.h"
#include "Activations.h"
#include "ConvFeedForward.h"
#include "ConvBackProb.h"
typedef Dictionary<string, matrix<float> > FC_Dictionary;
typedef Dictionary<string, Volume > Conv_Dictionary;
typedef matrix<float> Matrix;
//*************************************************************************************************/
//4D discription for a 4D element A(m,nc,nh,nw):
//1- A is a vector of volumes that has a size m
//2- A[i] is a volume with nc channels, it represents the activations of some layer for the ith example
//3- A[i][j] is a pointer to the jth Matrix with nh hight and nw width in the ith example
//4- A[i][0] represents the first channel in the volume, take them from top to down
//input:
//Aprev(m,nc_prev,nh_prev,nw_prev) , filters(numOfFilters,nc,f,f) , b(numOfFilters,nh,nw)
//output:
//A(m,numOfFilters,nh,nw)
//*************************************************************************************************/
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

