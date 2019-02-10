#pragma once
#ifndef NEURALNETWORK_H_INCLUDED
#define NEURALNETWORK_H_INCLUDED
#include "Matrix.h"
#include "Dictionary.h"
#include "layer.h"
typedef Dictionary< string, matrix<float> > dictionary;
typedef matrix<float> Matrix;

class NeuralNetwork
{
private:
     layer* layers;               // Array of layers holding the number of neurons at each layer and its activation
     dictionary parameters;       // Dictionary containing weights and biases of the network
     dictionary cache;            // Dictionary containing temporary internal activations of layers
     dictionary grades;           // Dictionary containing gradients of weights and biases of the network
     string ErrorType;            // Type of cost function or performance index used
     string optimizer;            // Type of algorithm used
	 int numOfLayers;             // Number of layers
     bool momentum;               // Indicates whether momentum is used or not
     Matrix** D;                  // Dropout Matrices
     float maxErr;                // Maximum err between labels and predictions
public:
    NeuralNetwork(layer* mylayers,int L);                                                                    // Constructor that initializes the weighs and biases of the network randomly based on the architecture of the network
    void test(Matrix X_test, Matrix Y_test, string path,bool batchNorm, bool dropout,float* keep_prob);      // Function that outputs the input training set X, their associated targets Y and the final activations Y_hat into a text file
    void print();                                                                                            // Prints all parameters of the network
    void train(const Matrix& X, Matrix& Y, Matrix& X_dev, Matrix& Y_dev, float alpha, int numOfEpochs, int minibatchSize, string optimizer1,int Numprint, string ET,float lambda,bool batchNorm, bool dropout,float* keep_prob);
    // Function that trains the network based on the following arguments:
    // X: input training set
    // Y: target associated with the input training set
    // alpha: learning rate or damping ratio in case of LM optimizer
    // numOfEpochs: maximum number of iterations
    // minibatchsize: size of mini-batch (don't care if LM algorithm)
    // optimizer: the algorithm used to train the network. It's either "GradientDescent", "Adam" or "LM"
    // Numprint: the number of iterations after which the cost is outputted on the screen
    // ET: the cost function or performance index. It's either "CrossEntropy" or "SquareErr"


private:
    // Feed forward
    Matrix feedforward(const Matrix& x, layer* layers,int L,bool batchNorm, bool dropout,float* keep_prob);
    // Back propagation
    void calGrads(const Matrix& X, const Matrix& Y, const Matrix& Y_hat, layer* layers, int L,float lambda,bool batchNorm, bool dropout,float* keep_prob);
    void updateParameters(float& alpha, layer* layers, int L, int iteration, Matrix& Q, Matrix& g, int m,bool batchNorm);
    void BackProp(const Matrix& X, const Matrix& Y, const Matrix& Y_hat, float& alpha, layer* layers, int L, int iteration, Matrix& Q, Matrix& g, int m,float lambda,bool batchNorm, bool dropout,float* keep_prob);
    // Cost
    float CostFunc(const Matrix& y,Matrix& yhat,float lambda,int L);
    Matrix costMul(Matrix Y, Matrix Y_hat);
    // Classify
    Matrix classify(Matrix Y_hat);
    void AccuracyTest(Matrix& Y,Matrix& Y_hat, const Matrix& X, layer*layers, int L, string devOrtest);
    // Error
    float AbsErr(Matrix* Y_hat, Matrix* Y);
    float numOfErrs(Matrix* Y_hat, Matrix* Y);
    // Store data
    void storedata(Matrix X, Matrix Y, Matrix Yhat, string path);
};
#endif // NEURALNETWORK_H_INCLUDED
