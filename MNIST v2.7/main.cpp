#include <iostream>
#include "NN_Tools.h"
#include "Matrix.h"
#include "Volume.h"
#include "VectVolume.h"
#include "Dictionary.h"
#include "Activations.h"
#include "DataSet.h"
#include "NeuralNetwork.h"
#include "ConvBackProb.h"
#define TRAIN_EXAMPLES 60000
#define DEV_EXAMPLES   10000
#define TEST_EXAMPLES  10000

using namespace std;
int main()
{
	srand(time(NULL));
	clock_t START = clock();

	Matrix* X = new Matrix(784, TRAIN_EXAMPLES);
	Matrix* Y = new Matrix(10, TRAIN_EXAMPLES);
	Matrix* X_test = new Matrix(784, DEV_EXAMPLES);
	Matrix* Y_test = new Matrix(10, DEV_EXAMPLES);
	Matrix* X_dev = new Matrix(784, TEST_EXAMPLES);
	Matrix* Y_dev = new Matrix(10, TEST_EXAMPLES);

    const char* dir1  = "F:\\GradProj 2\\dataset\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
	const char* dir2  = "F:\\GradProj 2\\dataset\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";
	const char* tdir1 = "F:\\GradProj 2\\dataset\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte";
	const char* tdir2 = "F:\\GradProj 2\\dataset\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte";


	get_dataset(X, Y, dir1, dir2, TRAIN_EXAMPLES);
	get_dataset(X_test, Y_test, tdir1, tdir2, TEST_EXAMPLES);
	Shuffle(X, Y);
	DevSet(X, Y, X_dev, Y_dev, DEV_EXAMPLES);
	clock_t END = clock();

	cout << "It took " << (END - START) / CLOCKS_PER_SEC << " seconds to prepare the dataset." << endl << endl;
	cout << "===>START TRAINING<===" << endl << endl;

    int     numOfLayers=4;
    layer*  layers = new layer[numOfLayers];
            layers[0].put(784, NONE);
			layers[1].put(400, LEAKYRELU);
            layers[2].put(200, LEAKYRELU);
            layers[3].put(10, SOFTMAX);

	ErrorType ErrType = CROSS_ENTROPY;
	Optimizer optimizer = ADAM;
    float   learingRate=.01;
    int     numOfEpochs=4;
    int     batchSize=256;
    int     numPrint=1;
	float   regularizationParameter=0;
	bool    batchNorm=false;
	bool    dropout=false;
	float*  keep_prob = new float[numOfLayers];
            keep_prob[0]=1;
            keep_prob[1]=.5;
            keep_prob[2]=.5;
            keep_prob[3]=1;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	NeuralNetwork LeNet(LENET1);
	LeNet.train(X, Y, X_dev, Y_dev, learingRate, numOfEpochs, batchSize, optimizer, numPrint, ErrType, 0, batchNorm, dropout, keep_prob);
	LeNet.test(X_test, Y_test, "anything", TEST, batchNorm, dropout, keep_prob);

	_getche();
	return 0;
}
