#include <iostream>
#include "NeuralNetwork.h"
#include "dataset.h"
using namespace std;
int main()
{
    srand(time(NULL));

    Matrix X(784,60000);
    Matrix Y(10,60000);
    Matrix X_test(784,10000);
    Matrix Y_test(10,10000);
    Matrix X_dev(784,10000);
    Matrix Y_dev(10,10000);

    char* dir1="C:\\Users\\delta\\Desktop\\GradProj 2\\dataset\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
    char* dir2="C:\\Users\\delta\\Desktop\\GradProj 2\\dataset\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";

    char* tdir1="C:\\Users\\delta\\Desktop\\GradProj 2\\dataset\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte";
    char* tdir2="C:\\Users\\delta\\Desktop\\GradProj 2\\dataset\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte";

    get_dataset(X,Y,dir1,dir2);
    get_dataset(X_test,Y_test,tdir1,tdir2);
    normalize(X,X_test,Y,Y_test);
    Shuffle(X,Y);
    DevSet(X,Y,X_dev,Y_dev);




    int     numOfLayers=4;
    layer*  layers = new layer[numOfLayers];
            layers[0].put(784, "");
            layers[1].put(110, "leakyRelu");
            layers[2].put(130, "leakyRelu");
            layers[3].put(10, "softmax");


    float   learingRate=.005;
    int     numOfEpochs=60;
    int     batchSize=2048;
    string  optimizer="Adam";
    int     numPrint=1;
	string  ErrorType="CrossEntropy";
	float   regularizationParameter=0;
	bool    batchNorm=true;
	bool    dropout=true;
	float*  keep_prob = new float[numOfLayers];
            keep_prob[0]=1;
            keep_prob[1]=.7;
            keep_prob[2]=.6;
            keep_prob[3]=1;

    NeuralNetwork NN(layers, numOfLayers);
    NN.train(X, Y, X_dev, Y_dev, learingRate, numOfEpochs, batchSize, optimizer, numPrint, ErrorType, regularizationParameter, batchNorm, dropout, keep_prob);
    NN.test(X_test, Y_test, "C:\\Users\\delta\\Desktop\\GradProj 2", batchNorm, dropout, keep_prob);

    return 0;
}

/*TIPS*/
/*
1- batch size is critical when using batchNorm, the larger the more accurate
2- start with lambda=.01 for L2 regularization
3- start with learning rate .3 and go down
4- dropout needs larger learning rates to be accurate
5- nans apear due to unstability in architecture or large learing rate, you may replace relu and get something limited
*/

/*TODO*/
/*
1-normalizing inputs (then use larger alpha) ==>gives nan cost ==>eps is added to softmax() and costMul() ==>[-1,1] range is not compatible with softmax ==>as a result try SquareErr with satLinear and alpha starting with .03
*/

