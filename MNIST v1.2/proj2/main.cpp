#include <iostream>
#include "NeuralNetwork.h"
#include "dataset.h"
#include "dataset2D.h"
#include "Volume.h"
#include "ElasticDistortion.h"
#include <conio.h>
using namespace std;
int main()
{
    srand(time(NULL));

    clock_t START=clock();
    Volume X_2D(60000*4);
    Matrix Y(10,60000);
    Matrix X_test(784,10000);
    Matrix Y_test(10,10000);
    Matrix X_dev(784,10000);
    Matrix Y_dev(10,10000);

    char* dir1="F:\\GradProj 2\\dataset\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
    char* dir2="F:\\GradProj 2\\dataset\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";
    char* tdir1="F:\\GradProj 2\\dataset\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte";
    char* tdir2="F:\\GradProj 2\\dataset\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte";

    get_dataset_2D(X_2D,Y,dir1,dir2);
    get_dataset(X_test,Y_test,tdir1,tdir2);
    Matrix* X=enlarge1D(X_2D,Y,3);
    DELETE(X_2D);
    Shuffle(*X,Y);
    DevSet(*X,Y,X_dev,Y_dev);
    clock_t END=clock();

    X.Write("XTrain_240K.bin");
    Y.Write("YTrain_240K.bin");
    X_test.Write("XTest.bin");
    Y_test.Write("YTest.bin");
    X_dev.Write("XDev.bin");
    Y_dev.Write("Ydev.bin");

    cout<<"It took "<<(END-START)/CLOCKS_PER_SEC<<" to prepare the dataset."<<endl<<endl;
    cout<<"===>START TRAINING<==="<<endl<<endl;



    int     numOfLayers=4;
    layer*  layers = new layer[numOfLayers];
            layers[0].put(784, "");
            layers[1].put(600, "leakyRelu");
            layers[2].put(200, "leakyRelu");
            layers[3].put(10, "softmax");


    float   learingRate=.03;
    int     numOfEpochs=1;
    int     batchSize=1024;
    string  optimizer="Adam";
    int     numPrint=1;
	string  ErrorType="CrossEntropy";
	float   regularizationParameter=0;
	bool    batchNorm=true;
	bool    dropout=true;
	float*  keep_prob = new float[numOfLayers];
            keep_prob[0]=1;
            keep_prob[1]=.5;
            keep_prob[2]=.5;
            keep_prob[3]=1;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    NeuralNetwork NN(layers, numOfLayers);
    NN.train(*X, Y, X_dev, Y_dev, learingRate, numOfEpochs, batchSize, optimizer, numPrint, ErrorType, regularizationParameter, batchNorm, dropout, keep_prob);
    NN.test(X_test, Y_test, "C:\\Users\\delta\\Desktop\\GradProj 2", batchNorm, dropout, keep_prob);
    bool CONTINUE=true;
    int EPOCHS=1;
    int i=numOfEpochs;
    while(CONTINUE)
    {
        cout<<"epoch : "<<++i<<endl;
        NN.continueTrain(*X, Y, X_dev, Y_dev, learingRate, EPOCHS, batchSize, optimizer, numPrint, ErrorType, regularizationParameter, batchNorm, dropout, keep_prob);
        NN.test(X_test, Y_test, "C:\\Users\\delta\\Desktop\\GradProj 2", batchNorm, dropout, keep_prob);
        if(i==50)
            CONTINUE=false;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    regularizationParameter=0.001;
    keep_prob[0]=1;
    keep_prob[1]=.5;
    keep_prob[2]=.5;
    keep_prob[3]=1;
    NeuralNetwork NN2(layers, numOfLayers);
    NN2.train(*X, Y, X_dev, Y_dev, learingRate, numOfEpochs, batchSize, optimizer, numPrint, ErrorType, regularizationParameter, batchNorm, dropout, keep_prob);
    NN2.test(X_test, Y_test, "C:\\Users\\delta\\Desktop\\GradProj 2", batchNorm, dropout, keep_prob);
    CONTINUE=true;
    EPOCHS=1;
    i=numOfEpochs;
    while(CONTINUE)
    {
        cout<<"epoch : "<<++i<<endl;
        NN2.continueTrain(*X, Y, X_dev, Y_dev, learingRate, EPOCHS, batchSize, optimizer, numPrint, ErrorType, regularizationParameter, batchNorm, dropout, keep_prob);
        NN2.test(X_test, Y_test, "C:\\Users\\delta\\Desktop\\GradProj 2", batchNorm, dropout, keep_prob);
        if(i==50)
            CONTINUE=false;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    return 0;
}

/*TIPS*/
/*
1- batch size is critical when using batchNorm, the larger the more accurate
2- start with lambda=.01 for L2 regularization
3- start with learning rate .1 and go down
4- dropout needs larger learning rates to be accurate
5- nans apear due to unstability in architecture or large learing rate, you may replace relu and get something limited
*/

/*TODO*/
/*
shuffle every epoch
*/


//5:50
