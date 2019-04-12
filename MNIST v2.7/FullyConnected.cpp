#include "NeuralNetwork.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::init_FC()
{
        Matrix*  MatPtr=nullptr;
        D=new Matrix*[numOfLayers];
        Matrix** Mw = new Matrix*[numOfLayers - 1];
        Matrix** Mb = new Matrix*[numOfLayers - 1];

        for (int i = 0; i<numOfLayers - 1; i++)  // L-1 = number of hidden layers + output layer
        {
            Mw[i] = new Matrix(layers[i + 1].neurons, layers[i].neurons, Random); // Mw[0] holds W1 and so on
            MatPtr = Mw[i];
            Mw[i] = Mw[i]->div(float(RAND_MAX));
			delete MatPtr;

            /*To make the standard deviation of weights = 1 and mean = 0*/
            if(Mw[i]->Rows() != 1 || Mw[i]->Columns() != 1) // Don't calculate if dimensions are 1x1
            {
                float Wmean = Mw[i]->sumall() / (Mw[i]->Rows() * Mw[i]->Columns());
                MatPtr=Mw[i];
                Mw[i]=Mw[i]->sub(Wmean);
				delete MatPtr;

                float Wstd = sqrt((Mw[i]->square()).sumall() / (Mw[i]->Rows() * Mw[i]->Columns()));
                MatPtr=Mw[i];
                Mw[i]=Mw[i]->div(Wstd);
				delete MatPtr;
            }

            if (layers[i+1].activation == SIGMOID)
            {
                MatPtr=Mw[i];
                Mw[i]=Mw[i]->mul(sqrt(2/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == SOFTMAX)
            {
                MatPtr=Mw[i];
                Mw[i]=Mw[i]->mul(sqrt(2/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == TANH)
            {
                MatPtr=Mw[i];
                Mw[i]=Mw[i]->mul(sqrt(1/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == RELU)
            {
                MatPtr=Mw[i];
                Mw[i]=Mw[i]->mul(sqrt(2/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == LEAKYRELU)
            {
                MatPtr=Mw[i];
                Mw[i]=Mw[i]->mul(sqrt(2/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == SATLINEAR)
            {
                MatPtr=Mw[i];
                Mw[i]=Mw[i]->mul(sqrt(1/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == LINEAR)
            {
                MatPtr=Mw[i];
                Mw[i]=Mw[i]->mul(sqrt(2/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == SATLINEAR2)
            {
                MatPtr=Mw[i];
                Mw[i]=Mw[i]->mul(sqrt(1/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == SATLINEAR3)
            {
                MatPtr=Mw[i];
                Mw[i]=Mw[i]->mul(sqrt(1/layers[i].neurons));
				delete MatPtr;
            }
            FC_Parameters.put(CharGen("W", i + 1), Mw[i]);


            Mb[i] = new Matrix(layers[i + 1].neurons, 1, Random);
            MatPtr = Mb[i];
            Mb[i] = Mb[i]->div(float(RAND_MAX));
			delete MatPtr;

            /*To make the standard deviation of biases = 1 and mean = 0*/
            if(Mb[i]->Rows() != 1 || Mb[i]->Columns() != 1)         // Don't calculate if dimensions are 1x1
            {
                float bmean = Mb[i]->sumall() / (Mb[i]->Rows() * Mb[i]->Columns());
                MatPtr=Mb[i];
                Mb[i]=Mb[i]->sub(bmean);
				delete MatPtr;

                float bstd = sqrt((Mb[i]->square()).sumall() / (Mb[i]->Rows() * Mb[i]->Columns()));
                MatPtr=Mb[i];
                Mb[i]=Mb[i]->div(bstd);
				delete MatPtr;
            }

           if (layers[i+1].activation == SIGMOID)
            {
                MatPtr=Mb[i];
                Mb[i]=Mb[i]->mul(sqrt(2/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == SOFTMAX)
            {
                MatPtr=Mb[i];
                Mb[i]=Mb[i]->mul(sqrt(2/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == TANH)
            {
                MatPtr=Mb[i];
                Mb[i]=Mb[i]->mul(sqrt(1/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == RELU)
            {
                MatPtr=Mb[i];
                Mb[i]=Mb[i]->mul(sqrt(2/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == LEAKYRELU)
            {
                MatPtr=Mb[i];
                Mb[i]=Mb[i]->mul(sqrt(2/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == SATLINEAR)
            {
                MatPtr=Mb[i];
                Mb[i]=Mb[i]->mul(1/layers[i].neurons);
				delete MatPtr;
            }
            else if (layers[i+1].activation == LINEAR)
            {
                MatPtr=Mb[i];
                Mb[i]=Mb[i]->mul(sqrt(2/layers[i].neurons));
				delete MatPtr;
            }
            else if (layers[i+1].activation == SATLINEAR2)
            {
                MatPtr=Mb[i];
                Mb[i]=Mb[i]->mul(1/layers[i].neurons);
				delete MatPtr;
            }
            else if (layers[i+1].activation == SATLINEAR3)
            {
                MatPtr=Mb[i];
                Mb[i]=Mb[i]->mul(1/layers[i].neurons);
				delete MatPtr;
            }
            FC_Parameters.put(CharGen("b", i + 1), Mb[i]);
        }

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::train_FC(Matrix* X,Matrix* Y,Matrix* X_div,Matrix* Y_div,float alpha, int numOfEpochs, int minibatchSize, Optimizer op,int Numprint, ErrorType ET,float lambda,bool batchNorm,bool dropout,float* keep_prob)
{
    ErrType = ET;
    optimizer = op;
    isLastepoch = false;
    momentum=true;
    int t = 0; //Counter used for Adam optimizer

    /*BatchNorm Initialization*/
	if (batchNorm)
	{
		Matrix** g1 = new Matrix*[numOfLayers - 1];   //gamma1
		Matrix** g2 = new Matrix*[numOfLayers - 1];   //gamma2
		for (int ii = 0; ii < numOfLayers - 1; ii++)
		{
			g1[ii] = new Matrix(layers[ii + 1].neurons, 1, 1);
			FC_Parameters.put(CharGen("g1", ii + 1), g1[ii]);
			g2[ii] = new Matrix(layers[ii + 1].neurons, 1);
			FC_Parameters.put(CharGen("g2", ii + 1), g2[ii]);
		}
	}
        /*Intialization For batch Norm at test time*/
        Matrix** running_mean = new Matrix*[numOfLayers - 1];     //mean of z for each layer
        Matrix** running_var = new Matrix*[numOfLayers- 1];       //standard deviation of z for each layer
        for(int ii=0;ii<numOfLayers-1;ii++)
        {
            running_mean[ii]=new Matrix (layers[ii+1].neurons,1);
            FC_Parameters.put(CharGen("rm",ii+1), running_mean[ii]);
            running_var[ii]=new Matrix (layers[ii+1].neurons,1);
            FC_Parameters.put(CharGen("rv",ii+1), running_var[ii]);
        }
        /*End of Intialization For batch Norm at test time*/

    /*End Of BatchNorm Initialization*/

    /*ADAM INITIALIZATION*/
	Matrix** Msdw = new Matrix*[numOfLayers - 1];
	Matrix** Mvdw = new Matrix*[numOfLayers - 1];
	Matrix** Msdb = new Matrix*[numOfLayers - 1];
	Matrix** Mvdb = new Matrix*[numOfLayers - 1];
	/*For BatchNorm*/
	Matrix** sdg1 = new Matrix*[numOfLayers - 1];
	Matrix** vdg1 = new Matrix*[numOfLayers - 1];
	Matrix** sdg2 = new Matrix*[numOfLayers - 1];
	Matrix** vdg2 = new Matrix*[numOfLayers - 1];
    if (optimizer == ADAM)
    {
        for (int i = 0; i < numOfLayers - 1; i++)   // L-1 = number of hidden layers + output layer
        {
            Msdw[i]=new Matrix(layers[i+1].neurons, layers[i].neurons);
			FC_ADAM.put(CharGen("Sdw", i + 1), Msdw[i]);

            Mvdw[i]=new Matrix(layers[i+1].neurons, layers[i].neurons);
			FC_ADAM.put(CharGen("Vdw", i + 1), Mvdw[i]);

            Msdb[i]=new Matrix(layers[i+1].neurons, 1);
			FC_ADAM.put(CharGen("Sdb", i + 1), Msdb[i]);

            Mvdb[i]=new Matrix(layers[i+1].neurons, 1);
			FC_ADAM.put(CharGen("Vdb", i + 1), Mvdb[i]);

            if(batchNorm)
            {
                sdg1[i]=new Matrix(layers[i+1].neurons,1);
				FC_ADAM.put(CharGen("sg1", i + 1), sdg1[i]);
                vdg1[i]=new Matrix(layers[i+1].neurons,1);
				FC_ADAM.put(CharGen("vg1", i + 1), vdg1[i]);

                sdg2[i]=new Matrix(layers[i+1].neurons,1);
				FC_ADAM.put(CharGen("sg2", i + 1), sdg2[i]);
                vdg2[i]=new Matrix(layers[i+1].neurons,1);
				FC_ADAM.put(CharGen("vg2", i + 1), vdg2[i]);
            }
        }

    }
    /*END OF ADAM INITIALIZATION*/


    /*BEGINNING OF EPOCHS ITERATIONS*/
    for (int i = 0; i<numOfEpochs; i++)
    {
        clock_t start = clock();

        /*Iterations on mini batches*/
        int m = X->Columns();
        int numOfMiniBatches = m / minibatchSize;
        int LastBatchSize=m-minibatchSize*numOfMiniBatches;
        int j;
        if(i==numOfEpochs-1)
            isLastepoch=true;

        for (j = 0; j<numOfMiniBatches; j++)
        {
            Matrix* cur_X = X->SubMat(0, j*minibatchSize, X->Rows() - 1, ((j + 1)*(minibatchSize)-1));
            Matrix* cur_Y = Y->SubMat(0, j*minibatchSize, Y->Rows() - 1, ((j + 1)*(minibatchSize)-1));
            FC_Cache.put("A0",cur_X);
            Matrix* Y_hat = FC_FeedForward(batchNorm,dropout,keep_prob,TRAIN);
            FC_BackProp(cur_X,cur_Y,Y_hat,alpha,t,lambda,batchNorm,dropout,keep_prob);
			t++;
			delete cur_Y;
			FC_Cache.DeleteThenClear();
        }

        if(LastBatchSize!=0)
        {
			Matrix* cur_X = X->SubMat(0, j*minibatchSize, X->Rows() - 1, X->Columns() - 1);
			Matrix* cur_Y = Y->SubMat(0, j*minibatchSize, Y->Rows() - 1, Y->Columns() - 1);
			FC_Cache.put("A0",cur_X);
            Matrix* Y_hat = FC_FeedForward(batchNorm,dropout,keep_prob,TRAIN);
            FC_BackProp(cur_X,cur_Y,Y_hat,alpha,t,lambda,batchNorm,dropout,keep_prob);
			t++;
			delete cur_Y;
            FC_Cache.DeleteThenClear();
            cout<<"Mini batch no:"<<j<<endl;
        }

        clock_t end = clock();
        double duration_sec = double(end - start) / CLOCKS_PER_SEC;
        cout << "epoch No." << i << " ended" << endl;
        cout << "Time = " << duration_sec << endl;
    }
    /*END OF EPOCHS ITERATIONS*/

     /*CLASSIFICATION AND ACCURACY TESTING OF ACTIVATIONS*/
	 FC_Cache.put("A0", X_div);
	 Matrix* Y_hat = FC_FeedForward(batchNorm, false, keep_prob, TEST);
	 AccuracyTest(Y_div, Y_hat, "dev");
	 FC_Cache.DeleteThenClear();
	 /*END OF CLASSIFICATION AND ACCURACY TESTING OF ACTIVATIONS*/

	 if (optimizer == ADAM)
		 FC_Grades.DeleteThenClear();
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* NeuralNetwork::test_FC( Matrix* X_test, Matrix* Y_test,bool batchNorm, bool dropout, float* keep_prob) //forward m examples and return the predictions in Y_hat
{
    int m=X_test->Columns();
	FC_Cache.put("A0", X_test);
	Matrix* Y_hat =FC_FeedForward(batchNorm,dropout,keep_prob,TEST);
	return Y_hat;
}
