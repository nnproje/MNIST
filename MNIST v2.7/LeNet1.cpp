#include "NeuralNetwork.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::init_LeNet1()
{
	/*Temporal variables*/
	Matrix* Matptr = nullptr;
	Matrix* Matptr1 = nullptr;
	Matrix* Matptr2 = nullptr;
	Matrix* Matptr3 = nullptr;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*First convolution Layer*/
	VectVolume WC1(4, 1, 5, 5, Random_Limited);
	Matrix* bC1 = new Matrix(4, 1, 0);

	//Normalizing weights & using Xavier initialization
	float W1mean = 0; float W1std = 0;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 1; j++)
		{
			W1mean += WC1[i][j]->sumall();
		}
	W1mean /= (4 * 1 * 5 * 5);

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			//WC1[i][j]=WC1[i][j]-W1mean;
			Matptr = WC1[i][j];
			WC1[i][j] = Matptr->sub(W1mean);
			delete Matptr;
			Matptr = nullptr;

			//W1std+=(WC1[i][j].square()).sumall();
			Matptr = WC1[i][j]->SQUARE();
			W1std = W1std + Matptr->sumall();
			delete Matptr;
			Matptr = nullptr;
		}
	}
	W1std = sqrt(W1std / (4 * 1 * 5 * 5));
	float XavierValue1 = sqrt(2.0 / (4 * 1 * 5 * 5)) / W1std;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 1; j++)
		{
			//Xavier initialization for relu (*sqrt(2/fin)),try using fout or fin+fout instead of fin...For tanh use 1 instead of 2
			//WC1[i][j] = WC1[i][j] * XavierValue1;
			Matptr = WC1[i][j];
			WC1[i][j] = Matptr->mul(XavierValue1);
			delete Matptr;
			Matptr = nullptr;
		}

	Conv_Weights.put("WC1", WC1);
	Conv_biases.put("bC1", bC1);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*Second convolution Layer*/
	VectVolume WC2(12, 4, 5, 5, Random_Limited);
	Matrix* bC2 = new Matrix(12, 1, 0);

	float W2mean = 0; float W2std = 0;
	for (int i = 0; i < 12; i++)
		for (int j = 0; j < 4; j++)
		{
			W2mean += WC2[i][j]->sumall();
		}
	W2mean /= (12 * 4 * 5 * 5);
	for (int i = 0; i < 12; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			//WC2[i][j]=WC2[i][j]-W2mean;
			Matptr = WC2[i][j];
			WC2[i][j] = Matptr->sub(W2mean);
			delete Matptr;
			Matptr = nullptr;

			//W2std+=(WC2[i][j].square()).sumall();
			Matptr = WC2[i][j]->SQUARE();
			W2std = W2std + Matptr->sumall();
			delete Matptr;
			Matptr = nullptr;
		}
	}
	W2std = sqrt(W2std / (12 * 4 * 5 * 5));
	float XavierValue2 = sqrt(2.0 / (12 * 4 * 5 * 5)) / W2std;
	for (int i = 0; i < 12; i++)
		for (int j = 0; j < 4; j++)
		{
			//Xavier initialization for relu (*sqrt(2/fin)),try using fout or fin+fout instead of fin...For tanh use 1 instead of 2
			//WC2[i][j] = WC2[i][j] * XavierValue2;
			Matptr = WC2[i][j];
			WC2[i][j] = Matptr->mul(XavierValue2);
			delete Matptr;
			Matptr = nullptr;
		}

	Conv_Weights.put("WC2", WC2);
	Conv_biases.put("bC2", bC2);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*Fully Connected Layers*/
	numOfLayers = 2;
	layers = new layer[numOfLayers];
	layers[0].put(192, RELU);
	layers[1].put(10, SOFTMAX);

	/*First FC Layer*/
	Matrix* W1 = new  Matrix(10, 192, Random_Limited);
	Matrix* b1 = new Matrix(10, 1, 0);

	//Normalizing weights & using Xavier initialization
	float Wmean = W1->sumall() / (W1->Rows() * W1->Columns());

	//W1 = W1 - Wmean;
	Matptr = W1;
	W1 = Matptr->sub(Wmean);
	delete Matptr;
	Matptr = nullptr;

	//Wstd = sqrt(((W1.square()).sumall()) / (W1.Rows() * W1.Columns()));
	Matptr = W1->SQUARE();
	float Wstd = sqrt((Matptr->sumall()) / (W1->Rows() * W1->Columns()));
	delete Matptr;
	Matptr = nullptr;


	//W1 = W1 / Wstd;
	Matptr = W1;
	W1 = Matptr->div(Wstd);
	delete Matptr;
	Matptr = nullptr;

	//W1 = W1  * sqrt(2.0/layers[0].neurons);
	Matptr = W1;
	W1 = Matptr->mul(sqrt(2.0 / layers[0].neurons));
	delete Matptr;
	Matptr = nullptr;

	//Xavier initialization for relu (*sqrt(2/fin)),try using fout or fin+fout instead of fin...For tanh use 1 instead of 2
	FC_Parameters.put("W1", W1);
	FC_Parameters.put("b1", b1);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::train_LeNet1(Matrix* X, Matrix* Y, Matrix* X_div, Matrix* Y_div, float alpha, int numOfEpochs, int minibatchSize, Optimizer optimizer1, int Numprint, ErrorType ET, float lambda, bool batchNorm, bool dropout, float* keep_prob)
{
	ErrType = ET;
	optimizer = optimizer1;
	isLastepoch = false;
	momentum = true;
	int t = 0;


	/*ADAM INITIALIZATION*/
	if (optimizer == ADAM)
	{
		/* ConvLayer 1 */
		VectVolume SdwC1(4, 1, 5, 5);
		ADAM_dWC.put("SdwC1", SdwC1);
		VectVolume VdwC1(4, 1, 5, 5);
		ADAM_dWC.put("VdwC1", VdwC1);
		Matrix* SdbC1 = new Matrix(4, 1);
		ADAM_dbC.put("SdbC1", SdbC1);
		Matrix* VdbC1 = new Matrix(4, 1);
		ADAM_dbC.put("VdbC1", VdbC1);
		/* END */


		/* ConvLayer 2 */
		VectVolume SdwC2(12, 4, 5, 5);
		ADAM_dWC.put("SdwC2", SdwC2);
		VectVolume VdwC2(12, 4, 5, 5);
		ADAM_dWC.put("VdwC2", VdwC2);
		Matrix* SdbC2 = new Matrix(12, 1);
		ADAM_dbC.put("SdbC2", SdbC2);
		Matrix* VdbC2 = new Matrix(12, 1);
		ADAM_dbC.put("VdbC2", VdbC2);
		/* END */

		/* FC_Layer */
		Matrix* Sdw1 = new Matrix(10, 192);
		FC_ADAM.put("Sdw1", Sdw1);
		Matrix* Vdw1 = new Matrix(10, 192);
		FC_ADAM.put("Vdw1", Vdw1);
		Matrix* Sdb1 = new Matrix(10, 1);
		FC_ADAM.put("Sdb1", Sdb1);
		Matrix* Vdb1 = new Matrix(10, 1);
		FC_ADAM.put("Vdb1", Vdb1);
		/* END */

	}
	/*END OF ADAM INITIALIZATION*/



	/*BEGINNING OF EPOCHS ITERATIONS*/
	for (int i = 0; i < numOfEpochs; i++)
	{
		clock_t start = clock();
		/*Iterations on mini batches*/
		int m = X->Columns();
		int numOfMiniBatches = m / minibatchSize;
		int LastBatchSize = m - minibatchSize * numOfMiniBatches;
		int j;
		if (i == numOfEpochs - 1)
			isLastepoch = true;

		for (j = 0; j < numOfMiniBatches; j++)
		{
			clock_t strt = clock();
			Matrix* cur_X = X->SubMat(0, j*minibatchSize, X->Rows() - 1, ((j + 1)*(minibatchSize)-1));
			Matrix* cur_Y = Y->SubMat(0, j*minibatchSize, Y->Rows() - 1, ((j + 1)*(minibatchSize)-1));
			int m = cur_X->Columns();

			VectVolume AC0 = to_VectorOfVolume(cur_X, 28, 28, 1, m);
			Conv_Cache.put("AC0", AC0);
			convLayer(1, 1, RELU);           //stride=1, A_index=1,W_index=1
			poolLayer(2, 2, AVG, 1);         //stride=2, f=2 ,mode ="avg",A_index=1
			convLayer(1, 2, RELU);           //stride=1 ,A_index=2,W_index=2
			poolLayer(2, 2, AVG, 2);         //f=5 ,mode ="avg",A_index=2
			VectVolume ACP2 = Conv_Cache["ACP2"];
			Matrix* A0 = to_FC(ACP2);
			FC_Cache.put("A0", A0);
			Matrix* Y_hat = FC_FeedForward(0, 0, keep_prob, TRAIN);
			///////////////////////////////////////////////////////
			FC_BackProp(A0, cur_Y, Y_hat, alpha, t, 0, 0, 0, keep_prob);
			Matrix* dA0 = FC_Grades["dA0"];
			VectVolume dACP2 = to_VectorOfVolume(dA0, ACP2[0][0]->Rows(), ACP2[0][0]->Columns(), ACP2[0].size(), m);
			Conv_Grades.put("dACP2", dACP2);
			pool_backward(2, 2, AVG, 2);			  //f=2,stride=2,mode ="avg",A_index=2
			ConvBackwardOptimized(1, 2, RELU);        //stride=1 ,A_index=2,W_index=2
			pool_backward(2, 2, AVG, 1);			  //f=2,stride=2,mode ="avg",A_index=1
			ConvBackwardOptimized(1, 1, RELU);        //stride=1 ,A_index=1,W_index=1
			updateparameters(alpha, t, 2);
			updateparameters(alpha, t, 1);
			t++;
			cur_X->DELETE();
			cur_Y->DELETE();
			FC_Cache.DeleteThenClear();
			Conv_Cache.DeleteThenClearObj();
			FC_Grades.DeleteThenClear();
			Conv_dbiases.DeleteThenClear();
			Conv_Grades.DeleteThenClearObj();
			cout << endl << "Minibatch No." << j << " ended , Time = " << double(clock() - start) / CLOCKS_PER_SEC << endl;
		}
		if (LastBatchSize != 0)
		{
			Matrix* cur_X = X->SubMat(0, j*minibatchSize, X->Rows() - 1, X->Columns() - 1);
			Matrix* cur_Y = Y->SubMat(0, j*minibatchSize, Y->Rows() - 1, Y->Columns() - 1);
			int m = cur_X->Columns();
			VectVolume AC0 = to_VectorOfVolume(cur_X, 28, 28, 1, m);
			Conv_Cache.put("AC0", AC0);
			convLayer(1, 1, RELU);             //stride=1 ,A_index=1,W_index=1
			poolLayer(2, 2, AVG, 1);           //f=5 ,mode ="avg",A_index=1
			convLayer(1, 2, RELU);             //stride=1 ,A_index=3,W_index=2
			poolLayer(2, 2, AVG, 2);           //f=5 ,mode ="average",A_index=2
			VectVolume ACP2 = Conv_Cache["ACP2"];
			Matrix* A0 = to_FC(ACP2);
			FC_Cache.put("A0", A0);
			Matrix* Y_hat = FC_FeedForward(0, 0, keep_prob, TRAIN);
			///////////////////////////////////////////////////////
			FC_BackProp(A0, cur_Y, Y_hat, alpha, t, 0, 0, 0, keep_prob);
			Matrix* dA0 = FC_Grades["dA0"];
			VectVolume dACP2 = to_VectorOfVolume(dA0, ACP2[0][0]->Rows(), ACP2[0][0]->Columns(), ACP2[0].size(), m);
			Conv_Grades.put("dACP2", dACP2);
			pool_backward(2, 2, AVG, 2);               //f=2,stride=2,mode ="avg",A_index=2
			ConvBackwardOptimized(1, 2, RELU);         //stride=1 ,A_index=2,W_index=2
			pool_backward(2, 2, AVG, 1);			   //f=2,stride=2,mode ="avg",A_index=1
			ConvBackwardOptimized(1, 1, RELU);         //stride=1 ,A_index=1,W_index=1
			updateparameters(alpha, t, 2);
			updateparameters(alpha, t, 1);
			t++;

			cur_X->DELETE();
			cur_Y->DELETE();
			FC_Cache.DeleteThenClear();
			Conv_Cache.DeleteThenClearObj();
			FC_Grades.DeleteThenClear();

			Conv_dbiases.DeleteThenClear();
			Conv_Grades.DeleteThenClearObj();
		}
		clock_t end = clock();
		double duration_sec = double(end - start) / CLOCKS_PER_SEC;
		cout << "Epoch no. " << i << "in Time = " << duration_sec << endl;
	}
	/* END OF EPOCHS ITERATIONS */


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* NeuralNetwork::test_LeNet1(Matrix* X_test, Matrix* Y_test, bool batchNorm, bool dropout, float* keep_prob) //forward m examples and return the predictions in Y_hat
{
	Matrix * Y_hat = nullptr;
	int m = X_test->Columns();
	VectVolume AC0 = to_VectorOfVolume(X_test, 28, 28, 1, m);
	Conv_Cache.put("AC0", AC0);
	convLayer(1, 1, RELU);           //stride=1 ,A_index=1,W_index=1
	poolLayer(2, 2, AVG, 1);         //f=5 ,mode ="avg",A_index=1
	convLayer(1, 2, RELU);           //stride=1 ,A_index=3,W_index=2
	poolLayer(2, 2, AVG, 2);         //f=5 ,mode ="avg",A_index=2
	VectVolume ACP2 = Conv_Cache["ACP2"];
	Matrix* A0 = to_FC(ACP2);
	FC_Cache.put("A0", A0);
	Y_hat = FC_FeedForward(0, 0, keep_prob, TEST);
	return Y_hat;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
