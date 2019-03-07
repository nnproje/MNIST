#include "CNN.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CNN::CNN(string TypeOfConvNet)
{
	ConvNetType = TypeOfConvNet;
	if (ConvNetType == "LeNet1")
	{
		init_LeNet1();
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CNN::train()
{

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CNN::test(Matrix& X_test, Matrix& Y_test, string path, string devOrtest, bool batchNorm, bool dropout, float* keep_prob)
{
	/******************get Y_hat********************************/
	FC_Cache.clear();
	Conv_Cache.clear();
	Matrix Y_hat;
	if (ConvNetType == "LeNet1")
	{
		//return result in Yhat
		test_LeNet1();
	}
	FC_Cache.clear();
	Conv_Cache.clear();


	/********************AccuracyTest*****************************/
	float errSum = 0;
	for (int j = 0; j<Y_hat.Columns(); j++)
	{
		float maximum = Y_hat.access(0, j);
		int index = 0;
		for (int i = 1; i<Y_hat.Rows(); i++)
		{
			if (Y_hat.access(i, j)>maximum)
			{
				maximum = Y_hat.access(i, j);
				index = i;
			}
		}

		if (Y_test.access(index, j) != 1)
			errSum++;

	}
	cout << endl << "<==============================================================================>" << endl;
	if (devOrtest == "dev")
		cout << endl << "Development set results : " << endl;
	else if (devOrtest == "test")
		cout << endl << "Test set results : " << endl;

	float Accur = 1 - ((errSum) / Y_test.Columns());
	cout << endl << "The number of false predictions = " << errSum << endl;
	cout << endl << "Accuracy = " << Accur * 100 << " %" << endl << endl;

	//storedata(X_test,Y_test,Y_hat,path);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////