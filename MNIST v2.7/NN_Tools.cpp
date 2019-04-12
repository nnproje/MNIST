#include "NN_Tools.h"

using namespace std;
////////////////////////////////////////////////////////////////////////////////////
string CharGen(string name, int i)
{
    int temp = i;
    int counter1;   //number of decimal digits in i

	if (temp == 0)
		counter1 = 1;
	else
	{
		for (counter1 = 0; temp != 0; counter1++)
			temp = temp / 10;
	}


    int counter2=name.size();   //number of chars in name

    string result;
    if(counter2==1){result="W0";}
    if(counter2==2){result="dW0";}
    if(counter2==3){result="Sdw0";}
    if(counter2==4){result="dACP0";}

    for (unsigned int j = 0; j<name.size(); j++) //copy the name into result
        result[j] = name[j];

    int j = counter1 + counter2 - 1;      //copy the number into result
    temp = i;
    do
    {
        result[j] = '0' + (temp % 10);
        temp = temp / 10;
        j--;
    }while (temp != 0);

    return result;
}
////////////////////////////////////////////////////////////////////////////////////
void AccuracyTest(Matrix* Y, Matrix* Y_hat, string devOrtest)
{
	float errSum = 0;
	for (int j = 0; j<Y_hat->Columns(); j++)
	{
		float maximum = Y_hat->access(0, j);
		int index = 0;
		for (int i = 1; i<Y_hat->Rows(); i++)
		{
			if (Y_hat->access(i, j)>maximum)
			{
				maximum = Y_hat->access(i, j);
				index = i;
			}
		}

		if (Y->access(index, j) != 1)
			errSum++;

	}
	cout << endl << "<==============================================================================>" << endl;
	if (devOrtest == "dev")
		cout << endl << "Development set results : " << endl;
	else if (devOrtest == "test")
		cout << endl << "Test set results : " << endl;

	float Accur = 1 - ((errSum) / Y->Columns());
	cout << endl << "The number of false predictions = " << errSum << endl;
	cout << endl << "Accuracy = " << Accur * 100 << " %" << endl << endl;
}