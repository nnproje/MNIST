#include "NeuralNetwork.h"
Matrix* NeuralNetwork::FC_FeedForward(bool batchNorm, bool dropout, float* keep_prob, Mode mode)
{
	//TODO : Memory management, use delete to get rid of junk (mean & var) from cache after every epoch
    int L=numOfLayers;
	/*Initialization for parameters needed in batch norm*/
	Matrix* g1= nullptr;					//gamma for each layer
	Matrix* g2= nullptr;					//beta for each layer
	Matrix** mean = new Matrix*[L - 1];     //mean of z for each layer
	Matrix** var = new Matrix*[L - 1];      //standard deviation of z for each layer
	Matrix* zmeu= nullptr;				    //z-mean of z
	Matrix* z_telda= nullptr;				//(z-mean)/varience of z
	Matrix* z_new=nullptr;					//z after normalization,scaling and shifting by gamma and beta
	float eps = 1e-7;                       //to make sure that we don`t divide by zero
	float beta = 0.9;

	Matrix* z = nullptr;
	Matrix* A = nullptr;
    Matrix*  MatPtr= nullptr;
    Matrix*  temp1= nullptr;
    Matrix*  temp2= nullptr;


	for (int i = 0; i<L - 1; i++)
	{
		Matrix* W = FC_Parameters[CharGen("W", i + 1)];
		Matrix* b = FC_Parameters[CharGen("b", i + 1)];
		Matrix* Aprev = FC_Cache[CharGen("A", i)];
		if (batchNorm)
		{
			z = W->dot(Aprev);

			if (mode == TRAIN)
			{
			    //*mean[i] = z.sum("column") / z.Columns();
                temp1=z->SUM("column");
				mean[i] = temp1 ->div(z->Columns());
				delete temp1;

				zmeu = z->sub(mean[i]);

				//*var[i]=(zmeu.square()).sum("column") / z.Columns();
				temp1=zmeu->SQUARE();
				temp2=temp1->SUM("column");
				var[i] = temp2 ->div(z->Columns());
				delete temp1;
				delete temp2;

				//z_telda = zmeu / (*var[i]+eps).Sqrt();
                temp1=var[i]->add(eps);
                temp2=temp1->SQRT();
				z_telda = zmeu->div(temp2);
				delete temp1;
				delete temp2;

				if (isLastepoch)
				{
					Matrix* r_mean = FC_Parameters[CharGen("rm", i + 1)];
					//r_mean=r_mean*beta+(*mean[i])*(1-beta);
					MatPtr=r_mean;
					temp1=r_mean->mul(beta);
					temp2=mean[i]->mul(1 - beta);
					r_mean = temp1->add(temp2);
					delete MatPtr;
					delete temp1;
					delete temp2;
					FC_Parameters.replace(CharGen("rm", i + 1), r_mean);

					Matrix* r_var = FC_Parameters[CharGen("rv", i + 1)];
					//rr_var=r_var*beta+(*var[i])*(1-beta);
					MatPtr=r_var;
					temp1=r_var->mul(beta);
					temp2=var[i]->mul(1 - beta);
					r_var = temp1->add(temp2);
					delete MatPtr;
					delete temp1;
					delete temp2;
					FC_Parameters.replace(CharGen("rv", i + 1), r_var);
				}
			}
			else
			{
				Matrix* r_mean = FC_Parameters[CharGen("rm", i + 1)];
				Matrix* r_var = FC_Parameters[CharGen("rv", i + 1)];
				zmeu = z->sub(r_mean);

				//z_telda = zmeu / (r_var+eps).Sqrt();
				temp1=r_var->add(eps);
				temp2=temp1->SQRT();
				z_telda = zmeu->div(temp2);
				delete temp1;
				delete temp2;
			}

			delete z;

			g1 = FC_Parameters[CharGen("g1", i + 1)];
			g2 = FC_Parameters[CharGen("g2", i + 1)];

			//z_new=z_telda*g1+g2
			temp1=z_telda->mul(g1);
			z_new = temp1->add(g2);
			delete temp1;

			if (mode == TRAIN)
			{
				FC_Cache.put(CharGen("zm", i + 1), zmeu);
				FC_Cache.put(CharGen("zt", i + 1), z_telda);
				FC_Cache.put(CharGen("zn", i + 1), z_new);
				FC_Cache.put(CharGen("m", i + 1), mean[i]);
				FC_Cache.put(CharGen("var", i + 1), var[i]);
			}
		}
		else
		{
		    //z=W.dot(Aprev)+b;
		    temp1=W->dot(Aprev);
			z = temp1->add(b);
			delete temp1;
			FC_Cache.put(CharGen("z", i + 1), z);
		}

		ActivationType activation = layers[i + 1].activation;
		if(batchNorm)
          A=activ(z_new,activation);
          else
		  A=activ(z,activation);

		if (dropout && i != L - 1)
		{
			D[i + 1] = new Matrix(A->Rows(), A->Columns(), Bernoulli, keep_prob[i + 1]);

			// A=A * (*(D[i+1]));
			// A=A/keep_prob[i+1];
			MatPtr=A;
			temp1=A->mul(D[i + 1]);
			A=temp1->div(keep_prob[i + 1]);
			delete MatPtr;
			delete temp1;
		}
		FC_Cache.put(CharGen("A", i + 1), A);
	}
	return A; //yhat
}

