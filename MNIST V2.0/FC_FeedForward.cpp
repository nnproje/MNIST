#include "CNN.h"
Matrix CNN::FC_FeedForward(layer* layers, int L, bool batchNorm, bool dropout, float* keep_prob, string mode)
{
	//TODO : Memory management, use delete to get rid of junk (mean & var) from cache after every epoch

	/*Initialization for parameters needed in batch norm*/
	Matrix g1(1, 1);   //gamma for each layer
	Matrix g2(1, 1);   //beta for each layer
	Matrix** mean = new Matrix*[L - 1];     //mean of z for each layer
	Matrix** var = new Matrix*[L - 1];      //standard deviation of z for each layer
	for (int ii = 0; ii<L - 1; ii++)
	{
		mean[ii] = new Matrix(layers[ii + 1].neurons, 1);
		var[ii] = new Matrix(layers[ii + 1].neurons, 1);
	}
	Matrix zmeu(1, 1);     //z-mean of z
	Matrix z_telda(1, 1);  //(z-mean)/varience of z
	Matrix z_new(1, 1);    //z after normalization,scaling and shifting by gamma and beta
	float eps = 1e-7;       //to make sure that we don`t divide by zero
	float beta = 0.9;
	/*End of initialization for parameters needed in batch norm*/

	for (int i = 0; i<L - 1; i++)
	{
		Matrix W = FC_Parameters[CharGen("W", i + 1)];
		Matrix b = FC_Parameters[CharGen("b", i + 1)];
		Matrix z(1, 1);
		Matrix A(1, 1);
		Matrix Aprev = FC_Cache[CharGen("A", i)];
		if (batchNorm)
		{
			z = W.dot(Aprev);

			if (mode == "train")
			{
				*mean[i] = z.sum("column") / z.Columns();
				zmeu = z - *mean[i];
				*var[i] = (zmeu.square()).sum("column") / z.Columns();
				z_telda = zmeu / (*var[i] + eps).Sqrt();

				if (isLastepoch)
				{
					Matrix r_mean = FC_Parameters[CharGen("rm", i + 1)];
					r_mean = r_mean * beta + (*mean[i])*(1 - beta);
					Matrix r_var = FC_Parameters[CharGen("rv", i + 1)];
					r_var = r_var * beta + (*var[i])*(1 - beta);

					FC_Parameters.replace(CharGen("rm", i + 1), r_mean);
					FC_Parameters.replace(CharGen("rv", i + 1), r_var);
				}
			}
			else
			{
				Matrix r_mean = FC_Parameters[CharGen("rm", i + 1)];
				Matrix r_var = FC_Parameters[CharGen("rv", i + 1)];
				zmeu = z - r_mean;
				z_telda = zmeu / (r_var + eps).Sqrt();
			}


			g1 = FC_Parameters[CharGen("g1", i + 1)];
			g2 = FC_Parameters[CharGen("g2", i + 1)];
			z_new = z_telda * g1 + g2;
			FC_Cache.put(CharGen("zm", i + 1), zmeu);
			FC_Cache.put(CharGen("zt", i + 1), z_telda);
			FC_Cache.put(CharGen("zn", i + 1), z_new);
			FC_Cache.put(CharGen("m", i + 1), *mean[i]);
			FC_Cache.put(CharGen("var", i + 1), *var[i]);
		}
		else
		{
			z = W.dot(Aprev) + b;
			FC_Cache.put(CharGen("z", i + 1), z);
		}

		if (batchNorm)
			z = z_new;

		string activation = layers[i + 1].activation;

		if (activation == "relu")
			A = relu(z);
		else if (activation == "leakyRelu")
			A = leakyRelu(z);
		else if (activation == "tanh")
			A = mytanh(z);
		else if (activation == "sigmoid")
			A = sigmoid(z);
		else if (activation == "softmax")
			A = softmax(z);
		else if (activation == "satLinear")
			A = satLinear(z);
		else if (activation == "Linear")
			A = Linear(z);

		if (dropout && i != L - 1)
		{
			D[i + 1] = new Matrix(A.Rows(), A.Columns(), Bernoulli, keep_prob[i + 1]);
			A = A * (*(D[i + 1]));
			A = A / keep_prob[i + 1];
			FC_Cache.put(CharGen("A", i + 1), A);
		}
		else
			FC_Cache.put(CharGen("A", i + 1), A);

	}

	Matrix yhat = FC_Cache[CharGen("A", L - 1)];
	return yhat;
}

