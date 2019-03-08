#include "CNN.h"
#include "NN_Tools.h"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CNN::FC_BackProp(const Matrix& X, const Matrix& Y, const Matrix& Y_hat, float& alpha, layer* layers, int L, int iteration, int m, float lambda, bool batchNorm, bool dropout, float* keep_prob)
{
	FC_CalGrads(X, Y, Y_hat, layers, L, lambda, batchNorm, dropout, keep_prob);
	FC_UpdateParameters(alpha, layers, L, iteration, m, batchNorm);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CNN::FC_CalGrads(const Matrix& X, const Matrix& Y, const Matrix& Y_hat, layer* layers, int L, float lambda, bool batchNorm, bool dropout, float* keep_prob)
{
	float m = X.Columns();
	/*Initialization of parameters for batch Norm*/
	Matrix** dg1 = new Matrix*[L - 1];   //mean of z for each layer
	Matrix** dg2 = new Matrix*[L - 1];   //standard deviation of z for each layer
	for (int ii = 0; ii<L - 1; ii++)
	{
		dg1[ii] = new Matrix(layers[ii + 1].neurons, 1);
		dg2[ii] = new Matrix(layers[ii + 1].neurons, 1);
	}
	Matrix g1(1, 1);
	Matrix g2(1, 1);
	Matrix mean(1, 1);
	Matrix var(1, 1);
	float eps = 1e-7;
	/*End of Initialization of parameters for batch Norm*/
	Matrix dbLast(1, 1);
	Matrix dbl(1, 1);
	Matrix db1(1, 1);
	Matrix b(1, 1);
	/*End of initialization for batch norm*/

	/*Output layer*/
	Matrix dzLast(1, 1);
	Matrix zLast(1, 1);
	if (batchNorm)
		zLast = FC_Cache[CharGen("zn", L - 1)];
	else
		zLast = FC_Cache[CharGen("z", L - 1)];

	if (ErrorType == "SquareErr") //not compatible with dropout
	{
		Matrix dALast = (Y_hat - Y) / m;

		if (layers[L - 1].activation == "softmax")
			dzLast = dALast * dsoftmax(zLast);
		if (layers[L - 1].activation == "sigmoid")
			dzLast = dALast * dsigmoid(zLast);
		else if (layers[L - 1].activation == "tanh")
			dzLast = dALast * dtanh(zLast);
		else if (layers[L - 1].activation == "relu")
			dzLast = dALast * drelu(zLast);
		else if (layers[L - 1].activation == "leakyRelu")
			dzLast = dALast * dleakyRelu(zLast);
		else if (layers[L - 1].activation == "satLinear")
			dzLast = dALast * dsatLinear(zLast);
		else if (layers[L - 1].activation == "Linear")
			dzLast = dALast * dLinear(zLast);
	}
	else if (ErrorType == "CrossEntropy")
	{
		dzLast = Y_hat - Y;
	}


	if (batchNorm)
	{

		//visit https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
		//visit https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

		Matrix g1 = FC_Parameters[CharGen("g1", L - 1)];
		Matrix g2 = FC_Parameters[CharGen("g2", L - 1)];
		Matrix var = FC_Cache[CharGen("var", L - 1)];
		Matrix z_telda = FC_Cache[CharGen("zt", L - 1)];

		//getting dgamma and dbeta
		/*dzlast here means dzlast_new after normalizing*/
		*dg1[L - 2] = (dzLast*z_telda).sum("column");
		*dg2[L - 2] = dzLast.sum("column");

		//getting dz_telda
		Matrix dz_telda = dzLast * (g1);

		//getting dvariance
		Matrix zmeu = FC_Cache[CharGen("zm", L - 1)];
		Matrix divar = (dz_telda*zmeu).sum("column");
		Matrix dsqrtvar = divar / (var + eps);
		Matrix t = (var + eps).Sqrt();
		Matrix dvar = (dsqrtvar*-0.5) / t;

		//getting dmeu
		Matrix dmeu1 = (dz_telda*-1) / t;
		dmeu1 = dmeu1.sum("column");
		Matrix dmeu2 = (zmeu*-2)*dvar;
		dmeu2 = (dmeu2.sum("column")) / m;
		Matrix dmeu = dmeu1 + dmeu2;

		//getting dzlast (dout) for the incoming layer
		/*
		This matrix (dzlast) contains the gradient of the loss function with respect to the input of the BatchNorm-Layer.
		This dzlast is the gradient of zlast=W.A[L-1]+b (b is neglected)
		This gradient dzlast is also what we give as input (dout) to the backwardpass of the next layer..this happens through dAprevLast
		As for this layer we receive dout from the layer above.
		*/
		dzLast = dz_telda / t;
		dzLast = dzLast + (zmeu*dvar)*(2 / m);
		dzLast = dzLast + dmeu / m;

		FC_Grades.put(CharGen("dg1", L - 1), *dg1[L - 2]);
		FC_Grades.put(CharGen("dg2", L - 1), *dg2[L - 2]);
	}
	Matrix AprevLast = FC_Cache[CharGen("A", L - 2)];
	Matrix dWLast = dzLast.dot(AprevLast.transpose()) / m;

	if (!batchNorm)
	{
		dbLast = dzLast.sum("column") / m;
	}

	Matrix WLast = FC_Parameters[CharGen("W", L - 1)];
	Matrix WLast_trans = WLast.transpose();
	Matrix dAprevLast = WLast_trans.dot(dzLast);

	if (dropout)
	{
		dAprevLast = dAprevLast * (*(D[L - 2]));
		dAprevLast = dAprevLast / keep_prob[L - 2];
	}

	if (lambda != 0)
	{
		dWLast = dWLast + WLast * (lambda / m);
	}

	FC_Grades.put(CharGen("dW", L - 1), dWLast);

	if (!batchNorm)
		FC_Grades.put(CharGen("db", L - 1), dbLast);

	FC_Grades.put(CharGen("dA", L - 2), dAprevLast);
	/*End of output layer*/

	/*Hidden layers*/
	for (int i = L - 2; i > 0; i--)
	{

		Matrix dAl = FC_Grades[CharGen("dA", i)];
		FC_Grades.erase(CharGen("dA", i));
		Matrix zl(1, 1);
		Matrix dzl(1, 1);

		if (batchNorm)
			zl = FC_Cache[CharGen("zn", i)];
		else
			zl = FC_Cache[CharGen("z", i)];


		if (layers[i].activation == "sigmoid")
			dzl = dAl * dsigmoid(zl);
		else if (layers[i].activation == "tanh")
			dzl = dAl * dtanh(zl);
		else if (layers[i].activation == "relu")
			dzl = dAl * drelu(zl);
		else if (layers[i].activation == "leakyRelu")
			dzl = dAl * dleakyRelu(zl);
		else if (layers[i].activation == "satLinear")
			dzl = dAl * dsatLinear(zl);
		else if (layers[i].activation == "Linear")
			dzl = dAl * dLinear(zl);

		if (batchNorm)
		{

			Matrix g1 = FC_Parameters[CharGen("g1", i)];
			Matrix g2 = FC_Parameters[CharGen("g2", i)];
			Matrix var = FC_Cache[CharGen("var", i)];
			Matrix z_telda = FC_Cache[CharGen("zt", i)];

			//getting dgamma and dbeta
			/*dz here means dznew after normalizing*/
			*dg1[i - 1] = (dzl*z_telda).sum("column");
			*dg2[i - 1] = dzl.sum("column");

			//getting dz_telda
			Matrix dz_telda = dzl * (g1);

			//getting dvariance
			Matrix zmeu = FC_Cache[CharGen("zm", i)];
			Matrix divar = (dz_telda*zmeu).sum("column");
			Matrix dsqrtvar = divar / (var + eps);
			Matrix t = (var + eps).Sqrt();
			Matrix dvar = (dsqrtvar*-0.5) / t;

			//getting dmeu
			Matrix dmeu1 = (dz_telda*-1) / t;
			dmeu1 = dmeu1.sum("column");
			Matrix dmeu2 = (zmeu*-2)*dvar;
			dmeu2 = (dmeu2.sum("column")) / m;
			Matrix dmeu = dmeu1 + dmeu2;

			//getting dz (dout for the incoming layer)
			/*
			This matrix (dz) contains the gradient of the loss function with respect to the input of the BatchNorm-Layer.
			This dz is the gradient of z=W.A[l-1]+b (b is neglected)
			This gradient dz is also what we give as input (dout) to the backwardpass of the next layer..this happens through dAl_1
			As for this layer we receive dout from the layer above.
			*/
			dzl = dz_telda / t;
			dzl = dzl + (zmeu*dvar)*(2 / m);
			dzl = dzl + dmeu / m;

			FC_Grades.put(CharGen("dg1", i), *dg1[i - 1]);
			FC_Grades.put(CharGen("dg2", i), *dg2[i - 1]);
		}
		Matrix Al_1 = FC_Cache[CharGen("A", i - 1)];
		Matrix dWl = dzl.dot(Al_1.transpose()) / m;
		if (!batchNorm)
			dbl = dzl.sum("column") / m;

		Matrix Wl = FC_Parameters[CharGen("W", i)];
		Matrix Wl_trans = Wl.transpose();
		Matrix dAl_1 = Wl_trans.dot(dzl);

		if (dropout && i != 1)
		{
			dAl_1 = dAl_1 * (*(D[i - 1]));
			dAl_1 = dAl_1 / keep_prob[i - 1];
		}

		if (lambda != 0)
		{
			dWl = dWl + Wl * (lambda / m);
		}

		FC_Grades.put(CharGen("dW", i), dWl);

		if (!batchNorm)
			FC_Grades.put(CharGen("db", i), dbl);

		if (i != 1)
			FC_Grades.put(CharGen("dA", i - 1), dAl_1);
	}

	if (dropout)
	{
		for (int j = 1; j<numOfLayers - 1; j++)
			delete D[j];
	}

	/*End of hidden layers*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CNN::FC_UpdateParameters(float& alpha, layer* layers, int L, int iteration, int m, bool batchNorm)
{
	/*START OF GRADIENT DESCENT OPTIMIZER*/
	if (optimizer == "GradientDescent")
	{

		for (int i = 0; i < L - 1; i++)
		{
			if (batchNorm)
			{
				Matrix g1 = FC_Parameters[CharGen("g1", i + 1)];
				FC_Parameters.erase(CharGen("g1", i + 1));
				Matrix dg1 = FC_Grades[CharGen("g1", i + 1)];
				g1 = g1 - dg1 * alpha;
				FC_Parameters.put(CharGen("g1", i + 1), g1);

				Matrix g2 = FC_Parameters[CharGen("g2", i + 1)];
				FC_Parameters.erase(CharGen("g2", i + 1));
				Matrix dg2 = FC_Grades[CharGen("g2", i + 1)];
				g2 = g2 - dg2 * alpha;
				FC_Parameters.put(CharGen("g2", i + 1), g2);

				FC_Grades.erase(CharGen("g1", i + 1));
				FC_Grades.erase(CharGen("g2", i + 1));
			}
			Matrix Wul = FC_Parameters[CharGen("W", i + 1)] - FC_Grades[CharGen("dW", i + 1)] * alpha;
			FC_Parameters.erase(CharGen("W", i + 1));
			FC_Parameters.put(CharGen("W", i + 1), Wul);
			FC_Grades.erase(CharGen("dW", i + 1));
			if (!batchNorm)
			{
				Matrix bul = FC_Parameters[CharGen("b", i + 1)] - FC_Grades[CharGen("db", i + 1)] * alpha;
				FC_Parameters.erase(CharGen("b", i + 1));
				FC_Parameters.put(CharGen("b", i + 1), bul);
				FC_Grades.erase(CharGen("db", i + 1));
			}
		}
	}
	/*END OF GRADIENT DESCENT OPTIMIZER*/

	/*START OF ADAM OPTIMIZER*/
	else if (optimizer == "Adam")
	{
		float beta1 = 0.9;
		float beta2 = 0.999;
		float epsilon = 1e-8;
		Matrix b(1, 1);
		Matrix db(1, 1);
		Matrix Vdb(1, 1);
		Matrix Sdb(1, 1);
		Matrix Sdb_corr(1, 1);
		Matrix Vdb_corr(1, 1);
		for (int i = 0; i < L - 1; i++)
		{
			/*Getting variables from dictionaries*/
			Matrix Vdw = FC_Grades[CharGen("Vdw", i + 1)];
			Matrix Sdw = FC_Grades[CharGen("Sdw", i + 1)];
			if (!batchNorm)
			{
				Vdb = FC_Grades[CharGen("Vdb", i + 1)];
				Sdb = FC_Grades[CharGen("Sdb", i + 1)];
			}

			Matrix dW = FC_Grades[CharGen("dW", i + 1)];
			Matrix W = FC_Parameters[CharGen("W", i + 1)];
			if (!batchNorm)
			{
				db = FC_Grades[CharGen("db", i + 1)];
				b = FC_Parameters[CharGen("b", i + 1)];
			}

			/*Updating Vdw, Vdb, Sdw, Sdb*/
			Vdw = (Vdw * (beta1 * momentum)) + (dW * (1 - beta1 * momentum));
			FC_Grades.erase(CharGen("Vdw", i + 1));
			FC_Grades.put(CharGen("Vdw", i + 1), Vdw);

			if (!batchNorm)
			{
				Vdb = (Vdb * (beta1 * momentum)) + (db * (1 - beta1 * momentum));
				FC_Grades.erase(CharGen("Vdb", i + 1));
				FC_Grades.put(CharGen("Vdb", i + 1), Vdb);
			}

			Sdw = (Sdw * beta2) + (dW.square() * (1 - beta2));
			FC_Grades.erase(CharGen("Sdw", i + 1));
			FC_Grades.put(CharGen("Sdw", i + 1), Sdw);

			if (!batchNorm)
			{
				Sdb = (Sdb * beta2) + (db.square() * (1 - beta2));
				FC_Grades.erase(CharGen("Sdb", i + 1));
				FC_Grades.put(CharGen("Sdb", i + 1), Sdb);
			}

			/*Correcting first iterations*/
			Matrix Vdw_corr = Vdw / (1 - pow(beta1, iteration + 1));
			Matrix Sdw_corr = Sdw / (1 - pow(beta2, iteration + 1));
			if (!batchNorm)
			{
				Sdb_corr = Sdb / (1 - pow(beta2, iteration + 1));
				Vdb_corr = Vdb / (1 - pow(beta1, iteration + 1));
			}

			/*Updating parameters*/
			Matrix temp1 = Vdw_corr / (Sdw_corr.Sqrt() + epsilon);
			Matrix Wu = W - temp1 * alpha;
			FC_Parameters.erase(CharGen("W", i + 1));
			FC_Parameters.put(CharGen("W", i + 1), Wu);

			if (!batchNorm)
			{
				Matrix temp2 = Vdb_corr / (Sdb_corr.Sqrt() + epsilon);
				Matrix bu = b - temp2 * alpha;
				FC_Parameters.erase(CharGen("b", i + 1));
				FC_Parameters.put(CharGen("b", i + 1), bu);
			}

			/*Erasing dW, db*/
			FC_Grades.erase(CharGen("dW", i + 1));
			if (!batchNorm)
				FC_Grades.erase(CharGen("db", i + 1));

			if (batchNorm)
			{
				/*Getting variables from dictionaries*/
				Matrix vdg1 = FC_Grades[CharGen("vg1", i + 1)];
				Matrix sdg1 = FC_Grades[CharGen("sg1", i + 1)];
				Matrix vdg2 = FC_Grades[CharGen("vg2", i + 1)];
				Matrix sdg2 = FC_Grades[CharGen("sg2", i + 1)];
				Matrix dg1 = FC_Grades[CharGen("dg1", i + 1)];
				Matrix dg2 = FC_Grades[CharGen("dg2", i + 1)];
				Matrix g1 = FC_Parameters[CharGen("g1", i + 1)];
				Matrix g2 = FC_Parameters[CharGen("g2", i + 1)];

				/*Updating vdg1, vdg2, sdg1, sdg2*/
				vdg1 = (vdg1 * (beta1 * momentum)) + (dg1 * (1 - beta1 * momentum));
				FC_Grades.erase(CharGen("vg1", i + 1));
				FC_Grades.put(CharGen("vg1", i + 1), vdg1);

				vdg2 = (vdg2 * (beta1 * momentum)) + (dg2 * (1 - beta1 * momentum));
				FC_Grades.erase(CharGen("vg2", i + 1));
				FC_Grades.put(CharGen("vg2", i + 1), vdg2);

				sdg1 = (sdg1 * beta2) + (dg1.square() * (1 - beta2));
				FC_Grades.erase(CharGen("sg1", i + 1));
				FC_Grades.put(CharGen("sg1", i + 1), sdg1);

				sdg2 = (sdg2 * beta2) + (dg2.square() * (1 - beta2));
				FC_Grades.erase(CharGen("sg2", i + 1));
				FC_Grades.put(CharGen("sg2", i + 1), sdg2);
				/*Correcting first iterations*/
				Matrix vdg1_corr = vdg1 / (1 - pow(beta1, iteration + 1));
				Matrix vdg2_corr = vdg2 / (1 - pow(beta1, iteration + 1));
				Matrix sdg1_corr = sdg1 / (1 - pow(beta2, iteration + 1));
				Matrix sdg2_corr = sdg2 / (1 - pow(beta2, iteration + 1));
				/*Updating parameters*/
				Matrix temp1 = vdg1_corr / (sdg1_corr.Sqrt() + epsilon);
				Matrix g1u = g1 - temp1 * alpha;
				FC_Parameters.erase(CharGen("g1", i + 1));
				FC_Parameters.put(CharGen("g1", i + 1), g1u);

				Matrix temp2 = vdg2_corr / (sdg2_corr.Sqrt() + epsilon);
				Matrix g2u = g2 - temp2 * alpha;
				FC_Parameters.erase(CharGen("g2", i + 1));
				FC_Parameters.put(CharGen("g2", i + 1), g2u);

				/*Erasing dgamma1, dgamma2*/
				FC_Grades.erase(CharGen("dg1", i + 1));
				FC_Grades.erase(CharGen("dg2", i + 1));
			}
		}

	}
	/*END OF ADAM OPTIMIZER*/
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
