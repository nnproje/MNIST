#include "NeuralNetwork.h"

/*perform the required calculations for a convolution layer*/
void NeuralNetwork::convLayer(int stride, int A_index, ActivationType activation)
{
	//We will use str to extract Aprev, Aprev is the input data only in the first layer, after that Aprev comes from the pooling layer
	string str;
	if (A_index == 1)
		str = "AC";
	else
		str = "ACP";

	VectVolume Aprev = Conv_Cache[CharGen(str, A_index - 1)];
	VectVolume filters = Conv_Weights[CharGen("WC", A_index)];
	Matrix* b = Conv_biases[CharGen("bC", A_index)];

	int m = Aprev.size();
	int numOfFilters = filters.size();

	VectVolume A(m, numOfFilters); //
	VectVolume Z(m, numOfFilters); //

	Matrix* convTemp = nullptr;
	Matrix* z = nullptr;
	Matrix* a = nullptr;

	for (int i = 0; i<m; i++)
	{
		for (int j = 0; j<numOfFilters; j++)
		{
			//stride must be got from dictionary
			int stride = 1;

			//convolve the ith activations with the jth filter to produce convTemp which is pointer to (nh,nw) matrix
			convTemp = convolve(Aprev[i], filters[j], stride);

			//add the bias to the result of convolution (*z) = (*z) + b.access(j,0);
			z = convTemp->add(b->access(j, 0));

			//store z, needed later
			Z[i][j] = z;

			//pass the result to the activation a=activation(z)
			a = activ(z, activation);

			//a is pointer to the output of convolution, push it into the volume A[i]
			A[i][j] = a;

			//delete junk
			delete convTemp;
		}
	}

	Conv_Cache.put(CharGen("ZC", A_index), Z);
	Conv_Cache.put(CharGen("AC", A_index), A);
}

/*perform the required calculations for a pooling layer*/
void NeuralNetwork::poolLayer(int stride, int f, Mode mode, int A_index)
{
	VectVolume Aprev = Conv_Cache[CharGen("AC", A_index)];

	int m = Aprev.size();

	VectVolume A(m, Aprev[0].size());

	for (int i = 0; i<m; i++)
	{
		if (mode == MAX)
			maxPool(Aprev[i], A[i], f, stride);
		else if (mode == AVG)
			avgPool(Aprev[i], A[i], f, stride);
	}
	Conv_Cache.put(CharGen("ACP", A_index), A);
}



/*converts a Volume(vector of 2D images) into a Matrix(vector of 1D -flat- images) ==>*/
Matrix* to_1D(Volume& X_2D)
{
	Matrix* X_1D = new Matrix(X_2D[0]->Rows() * X_2D[0]->Columns(), X_2D.size());

	for (int k = 0; k < X_2D.size(); k++)
	{
		Matrix* curImg = X_2D[k];
		for (int i = 0; i < curImg->Rows(); i++)
		{
			for (int j = 0; j < curImg->Columns(); j++)
			{
				X_1D->access(i * curImg->Columns() + j, k) = curImg->access(i, j);
			}
		}
	}
	return X_1D;
}

/*converts a vector of Volume(output of last conv layer) into a Matrix(noFeatures X m) ==>*/
Matrix* to_FC(VectVolume A)
{
    int nh=A[0][0]->Rows();
    int nw=A[0][0]->Columns();
    int nc=A[0].size();
    int m=A.size();
	Matrix* A_1D = new Matrix(nh*nw*nc,m);

	for (int k = 0; k < m; k++)
	{
	    for(int kk=0;kk<nc;kk++)
        {
            for(int i=0;i<nh;i++)
            {
                for(int j=0;j<nw;j++)
                {
                    A_1D->access(j+nh*i+kk*nc,k)=A[k][kk]->access(i,j);
                }
            }
        }
	}
	return A_1D;
}


/*converts a Matrix(vector of 1D -flat- images) into a Volume(vector of 2D images) ==>*/
Volume to_2D(Matrix* X)
{
	int numOfImgs = X->Columns();
	int dim = sqrt(X->Rows());
	Volume X_2D(numOfImgs);
	for (int k = 0; k < numOfImgs; k++)
	{
		X_2D[k] = new Matrix(dim, dim);
		for(int i=0; i<dim; i++)
			for (int j = 0; j < dim; j++)
			{
				X_2D[k]->access(i, j) = X->access(k, i*dim + j);
			}
	}
	return X_2D;
}

/*converts a Matrix(vector of 1D -flat- images) into a VectVolume(vector of volumes)*/
VectVolume to_VectorOfVolume(Matrix* A, int nh, int nw, int nc, int m)
{
	VectVolume V(m,nc,nh,nw);
	for (int k = 0; k < m; k++)
	{
		for (int kk = 0; kk<nc; kk++)
		{
			for (int i = 0; i<nh; i++)
			{
				for (int j = 0; j<nw; j++)
				{
					V[k][kk]->access(i, j) = A->access(j + nh * i + kk * nc, k);
				}
			}
		}
	}
	return V;
}


/*Extend a square matrix into (n+p x n+p) dims, the extended entries have the value value ==>*/
Matrix* pad(Matrix* img, int p, float value)
{
	if (img->Rows() != img->Columns())
		cout << "this is not square matrix" << endl;

	int n = img->Rows();
	int m = n + 2 * p;

	Matrix* newImg = new Matrix(m, m);

	for (int i = 0; i < m; i++)
		for (int j = 0; j < m; j++)
		{
			if (i < (m - n - p) || j<(m - n - p) || i>(p + n - 1) || j >(p + n - 1))
				newImg->access(i, j) = 0;
			else
				newImg->access(i, j) = img->access(i - p, j - p);
		}

	return newImg;
}

/*convolve a volume Aprev with filter*/
Matrix* convolve(Volume& Aprev, Volume& filter, int stride)
{
	int nc = filter.size();
	int f = filter[0]->Rows();
	int nc_prev = Aprev.size();
	int nh_prev = Aprev[0]->Rows();
	int nw_prev = Aprev[0]->Columns();
	int nh = (nh_prev - f) / stride + 1;
	int nw = (nw_prev - f) / stride + 1;
	Matrix* result = new Matrix(nh, nw);
	Matrix* Acc = new Matrix(f, f, 0);
	Matrix* slice = nullptr;
	Matrix* temp1 = nullptr;
	Matrix* temp2 = nullptr;

	if (nc != nc_prev)
	{
		cout << "dimension err in convolution!" << endl;
	}

	for (int i = 0; i < nh; i++)
	{

		for (int j = 0; j<nw; j++)
		{
			int vert_start = i * stride;
			int vert_end = vert_start + f;
			int horz_start = j * stride;
			int horz_end = horz_start + f;
			Acc = new Matrix(f, f, 0);

			for (int c = 0; c < nc; c++)
			{
				//slice = Aprev[c](vert_start, horz_start, vert_end - 1, horz_end - 1);
				slice = Aprev[c]->SubMat(vert_start, horz_start, vert_end - 1, horz_end - 1);

				//Acc = Acc + slice * filter[c];
				temp1 = slice->mul(filter[c]);
				temp2 = Acc;
				Acc = Acc->add(temp1);

				delete slice;
				delete temp1;
				delete temp2;
			}


			result->access(i, j) = Acc->sumall();
			delete Acc;
		}
	}


	return result;
}


/*perform max pooling in Aprev and return the result in A*/
/*A must be empty volume, its contents is created inside*/
void maxPool(Volume& Aprev, Volume& A, int f, int stride)
{
	int nc = Aprev.size();
	int nh_prev = Aprev[0]->Rows();
	int nw_prev = Aprev[0]->Columns();
	int nh = (nh_prev - f) / stride + 1;
	int nw = (nw_prev - f) / stride + 1;
	Matrix* slice = nullptr;

	for (int c = 0; c < nc; c++)
	{
		A[c] = new Matrix(nh, nw);
		for (int i = 0; i<nh; i++)
			for (int j = 0; j<nw; j++)
			{
				int vert_start = i * stride;
				int vert_end = vert_start + f;
				int horz_start = j * stride;
				int horz_end = horz_start + f;

				slice = Aprev[c]->SubMat(vert_start, horz_start, vert_end - 1, horz_end - 1);

				A[c]->access(i, j) = slice->MaxElement();

				delete slice;
			}
	}
}


/*perform average pooling in Aprev and return the result in A*/
void avgPool(Volume& Aprev, Volume& A, int f, int stride)
{
	int nc = Aprev.size();
	int nh_prev = Aprev[0]->Rows();
	int nw_prev = Aprev[0]->Columns();
	int nh = (nh_prev - f) / stride + 1;
	int nw = (nw_prev - f) / stride + 1;
	Matrix* slice = nullptr;

	for (int c = 0; c < nc; c++)
	{
		A[c] = new Matrix(nh, nw);
		for (int i = 0; i<nh; i++)
			for (int j = 0; j<nw; j++)
			{
				int vert_start = i * stride;
				int vert_end = vert_start + f;
				int horz_start = j * stride;
				int horz_end = horz_start + f;

				slice = Aprev[c]->SubMat(vert_start, horz_start, vert_end - 1, horz_end - 1);

				float sum = slice->sumall();

				A[c]->access(i, j) = sum / (slice->Rows() * slice->Columns());

				delete slice;
			}
	}

}




