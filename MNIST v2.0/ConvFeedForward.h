#pragma once
#include "Volume.h"
typedef matrix<float> Matrix;

//*************************************************************************************************/
//4D discription for a 4D element A(m,nc,nh,nw):
//1- A is a vector of volumes that has a size m
//2- A[i] is a volume with nc channels, it represents the activations of some layer for the ith example
//3- A[i][j] is a pointer to the jth Matrix with nh hight and nw width in the ith example
//4- A[i][0] represents the first channel in the volume, take them from top to down
//input:
//Aprev(m,nc_prev,nh_prev,nw_prev) , filters(numOfFilters,nc,f,f) , b(numOfFilters,nh,nw)
//output:
//A(m,numOfFilters,nh,nw)
//*************************************************************************************************/

Volume	to_2D(Matrix* X);
Matrix* to_1D(Volume& X_2D);
Matrix* pad(Matrix* img, int p, float value);
Matrix* convolve(Volume& Aprev, Volume& filter, int s);
vector<Volume> convLayer(vector<Volume>& Aprev, vector<Volume>& filters, Volume b);
void maxPool(Volume& Aprev, Volume& A, int f, int s);
void avgPool(Volume& Aprev, Volume& A, int f, int s);
vector<Volume> poolLayer(vector<Volume>& Aprev, int f, string mode);


/*converts a Volume(vector of 2D images) into a Matrix(vector of 1D -flat- images)*/
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

/*converts a Matrix(vector of 1D -flat- images) into a Volume(vector of 2D images)*/
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

/*Extend a square matrix into (n+p x n+p) dims, the extended entries have the value value*/
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

	if (nc != nc_prev)
	{
		cout << "dimension err in convolution!" << endl;
	}

	for (int i = 0; i<nh; i++)
		for (int j = 0; j<nw; j++)
		{
			int vert_start = i * stride;
			int vert_end = vert_start + f;
			int horz_start = j * stride;
			int horz_end = horz_start + f;
			Matrix temp(f, f, 0);
			Matrix slice(f, f, 0);

			for (int c = 0; c < nc; c++)
			{
				slice = (*Aprev[c])(vert_start, horz_start, vert_end - 1, horz_end - 1);
				temp = temp + slice * (*filter[c]);
			}
			result->access(i, j) = temp.sumall();
		}
	return result;
}


/*perform the required calculations for a convolution layer*/
vector<Volume> convLayer(vector<Volume>& Aprev, vector<Volume>& filters, Volume b)
{
	int numOfFilters = filters.size();
	int m = Aprev.size();
	int stride = 1;
	vector<Volume> A(m);
	for (int i = 0; i < m; i++)
	{
		A[i].resize(numOfFilters);
	}

	for (int i = 0; i<m; i++)
	{
		for (int j = 0; j<numOfFilters; j++)
		{
			//stride must be got from dictionary
			int stride = 1;
			//convolve the ith activations with the jth filter to produce z which is pointer to (nh,nw) matrix
			Matrix* z = convolve(Aprev[i], filters[j], stride);
			//add the bias to the result of convolution
			(*z) = (*z) + (*b[j]);
			//pass the result to the activation
			(*z) = (*z); //relu or anything
						 //z is pointer to the output of convolution, push it into the volume A[i]
			A[i][j] = z;
		}
	}
	return A;
}


/*perform max pooling in Aprev and return the result in A*/
void maxPool(Volume& Aprev, Volume& A, int f, int stride)
{
	int nc = Aprev.size();
	int nh_prev = Aprev[0]->Rows();
	int nw_prev = Aprev[0]->Columns();
	int nh = (nh_prev - f) / stride + 1;
	int nw = (nw_prev - f) / stride + 1;
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
				float MAX = ((*Aprev[c])(vert_start, horz_start, vert_end - 1, horz_end - 1)).MaxElement();
				A[c]->access(i, j) = MAX;
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
				Matrix slice(nh, nw);
				slice = (*Aprev[c])(vert_start, horz_start, vert_end - 1, horz_end - 1);
				float AVG = slice.sumall();
				A[c]->access(i, j) = AVG / (slice.Rows() * slice.Columns());
			}
	}

}


/*perform the required calculations for a pooling layer*/
vector<Volume> poolLayer(vector<Volume>& Aprev, int f, string mode)
{
	int stride = 1;					//you get it from hyberparameters
	int m = Aprev.size();
	vector<Volume> A(m);
	for (int i = 0; i < m; i++)
	{
		A[i].resize(Aprev[i].size());
	}

	for (int i = 0; i<m; i++)
	{
		if (mode == "max")
			maxPool(Aprev[i], A[i], f, stride);
		else if (mode == "avg")
			avgPool(Aprev[i], A[i], f, stride);
	}
	return A;
}

