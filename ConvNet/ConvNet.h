#pragma once
#include <iostream>
#include <conio.h>
#include <process.h>
#include <cmath>
#include <math.h>
#include "opencv2\opencv.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "Matrix.h"
#include "Dictionary.h"
#include "Activations.h"
#include "Volume.h"
#include "layer.h"
#include "dataset2D.h"
#include "CImg.h"

using namespace std;
using namespace cimg_library;
using namespace cv;
typedef matrix<float> Matrix;



Matrix* pad(Matrix* img, int p, float value);
Matrix* gausianFilter(int x, int y, float sigma);
Matrix* gaussianBlur(Matrix* img, int filterSize, float sigma);
Matrix* convolve(Volume& Aprev, Volume& filter, int s);
vector<Volume> convLayer(vector<Volume>& Aprev, vector<Volume>& filters, Volume b);
float norm_L1(Matrix* x);
Matrix* elasticDistortion(Matrix* img, int filterSize, float sigma, float alpha);
void visualize(Matrix* img);

Matrix* pad(Matrix* img, int p, float value)
{
	//Extend a square matrix into (n+p x n+p) dims, the extended entries have the value value
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
			int stride = 1;
			//convolve the ith activations with the jth filter to produce z which is pointer to (nh,nw) matrix
			Matrix* z = convolve(Aprev[i], filters[j], stride);
			//add the bias to the result of convolution
			(*z) = (*z) + (*b[j]);
			//pass the result to the activation
			(*z) = relu(*z);
			//z is pointer to the output of convolution, push it into the volume A[i] 
			A[i][j] = z;
		}
	}
	return A;
}


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
			int vert_end = i + f;
			int horz_start = j * stride;
			int horz_end = j + f;
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


Matrix* gausianFilter(int x, int y, float sigma)
{
	Matrix* GaussianFilter = new Matrix(x, y);
	float sum = 0.0;
	int xx = (x - 1) / 2;
	int yy = (y - 1) / 2;

	for (int i = 0; i<x; i++)
	{
		for (int j = 0; j<y; j++)
		{
			int ii = (i - xx);
			int jj = (j - yy);
			float r = sqrt(ii*ii + jj * jj);
			GaussianFilter->access(i, j) = exp(-(r*r) / (2 * sigma*sigma)) / (2 * 3.14159265358979323846*sigma*sigma);
			sum += GaussianFilter->access(i, j);
		}
	}

	for (int i = 0; i<x; i++)
	{
		for (int j = 0; j<y; j++)
		{
			GaussianFilter->access(i, j) = GaussianFilter->access(i, j) / sum;
		}
	}

	return GaussianFilter;
}


Matrix* gaussianBlur(Matrix* img, int filterSize, float sigma)
{
	int p = (filterSize - 1) / 2;

	Matrix* paddedImg = pad(img, p, 0);

	Matrix* filter = gausianFilter(filterSize, filterSize, sigma);


	Volume Img(1);
	Img[0] = paddedImg;

	Volume Filter(1);
	Filter[0] = filter;

	Matrix* newImg = convolve(Img, Filter, 1);


	delete filter;
	delete paddedImg;
	return newImg;
}

float norm_L1(Matrix* x)
{
	float sum = 0;
	for (int i = 0; i<x->Rows(); i++)
		for (int j = 0; j < x->Columns(); j++)
		{
			if (x->access(i, j) < 0)
				sum = sum - x->access(i, j);
			else
				sum = sum + x->access(i, j);
		}
	return sum;
}

void visualize(Matrix* img)
{
	Mat mat1;
	mat1 = Mat::ones(28, 28, CV_32FC1);

	for (int i = 0; i<28; i++)
		for (int j = 0; j < 28; j++)
		{
			mat1.at<float>(i, j) = img->access(i, j);
		}

	imshow("test", mat1);
	waitKey(0);
}


Matrix* elasticDistortion(Matrix* img, int filterSize, float sigma, float alpha)
{
	//displacement matrices dx & dy with uniform random values in range -1,1
	Matrix* dx = new Matrix(img->Rows(), img->Columns(), Random);
	Matrix* dy = new Matrix(img->Rows(), img->Columns(), Random);
	(*dx) = (*dx) / float(RAND_MAX);
	(*dy) = (*dy) / float(RAND_MAX);

	//apply gaussian filter to the displacements
	dx = gaussianBlur(dx, filterSize, sigma);
	dy = gaussianBlur(dy, filterSize, sigma);

	//normalizing dx & dy
	(*dx) = (*dx) / norm_L1(dx);
	(*dy) = (*dy) / norm_L1(dy);

	//alpha controls the intensity of deformation
	(*dx) = (*dx) * alpha;
	(*dy) = (*dy) * alpha;

	//apply displacements
	Matrix* distImg = new Matrix(img->Rows(), img->Columns());
	for (int i = 0; i<img->Rows(); i++)
		for (int j = 0; j < img->Columns(); j++)
		{
			int org_x = i - dx->access(i, j);
			int org_y = j - dy->access(i, j);
			if ((org_x > 0 && org_x < img->Columns()) && (org_y > 0 && org_y < img->Rows()))
				distImg->access(i, j) = img->access(org_x, org_y);
		}

	delete dx;
	delete dy;
	return distImg;
}