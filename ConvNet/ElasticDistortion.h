#pragma once
#include "ConvNet.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* pad(Matrix* img, int p, float value);
Matrix* gausianFilter(int x, int y, float sigma);
Matrix* gaussianBlur(Matrix* img, int filterSize, float sigma);
Matrix* convolve(Volume& Aprev, Volume& filter, int s);
float norm_L1(Matrix* x);
Matrix* elasticDistortion(Matrix* img, int filterSize, float sigma, float alpha);
void enlarge2D(Volume& X, Matrix& Y, int enlargeFact);
Matrix* to_1D(Volume& X_2D);
Matrix* enlarge1D(Volume& X, Matrix& Y, int enlargeFact);
void visualize(Matrix* img);
/////////////////////////////////////////////////////////////////////////////////////////////////


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
/////////////////////////////////////////////////////////////////////////////////////////////////


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
/////////////////////////////////////////////////////////////////////////////////////////////////

/*returns a gaussian filter (x,y) with standard deviation sigma*/
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
/////////////////////////////////////////////////////////////////////////////////////////////////

/*perform a gaussian blur to a 2D img*/
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
/////////////////////////////////////////////////////////////////////////////////////////////////

/*L1 norm to matrix x*/
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
/////////////////////////////////////////////////////////////////////////////////////////////////


/*performs elastric distortion on an img*/
Matrix* elasticDistortion(Matrix* img, int filterSize, float sigma, float alpha)
{
	//displacement matrices dx & dy with uniform random values in range -1,1
	Matrix* dx = new Matrix(img->Rows(), img->Columns(), Random);
	Matrix* dy = new Matrix(img->Rows(), img->Columns(), Random);
	Matrix* dx_old = dx;
	Matrix* dy_old = dy;
	(*dx) = (*dx) / float(RAND_MAX);
	(*dy) = (*dy) / float(RAND_MAX);
	(*dx) = (*dx) * 2;
	(*dy) = (*dy) * 2;
	(*dx) = (*dx) - 1;
	(*dy) = (*dy) - 1;


	//apply gaussian filter to the displacements
	dx = gaussianBlur(dx, filterSize, sigma);
	dy = gaussianBlur(dy, filterSize, sigma);


	//normalizing dx & dy
	(*dx) = (*dx) / norm_L1(dx);
	(*dy) = (*dy) / norm_L1(dy);




	//alpha controls the intensity of deformation
	(*dx) = (*dx) * alpha;
	(*dy) = (*dy) * alpha;

	//apply displacements, we assume the top left corner is position (0,0) and bottom right corner is position (img->Rows()-1 , img->Columns()-1)
	Matrix* distImg = new Matrix(img->Rows(), img->Columns());

	for (int i = 0; i<img->Rows(); i++)
		for (int j = 0; j < img->Columns(); j++)
		{
			//the position of the new pixel value
			float x = i + dx->access(i, j); //x=0+1.75=1.75
			float y = j + dy->access(i, j); //y=0+0.5=0.5

											//if the new position is outside the image put 0 in it
			if (x < 0 || y < 0)
			{
				distImg->access(i, j) = 0;
			}
			else
			{
				//applying bilinear interpolation to the unit square with (xmin,ymin) (xmax,ymax), (xdis,ydis) is a point in the unit square
				int xmin = int(x);
				int xmax = xmin + 1;
				int ymin = int(y);
				int ymax = ymin + 1;
				float xdis = x - int(x);
				float ydis = y - int(y);

				if (xmin >= img->Columns() || xmax >= img->Columns() || ymin >= img->Rows() || ymax >= img->Rows())
				{
					distImg->access(i, j) = 0;
				}
				else
				{
					//getting the pixels of current square
					float topLeft = img->access(xmin, ymin);
					float bottomLeft = img->access(xmin, ymax);
					float topRight = img->access(xmax, ymin);
					float bottomRight = img->access(xmax, ymax);

					//horizontal interpolation
					float horizTop = topLeft + xdis * (topRight - topLeft);
					if (horizTop < 0)
						horizTop = 0;
					float horizBottom = bottomLeft + xdis * (bottomRight - bottomLeft);
					if (horizBottom < 0)
						horizBottom = 0;

					//vertical interpolation
					float newPixel = horizTop + ydis * (horizBottom - horizTop);
					if (newPixel >= 0)
						distImg->access(i, j) = newPixel;
					else
						distImg->access(i, j) = 0;
				}
			}
		}

	delete dx;
	delete dy;
	delete dx_old;
	delete dy_old;
	return distImg;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/*enlarge the 2D dataset X with a factor enlargeFact*/
void enlarge2D(Volume& X, Matrix& Y, int enlargeFact)
{

	int SIZE = 10;
	int X_count = 10;

	Matrix Y_new(Y.Rows(), Y.Columns() + Y.Columns()*enlargeFact);
	for (int k = 0; k < enlargeFact + 1; k++)
	{
		for (int i = 0; i < Y.Rows(); i++)
		{
			for (int j = 0; j < Y.Columns(); j++)
			{
				Y_new.access(i, k * Y.Columns() + j) = Y.access(i, j);
			}
		}
	}
	Y = Y_new;

	cout << "got Y_new" << endl;

	if (enlargeFact > 0)
	{
		Matrix* newImg = NULL;
		for (int i = 0; i < SIZE; i++)
		{
			newImg = elasticDistortion(X[i], 17, 10, 600);
			X[X_count] = newImg;
			X_count++;

			if (i % 500 == 0)
				cout << setw(7) << X_count << " ";

		}
		enlargeFact--;
	}

	cout << "===================>>got first distorted<======================" << endl;

	if (enlargeFact > 0)
	{
		Matrix* newImg = NULL;
		for (int i = 0; i < SIZE; i++)
		{
			newImg = elasticDistortion(X[i], 19, 10, 600);
			X[X_count] = newImg;
			X_count++;

			if (i % 500 == 0)
				cout << setw(7) << X_count << " ";
		}
		enlargeFact--;
	}

	cout << endl << "===================>>got second distorted<======================" << endl;

	if (enlargeFact > 0)
	{
		Matrix* newImg = NULL;
		for (int i = 0; i < SIZE; i++)
		{
			newImg = elasticDistortion(X[i], 21, 10, 600);
			X[X_count] = newImg;
			X_count++;

			if (i % 500 == 0)
				cout << setw(7) << X_count << " ";
		}
		enlargeFact--;
	}

	cout << endl << "===================>>got third distorted<======================" << endl;

	if (enlargeFact > 0)
	{
		Matrix* newImg = NULL;
		for (int i = 0; i < SIZE; i++)
		{
			newImg = elasticDistortion(X[i], 23, 10, 600);
			X[X_count] = newImg;
			X_count++;
		}
		enlargeFact--;
	}

	if (enlargeFact > 0)
	{
		Matrix* newImg = NULL;
		for (int i = 0; i < SIZE; i++)
		{
			newImg = elasticDistortion(X[i], 25, 10, 600);
			X[X_count] = newImg;
			X_count++;
		}
		enlargeFact--;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/*enlarge the 1D dataset X with a factor enlargeFact*/
Matrix* enlarge1D(Volume& X, Matrix& Y, int enlargeFact)
{
	enlarge2D(X, Y, enlargeFact);
	cout << "=================>END OF enlarge2D<=================" << endl;

	Matrix* X_1D = to_1D(X);
	cout << "=================>END OF to_1D<=================" << endl;

	return X_1D;
}
/////////////////////////////////////////////////////////////////////////////////////////////////

/*converts a volume(vector of 2D images) into a Matrix(vector of 1D -flat- images)*/
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
/////////////////////////////////////////////////////////////////////////////////////////////////

/*display an image with pixel in img*/
void visualize(Matrix* img)
{
	Mat mat1;
	mat1 = Mat::ones(img->Rows(), img->Columns(), CV_32FC1);

	for (int i = 0; i<img->Rows(); i++)
		for (int j = 0; j < img->Columns(); j++)
		{
			mat1.at<float>(i, j) = img->access(i, j);
		}

	imshow("test", mat1);
	waitKey(0);
}
/////////////////////////////////////////////////////////////////////////////////////////////////




