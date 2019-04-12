#include "DataSet.h"
#include "ConvFeedForward.h"
///////////////////////////////////////////////////////////////////////////////
////////////////////////////GET DATASET FROM HARD DISK/////////////////////////
///////////////////////////////////////////////////////////////////////////////
int LittleEndian(uint32_t ii)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=ii&255;
    ch2=(ii>>8)&255;
    ch3=(ii>>16)&255;
    ch4=(ii>>24)&255;
   return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+int(ch4);
}
///////////////////////////////////////////////////////////////////////////////
void get_dataset(Matrix* X,Matrix* Y,const char*Xdir,const char*Ydir,int EXAMPLES)
{
    //X (28*28,ImagesNUM)
    //Y (10,ImagesNUM)
    ifstream pixels(Xdir,ios::binary);
    uint32_t magicNum1;
    uint32_t ImagesNum1;
    uint32_t RowsNum;
    uint32_t ColumnsNum;
    pixels.read((char*)&magicNum1,sizeof(magicNum1));
    magicNum1=LittleEndian(magicNum1);
    pixels.read((char*)&ImagesNum1,sizeof(ImagesNum1));
    ImagesNum1=LittleEndian(ImagesNum1);
    cout<<"Number of images of Data = "<<ImagesNum1<<endl;
    pixels.read((char*)&RowsNum,sizeof(RowsNum));
    RowsNum=LittleEndian(RowsNum);
    cout<<"Number of Rows = "<<RowsNum<<endl;
    pixels.read((char*)&ColumnsNum,sizeof(ColumnsNum));
    ColumnsNum=LittleEndian(ColumnsNum);
    cout<<"Number of Columns = "<<ColumnsNum<<endl;

    ifstream labels(Ydir,ios::binary);
    uint32_t magicNum2;
    uint32_t ImagesNum2;
    labels.read((char*)&magicNum2,sizeof(magicNum2));
    magicNum2=LittleEndian(magicNum2);
    labels.read((char*)&ImagesNum2,sizeof(ImagesNum2));
    ImagesNum2=LittleEndian(ImagesNum2);


    for (uint32_t j=0;j<EXAMPLES;j++)
    {
        for(uint32_t i=0;i<RowsNum*ColumnsNum;i++)
        {
           unsigned char temp1;
           pixels.read((char*)&temp1,1);
           X->access(i,j)=temp1/255.0;
        }

        unsigned char temp2;
        labels.read((char*)&temp2,1);
        Y->access(int(temp2),j)=1;
    }

}
///////////////////////////////////////////////////////////////////////////////
void Shuffle(Matrix* X, Matrix* Y)
{
	void SWAP(Matrix* MAT, int i, int k);

	for (int i = 0; i<X->Columns(); i++)
	{
		int s = rand() % X->Columns();
		SWAP(X, i, s);
		SWAP(Y, i, s);
	}

}
///////////////////////////////////////////////////////////////////////////////
int get_dataset_2D(Volume& X, Matrix* Y,const char*Xdir,const  char*Ydir,int EXAMPLES)
{
	//X Volume(60000)
	//Y (10,ImagesNUM)
	ifstream pixels(Xdir, ios::binary);
	uint32_t magicNum1;
	uint32_t ImagesNum1;
	uint32_t RowsNum;
	uint32_t ColumnsNum;
	pixels.read((char*)&magicNum1, sizeof(magicNum1));
	magicNum1 = LittleEndian(magicNum1);
	cout << magicNum1 << endl;
	pixels.read((char*)&ImagesNum1, sizeof(ImagesNum1));
	ImagesNum1 = LittleEndian(ImagesNum1);
	cout << ImagesNum1 << endl;
	pixels.read((char*)&RowsNum, sizeof(RowsNum));
	RowsNum = LittleEndian(RowsNum);
	cout << RowsNum << endl;
	pixels.read((char*)&ColumnsNum, sizeof(ColumnsNum));
	ColumnsNum = LittleEndian(ColumnsNum);
	cout << ColumnsNum << endl;

	ifstream labels(Ydir, ios::binary);
	uint32_t magicNum2;
	uint32_t ImagesNum2;
	labels.read((char*)&magicNum2, sizeof(magicNum2));
	magicNum2 = LittleEndian(magicNum2);
	cout << magicNum2 << endl;
	labels.read((char*)&ImagesNum2, sizeof(ImagesNum2));
	ImagesNum2 = LittleEndian(ImagesNum2);
	cout << ImagesNum2 << endl;


	for (uint32_t k = 0; k<10; k++)
	{
		X[k] = new Matrix(RowsNum, ColumnsNum, 0);
		for (uint32_t i = 0; i < RowsNum; i++)
			for (uint32_t j = 0; j < ColumnsNum; j++)
			{
				unsigned char temp1;
				pixels.read((char*)&temp1, 1);
				X[k]->access(i,j)= temp1 / 255.0;
			}

		unsigned char temp2;
		labels.read((char*)&temp2, 1);
		Y->access(int(temp2), k) = 1;
	}
	return 0;
}
///////////////////////////////////////////////////////////////////////////////
void Shuffle(Volume& X, Matrix* Y)
{
	void SWAP(Matrix* MAT, int i, int k);
	void SWAP(Volume& Vol, int i, int k);

	for (int i = 0; i<Y->Columns(); i++)
	{
		int s = rand() % Y->Columns();
		SWAP(X, i, s);
		SWAP(Y, i, s);
	}

}
///////////////////////////////////////////////////////////////////////////////
void SWAP(Matrix* MAT, int i, int k)
{
	Matrix* temp = new Matrix(MAT->Rows(), 1);
	for (int j = 0; j<MAT->Rows(); j++)
	{
		temp->access(j, 0) = MAT->access(j, i);
		MAT->access(j, i) = MAT->access(j, k);
		MAT->access(j, k) = temp->access(j, 0);
	}
	delete temp;
}
///////////////////////////////////////////////////////////////////////////////
void SWAP(Volume& Vol, int i, int k)
{
	Matrix* temp;
	temp = Vol[i];
	Vol[i] = Vol[k];
	Vol[k] = temp;
}
///////////////////////////////////////////////////////////////////////////////
void DevSet(Matrix* X, Matrix* Y, Matrix* X_dev, Matrix* Y_dev, int DEV)
{
    for(int j=0; j<DEV; j++)
        for(int i=0; i<X->Rows(); i++)
            X_dev->access(i,j)=X->access(i,j);

    for(int j=0; j<DEV; j++)
        for(int i=0; i<Y->Rows(); i++)
            Y_dev->access(i,j)=Y->access(i,j);
}
///////////////////////////////////////////////////////////////////////////////
void normalize(Matrix& X, Matrix& X_test, Matrix& Y, Matrix& Y_test)
{
    Matrix mean(784,1);
    Matrix variance(784,1);
    float eps=1e-8;


    mean=(X.sum("column")+X_test.sum("column"))/(X.Columns()+X_test.Columns());
    X=X-mean;
    X_test=X_test-mean;
    variance=(X.square().sum("column")+X_test.square().sum("column"))/(X.Columns()+X_test.Columns());
    variance=variance+eps;
    X=X/(variance.Sqrt());
    X_test=X_test/(variance.Sqrt());

    /*for(int i=0; i<Y.Rows(); i++)
        for(int j=0; j<Y.Columns(); j++)
        {
            if(Y.access(i,j)==0)
                Y.access(i,j)=-1;
        }


   for(int i=0; i<Y_test.Rows(); i++)
        for(int j=0; j<Y_test.Columns(); j++)
        {
            if(Y_test.access(i,j)==0)
                Y_test.access(i,j)=-1;
        }*/
}
///////////////////////////////////////////////////////////////////////////////
void DELETE(Volume& X)
{
    for(int i=0; i<X.size(); i++)
    {
        delete X[i];
    }
   X.DELETE();
}
///////////////////////////////////////////////////////////////////////////////
//////////////////////////ELASTIC DISTORTION///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

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

