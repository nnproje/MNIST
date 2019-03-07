#pragma once
#include <fstream>
#include "Volume.h"

#ifndef dataset_H_INCLUDED
#define dataset_H_INCLUDED

typedef matrix<float> Matrix;

int LittleEndian(uint32_t ii)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = ii & 255;
	ch2 = (ii >> 8) & 255;
	ch3 = (ii >> 16) & 255;
	ch4 = (ii >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + int(ch4);
}

void get_dataset_2D(Volume& X, Matrix &Y,const char*Xdir, const char*Ydir)
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


	for (int k = 0; k<ImagesNum1; k++)
	{
		X[k] = new Matrix(RowsNum, ColumnsNum, 0);
		for (int i = 0; i < RowsNum; i++)
			for (int j = 0; j < ColumnsNum; j++)
			{
				unsigned char temp1;
				pixels.read((char*)&temp1, 1);
				X[k]->access(i,j)= temp1 / 255.0;
			}

		unsigned char temp2;
		labels.read((char*)&temp2, 1);
		Y.access(int(temp2), k) = 1;
	}
}

void Shuffle(Volume& X, Matrix& Y)
{
	void SWAP(Matrix& MAT, int i, int k);
	void SWAP(Volume& Vol, int i, int k);

	for (int i = 0; i<Y.Columns(); i++)
	{
		int s = rand() % Y.Columns();
		SWAP(X, i, s);
		SWAP(Y, i, s);
	}

}

void SWAP(Matrix& MAT, int i, int k)
{
	Matrix temp(MAT.Rows(), 1);
	for (int j = 0; j<MAT.Rows(); j++)
	{
		temp.access(j, 0) = MAT.access(j, i);
		MAT.access(j, i) = MAT.access(j, k);
		MAT.access(j, k) = temp.access(j, 0);
	}
}

void SWAP(Volume& Vol, int i, int k)
{
	Matrix* temp;
	temp = Vol[i];
	Vol[i] = Vol[k];
	Vol[k] = temp;
}



void DevSet(Matrix& X, Matrix& Y, Matrix& X_dev, Matrix& Y_dev)
{
	for (int j = 0; j<10000; j++)
		for (int i = 0; i<X.Rows(); i++)
			X_dev.access(i, j) = X.access(i, j);

	for (int j = 0; j<10000; j++)
		for (int i = 0; i<Y.Rows(); i++)
			Y_dev.access(i, j) = Y.access(i, j);
}


void normalize(Matrix& X, Matrix& X_test, Matrix& Y, Matrix& Y_test)
{
	Matrix mean(784, 1);
	Matrix variance(784, 1);
	float eps = 1e-8;


	mean = (X.sum("column") + X_test.sum("column")) / (X.Columns() + X_test.Columns());
	X = X - mean;
	X_test = X_test - mean;
	variance = (X.square().sum("column") + X_test.square().sum("column")) / (X.Columns() + X_test.Columns());
	variance = variance + eps;
	X = X / (variance.Sqrt());
	X_test = X_test / (variance.Sqrt());

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

#endif // dataset_H_INCLUDED
