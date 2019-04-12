//#include "ConvBackProb.h
#include "NeuralNetwork.h"
////////////////////////////////////////////////////////////////////////////////////////////////
Matrix* create_mask_from_window(Matrix* x)
{
	int indexi = 0;
	int indexj = 0;
	float maximum = x->access(0, 0);
	for (int i = 0; i<x->Rows(); i++)
	{
		for (int j = 0; j<x->Columns(); j++)
		{
			if (x->access(i, j) > maximum)
			{
				maximum = x->access(i, j);
				indexi = i;
				indexj = j;
			}
		}
	}
	Matrix* mask = new Matrix(x->Rows(), x->Columns(), 0);
	mask->access(indexi, indexj) = 1;
	return mask;
}
/////////////////////////////////////////////////////////////////////////////////////////////////*
Matrix* distribute_value(float dz, int nh, int nw)
{
	float average = dz / (nh*nw);
	Matrix* a = new Matrix(nh, nw, average);
	return a;
}
////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::pool_backward(int f, int stride, Mode mode, int A_index)
{
	VectVolume dA = Conv_Grades[CharGen("dACP", A_index)];
	VectVolume Aprev = Conv_Cache[CharGen("AC", A_index)];

	int m = Aprev.size();
	int nc = dA[0].size();
	int nh = dA[0][0]->Rows();
	int nw = dA[0][0]->Columns();
	int nc_prev = Aprev[0].size();
	int nh_prev = Aprev[0][0]->Rows();
	int nw_prev = Aprev[0][0]->Columns();

	VectVolume dAprev(m, nc_prev, nh_prev, nw_prev);

	for (int i = 0; i<m; i++)
	{
		for (int c = 0; c < nc; c++)
		{
			for (int h = 0; h < nh; h++)
			{
				for (int w = 0; w < nw; w++)
				{
					int vert_start = h * stride;
					int vert_end = vert_start + f;
					int horz_start = w * stride;
					int horz_end = horz_start + f;

					if (mode == MAX)
					{
						Matrix* aprev_slice = Aprev[i][c]->SubMat(vert_start, horz_start, vert_end - 1, horz_end - 1);

						Matrix* tempMask = create_mask_from_window(aprev_slice);

						Matrix* mask = tempMask->mul(dA[i][c]->access(h, w));

						int k = 0; int kk = 0;
						for (int ii = vert_start; ii < vert_end; ii++)
						{
							for (int jj = horz_start; jj < horz_end; jj++)
							{
								dAprev[i][c]->access(ii, jj) = dAprev[i][c]->access(ii, jj) + mask->access(k, kk);
								kk++;
							}
							k++; kk = 0;
						}

						delete aprev_slice;
						delete tempMask;
						delete mask;
					}
					else if (mode == AVG)
					{
						float avg = dA[i][c]->access(h, w) / (f * f);

						for (int ii = vert_start; ii < vert_end; ii++)
						{
							for (int jj = horz_start; jj < horz_end; jj++)
							{
								dAprev[i][c]->access(ii, jj) = dAprev[i][c]->access(ii, jj) + avg;
							}
						}
					}
				}
			}
		}
	}
	Conv_Grades.put(CharGen("dAC", A_index), dAprev);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::updateparameters(float alpha, int iteration, int W_index)
{
	/* WARNING: when dealing with VectVolume, dellocating is done by deleting inner matrices using DELETE() */
	/* filters(W) & b will be in dictionary parameters..dW &db will be in dictionary grades */
	VectVolume WC = Conv_Weights[CharGen("WC", W_index)];
	VectVolume dWC = Conv_Grades[CharGen("dWC", W_index)];
	Matrix* bC = Conv_biases[CharGen("bC", W_index)];
	Matrix* dbC = Conv_dbiases[CharGen("dbC", W_index)];

	/* Temporary Ptrs */
	Matrix* Matptr1 = nullptr;
	Matrix* Matptr2 = nullptr;
	Matrix* Matptr3 = nullptr;
	Matrix* Matptr = nullptr;
	Matrix* temp1 = nullptr;
	Matrix* temp2 = nullptr;
	Matrix* temp3 = nullptr;
	Matrix* temp4 = nullptr;

	/*START OF GRADIENT DESCENT OPTIMIZER*/
	if (optimizer == GRADIENT_DESCENT)
	{
		for (int i = 0; i < WC.size(); i++)
		{
			for (int j = 0; j < WC[0].size(); j++)
			{
				//WC[i][j] = WC[i][j] - dWC[i][j] * alpha;
				Matptr1 = WC[i][j];
				Matptr2 = dWC[i][j]->mul(alpha);
				WC[i][j] = Matptr1->sub(Matptr2);
				delete Matptr1;
				delete Matptr2;
			}
		}

		//bC = bC - dbC * alpha;
		Matptr1 = bC;
		Matptr2 = dbC->mul(alpha);
		bC = Matptr1->sub(Matptr2);
		Conv_biases.replace(CharGen("bC", W_index), bC);
		delete Matptr1;
		delete Matptr2;
	}
	/*END OF GRADIENT DESCENT OPTIMIZER*/

	else if (optimizer == ADAM)
	{
		float beta1 = 0.9;
		float beta2 = 0.999;
		float epsilon = 1e-8;

		VectVolume VdwC = ADAM_dWC[CharGen("VdwC", W_index)];
		VectVolume SdwC = ADAM_dWC[CharGen("SdwC", W_index)];
		Matrix* VdbC = ADAM_dbC[CharGen("VdbC", W_index)];
		Matrix* SdbC = ADAM_dbC[CharGen("SdbC", W_index)];

		/* Updating VdwC, SdwC */
		for (int i = 0; i < VdwC.size(); i++)
		{
			for (int j = 0; j < VdwC[0].size(); j++)
			{
				//VdwC[i][j] = (VdwC[i][j] * (beta1 * momentum)) + (dWC[i][j] * (1 - beta1 * momentum));
				Matptr = VdwC[i][j];
				temp1 = VdwC[i][j]->mul(beta1 * momentum);
				temp2 = dWC[i][j]->mul(1 - beta1 * momentum);
				VdwC[i][j] = temp1->add(temp2);
				delete Matptr;
				delete temp1;
				delete temp2;
				
				//SdwC[i][j] = (SdwC[i][j] * beta2) + (dWC[i][j].square() * (1 - beta2));
				Matptr = SdwC[i][j];
				temp1 = SdwC[i][j]->mul(beta2);
				temp2 = dWC[i][j]->SQUARE();
				temp3 = temp2->mul(1 - beta2);
				SdwC[i][j] = temp1->add(temp3);
				delete Matptr;
				delete temp1;
				delete temp2;
				delete temp3;
			}
		}

		/* Updating VdbC, SdbC */
		//VdbC = (VdbC * (beta1 * momentum)) + (dbC * (1 - beta1 * momentum));
		Matptr = VdbC;
		temp1 = VdbC->mul(beta1 * momentum);
		temp2 = dbC->mul(1 - beta1 * momentum);
		VdbC = temp1->add(temp2);
		delete Matptr;
		delete temp1;
		delete temp2;
		ADAM_dbC.replace(CharGen("VdbC", W_index), VdbC);

		//SdbC = (SdbC * beta2) + (dbC.square() * (1 - beta2));
		Matptr = SdbC;
		temp1 = SdbC->mul(beta2);
		temp2 = dbC->SQUARE();
		temp3 = temp2->mul(1 - beta2);
		SdbC = temp1->add(temp3);
		delete Matptr;
		delete temp1;
		delete temp2;
		delete temp3;
		ADAM_dbC.replace(CharGen("SdbC", W_index), SdbC);

		/* Correcting first iterations */ 
		VectVolume VdwC_corr(WC.size(), WC[0].size());
		VectVolume SdwC_corr(WC.size(), WC[0].size());
		Matrix* VdbC_corr = nullptr;
		Matrix* SdbC_corr = nullptr;
		for (int i = 0; i < VdwC_corr.size(); i++)
		{
			for (int j = 0; j < VdwC_corr[0].size(); j++)
			{
				VdwC_corr[i][j] = VdwC[i][j]->div(1 - pow(beta1, iteration + 1));
				SdwC_corr[i][j] = SdwC[i][j]->div(1 - pow(beta2, iteration + 1));
			}
		}
		VdbC_corr = VdbC->div(1 - pow(beta1, iteration + 1));
		SdbC_corr = SdbC->div(1 - pow(beta2, iteration + 1));

		/* Updating Parameters */
		for (int i = 0; i < WC.size(); i++)
		{
			for (int j = 0; j < WC[0].size(); j++)
			{
				//temp = VdwC[i][j]_corr / (SdwC[i][j]_corr.Sqrt() + epsilon);
				//WC[i][j] = WC[i][j] - temp * alpha;
				Matptr = WC[i][j];
				temp1 = SdwC_corr[i][j]->SQRT();
				temp2 = temp1->add(epsilon);
				temp3 = VdwC_corr[i][j]->div(temp2);
				temp4 = temp3->mul(alpha);
				WC[i][j] = WC[i][j]->sub(temp4);
				delete Matptr;
				delete temp1;
				delete temp2;
				delete temp3;
				delete temp4;

			}
		}
		//Matrix temp = VdbC_corr / (SdbC_corr.Sqrt() + epsilon);
		//Matrix buC = bC - temp * alpha;
		Matptr = bC;
		temp1 = SdbC_corr->SQRT();
		temp2 = temp1->add(epsilon);
		temp3 = VdbC_corr->div(temp2);
		temp4 = temp3->mul(alpha);
		bC = bC->sub(temp4);
		delete Matptr;
		delete temp1;
		delete temp2;
		delete temp3;
		delete temp4;
		Conv_biases.replace(CharGen("bC", W_index), bC);



		VdwC_corr.DELETE();
		SdwC_corr.DELETE();
		delete VdbC_corr;
		delete SdbC_corr;

	}


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void NeuralNetwork::ConvBackwardOptimized(int stride, int A_index, ActivationType activation)
{
	string str;
	if (A_index == 1)
		str = "AC";
	else
		str = "ACP";

	VectVolume ACprev = Conv_Cache[CharGen(str, A_index - 1)];
	VectVolume ZC = Conv_Cache[CharGen("ZC", A_index)];
	VectVolume dAC = Conv_Grades[CharGen("dAC", A_index)];
	VectVolume WC = Conv_Weights[CharGen("WC", A_index)];
	VectVolume dZC = Calc_dZC(ZC, dAC, activation);

	int m = dZC.size();
	int n_C = dZC[0].size();
	int n_H = dZC[0][0]->Rows();
	int n_W = dZC[0][0]->Columns();
	int f = WC[0][0]->Rows();

	VectVolume dACPprev(m, ACprev[0].size(), ACprev[0][0]->Rows(), ACprev[0][0]->Columns());
	VectVolume dWC(WC.size(), WC[0].size(), WC[0][0]->Rows(), WC[0][0]->Columns());
	Matrix* dbC = new Matrix(n_C, 1);

	Matrix* Matptr1 = nullptr;
	Matrix* Matptr2 = nullptr;
	Matrix* Matptr3 = nullptr;

	for (int i = 0; i < m; i++)
	{
		Volume a_prev = ACprev[i];
		for (int c = 0; c < n_C; c++)
		{
			for (int h = 0; h < n_H; h++)
			{
				for (int w = 0; w < n_W; w++)
				{
					int vert_start = h;
					int vert_end = h + f;
					int horiz_start = w;
					int horiz_end = w + f;
					int dWC_channels = dWC[0].size();

					if (A_index != 1)
						for (int ii = 0; ii < dWC_channels; ii++)
							for (int jj = vert_start, jjW = 0; jj < vert_end; jj++, jjW++)
								for (int kk = horiz_start, kkW = 0; kk < horiz_end; kk++, kkW++)
									dACPprev[i][ii]->access(jj, kk) = dACPprev[i][ii]->access(jj, kk) + WC[c][ii]->access(jjW, kkW) * dZC[i][c]->access(h, w);

					for (int ii = 0; ii < dWC_channels; ii++)
					{
						//dWC[c][ii] = dWC[c][ii] + a_prev[ii]->SubMat(vert_start, horiz_start, vert_end - 1, horiz_end - 1) * dZC[i][c]->access(h, w);
						Matptr1 = a_prev[ii]->SubMat(vert_start, horiz_start, vert_end - 1, horiz_end - 1);
						Matptr2 = Matptr1->mul(dZC[i][c]->access(h, w));
						Matptr3 = dWC[c][ii];
						dWC[c][ii] = Matptr3->add(Matptr2);
						delete Matptr1;
						delete Matptr2;
						delete Matptr3;
					}

					//dbC = dbC + dZC[i][c].access(h, w);
					Matptr1 = dbC;
					dbC = Matptr1->add(dZC[i][c]->access(h, w));
					delete Matptr1;
				}
			}
		}
	}

	for (int i = 0; i < dWC.size(); i++)
		for (int j = 0; j < dWC[0].size(); j++)
		{
			//dWC[i][j] = dWC[i][j] / m;
			Matptr1 = dWC[i][j];
			dWC[i][j] = Matptr1->div(m);
			delete Matptr1;
		}

	//dbC = dbC / m;
	Matptr1 = dbC;
	dbC = Matptr1->div(m);
	delete Matptr1;

	dZC.DELETE();

	if (A_index != 1)
		Conv_Grades.put(CharGen("dACP", A_index - 1), dACPprev);
	else
		dACPprev.DELETE();


	Conv_Grades.put(CharGen("dWC", A_index), dWC);
	Conv_dbiases.put(CharGen("dbC", A_index), dbC);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
VectVolume Calc_dZC(VectVolume ZC, VectVolume dAC, ActivationType activation)
{
	int m = dAC.size();
	int numOfVolumes = dAC[0].size();
	VectVolume dZC(m, numOfVolumes);
	Matrix* dactiv_z = nullptr;

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < numOfVolumes; j++)
		{
			dactiv_z = dactiv(ZC[i][j], activation);

			dZC[i][j] = dAC[i][j]->mul(dactiv_z);

			delete dactiv_z;
		}
	}
	return dZC;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*void NeuralNetwork::ConvBackward(int stride,int A_index, string activation)
{
/* string str;
if(A_index == 1)
str = "AC";
else
str = "ACP";
VectVolume ACprev = Conv_Cache[CharGen(str, A_index - 1)];
VectVolume ZC = Conv_Cache[CharGen("ZC", A_index)];
VectVolume dAC = Conv_Grades[CharGen("dAC", A_index)];
VectVolume filters = Conv_Weights[CharGen("WC", A_index)];
VectVolume dZC = Calc_dZC(ZC, dAC, activation);

VectVolume dWC = FilterGrades(ACprev, dZC);
Matrix dbC = biasGrades(dZC);
if(A_index != 1)
{
VectVolume dACPprev = FullConvolution(dZC, filters);
Conv_Grades.put(CharGen("dACP", A_index - 1), dACPprev);
}
Conv_Grades.put(CharGen("dWC", A_index), dWC);
Conv_dbiases.put(CharGen("dbC", A_index), dbC);
}*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*VectVolume FullConvolution(VectVolume dZC, VectVolume filters)
{
//Rotate filters 180 degrees and rearrange them
VectVolume RotatedFilters(filters.size());
RotatedFilters = RotateAllVolumes(filters);
RotatedFilters = RearrangeFilters(RotatedFilters);

//Determine the amount of padding required for dZC (p = filter size - 1)
int p = RotatedFilters[0][0].Rows() - 1;
VectVolume Padded_dZC(dZC.size());
Padded_dZC = PadAllVolumes(dZC, p, 0);

int numOfFilteres = RotatedFilters.size();
int m = Padded_dZC.size();
VectVolume A(m);
for (int i = 0; i < m; i++)
A[i].resize(numOfFilteres);

for (int i = 0; i<m; i++)
for (int j = 0; j<numOfFilteres; j++)
{
//convolve the ith activations with the jth filter to produce z which is pointer to (nh,nw) matrix
Matrix* z = convolve(Padded_dZC[i], RotatedFilters[j], 1);
//z is pointer to the output of convolution, push it into the volume A[i]
A[i](j) = z;
}
return A;
}*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*VectVolume FilterGrades(VectVolume ACprev, VectVolume dZC)
{
//Modify dimensions
VectVolume dZC_Modified = RearrangeFilters(dZC);
VectVolume ACprev_Modified = RearrangeFilters(ACprev);

int numOfFilters = dZC[0].size();
int m = dZC.size();
int FilterSize = ACprev[0].size();
int f = ACprev[0][0].Rows() - dZC[0][0].Rows() + 1;

//Final result
VectVolume Result(numOfFilters, FilterSize, f, f);

//Cycle over all training examples
for (int i = 0; i<m; i++)
{
VectVolume A(numOfFilters, FilterSize, f, f);

for(int k = 0; k<FilterSize; k++)
{
for (int j = 0; j<numOfFilters; j++)
{
//convolve the ith activations with the jth filter to produce z which is pointer to (nh,nw) matrix
Matrix* z = convolve(ACprev_Modified[k], dZC_Modified[j], 1);
//z is pointer to the output of convolution, push it into the volume A[i]
A[j](k) = z;
}
}

//Increament every training example
for(int k = 0 ; k<numOfFilters; k++)
{
for(int j = 0; j<FilterSize; j++)
{
Result[k][j] = Result[k][j] + A[k][j];
}
}

}

return Result;
}*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*Matrix biasGrades(VectVolume dZC)
{
int m = dZC.size();
int nc = dZC[0].size();
Matrix db(nc, 1, 0);
for (int i = 0; i < nc; i++)
{
int temp = 0;
for(int k = 0; k < m; k++)
{
Matrix* dz = dZC[k](i);
temp += dz->sumall();
}
db.access(i, 0) = temp;
}
return db;
}*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*VectVolume RearrangeFilters(VectVolume filter)
{
VectVolume Result(filter[0].size(),filter.size());
for (int i = 0; i < filter[0].size(); i++)
for (int j = 0; j < filter.size(); j++)
Result[i](j) = filter[j](i);

return Result;
}*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*VectVolume PadAllVolumes(VectVolume Original, int p, int value)
{
VectVolume Result(Original.size());
for (int i = 0; i<Original.size(); i++)
Result[i] = PadVolume(Original[i], p, value);
return Result;
}*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*Volume PadVolume(Volume Original, int p, int value)
{
Volume Result(Original.size());
for (int i = 0; i<Original.size(); i++)
{
Matrix* PaddedMat = new Matrix(Original[i].Rows(), Original[i].Columns(), 0);
PaddedMat = pad(Original(i), p, value);
Result(i) = new Matrix(Original[i].Rows(), Original[i].Columns(), 0);
Result(i) = PaddedMat;
}
return Result;
}*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*VectVolume RotateAllVolumes(VectVolume filters)
{
VectVolume RotatedFilters(filters.size());
for (int i = 0; i<filters.size(); i++)
RotatedFilters[i] = RotateVolume(filters[i]);

return RotatedFilters;
}*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*Volume RotateVolume(Volume filter)
{
Volume Result(filter.size());
for (int i = 0; i<filter.size(); i++)
{
Result(i) = new Matrix(filter[i].Rows(), filter[i].Columns());
Result[i] = filter[i].Rotate180();
}

return Result;
}*/
