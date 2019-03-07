#ifndef BACKPROB2_H_INCLUDED
#define BACKPROB2_H_INCLUDED

#include "Volume.h"
typedef matrix<float> Matrix;

Matrix create_mask_from_window(Matrix & x)
{
   int indexi=0; int indexj=0;
   float maximum=x.access(0,0);
   for(int i=0;i<x.Rows();i++)
   {
       for(int j=0;j<x.Columns();j++)
       {
           if(x.access(i,j)>maximum)
           {
              maximum=x.access(i,j);
              indexi=i;
              indexj=j;
           }
       }
   }
   Matrix mask (x.Rows(),x.Columns(),0);
   mask.access(indexi,indexj)=1;
   return mask;
}
////////////////////////////////////////////////////////////////////////////////////////////////
Matrix distribute_value(float dz,int nh,int nw)
{
    float average=dz/(nh*nw);
    Matrix a(nh,nw,average);
    return a;
}
////////////////////////////////////////////////////////////////////////////////////////////////
vector<Volume> pool_backward(vector<Volume>& dA,vector<Volume> & Aprev,int f,int stride, string mode)
{
    //stride,f may not need to be passed in class-f,stride will be in dictionary hparameters
    int m=Aprev.size();
    vector<Volume> dAprev(m);
    int nc=dA[0].size();
    int nh=dA[0][0]->Rows();
    int nw=dA[0][0]->Columns();

    int nc_prev=Aprev[0].size();
    int nh_prev=Aprev[0][0]->Rows();
    int nw_prev=Aprev[0][0]->Columns();

    for (int i = 0; i < m; i++)
	{
		dAprev[i].resize(nc_prev);
		for(int j=0;j<nc_prev;j++)
        {
            dAprev[i][j]=new Matrix(nh_prev,nw_prev,0);
        }
	}
	for(int i=0;i<m;i++)
    {
        Volume aprev=Aprev[i];
         for(int c=0;c<nc;c++)
            {
                for(int h=0;h<nh;h++)
                    {
                        for(int w=0;w<nw;w++)
                            {
                                int vert_start = h * stride;
                                int vert_end = vert_start + f;
                                int horz_start = w * stride;
                                int horz_end = horz_start + f;
                                if(mode=="max")
                                  {
									  Matrix aprev_slice=(*aprev[c])(vert_start, horz_start, vert_end - 1, horz_end - 1);
                                      Matrix mask=create_mask_from_window(aprev_slice)* (dA[i][c]->access(h,w));
                                      int k=0; int kk=0;
                                      for(int ii=vert_start;ii<vert_end;ii++)
                                      {
                                          for(int jj=horz_start;jj<horz_end;jj++)
                                          {
                                              dAprev[i][c]->access(ii,jj)= dAprev[i][c]->access(ii,jj)+mask.access(k,kk);
                                              kk++;
                                          }
                                          k++; kk=0;
                                      }
                                  }
                                 else if(mode=="average")
                                 {
                                     float avg=dA[i][c]->access(h,w)/(f*f);
                                      for(int ii=vert_start;ii<vert_end;ii++)
                                      {
                                          for(int jj=horz_start;jj<horz_end;jj++)
                                          {
                                              dAprev[i][c]->access(ii,jj)= dAprev[i][c]->access(ii,jj)+avg;
                                          }
                                      }
                                 }
                            }
                    }
            }

    }
    return dAprev;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void updateparameters (float alpha,int iteration,vector<Volume>& W, Volume& b,vector<Volume>& dW, Volume &db)
{
    //filters(W)& b will be in dictionary parameters..dW &db will be in dictionary grades

    /*START OF GRADIENT DESCENT OPTIMIZER*/
    //if (optimizer == "GradientDescent")
      {
        for(int i=0;i<W.size();i++)
        {
            for(int j=0;j<W[0].size();j++)
            {
                *(W[i][j])=*(W[i][j])-(*(dW[i][j]))*alpha;
                if(i==0)
                *b[j]=*b[j]-(*db[j])*alpha;
            }
        }
      }
    /*END OF GRADIENT DESCENT OPTIMIZER*/



    /*START OF ADAM OPTIMIZER*/
     /*
    else if(optimizer == "Adam")
    {
        float beta1 = 0.9;
        float beta2 = 0.999;
        float epsilon = 1e-8;
        /*Getting variables from dictionaries*/
        /*
        vector<volume> Vdw(W.size());
        vector<volume> Sdw(W.size());
        Volume Vdb(W[0].size());
        Volume Sdb(W[0].size());
        for(int i=0;i<W.size();i++)
        {
                Vdw[i].resize(W[0].size());
                Sdw[i].resize(W[0].size());
        }
        Vdw = grades[CharGen("Vdw", 1)];  //modify the number
        Sdw = grades[CharGen("Sdw", 1)];
        Vdb = grades[CharGen("Vdb",  1)];
        Sdb = grades[CharGen("Sdb",  1)];
        /*Updating Vdw, Vdb, Sdw, Sdb*/
        /*
        for(int i=0;i<W.size();i++)
        {
            for(int j=0;j<W[0].size();j++)
            {
                if(i==0)
                {   /*Updating Vdb, Sdb*/
                    //*Vdb[j] = ((*Vdb[j]) * (beta1 * momentum)) + ((*db[j]) * (1-beta1 * momentum));
                    //*Sdb[j] = ((*Sdb[j]) * beta2) + (db[j]->square() * (1-beta2));
                    /*Correcting first iterations*/
                    //Matrix Sdb_corr = *Sdb[j] / (1 - pow(beta2, iteration+1));
                    //Matrix Vdb_corr = *Vdb[j] / (1 - pow(beta1, iteration+1));
                    /*Updating parameters*/
                    //Matrix temp2 = Vdb_corr / (Sdb_corr.Sqrt() + epsilon);
                    //*b[j] = *b[j] - temp2 * alpha;
               // }
                /*Updating Vdw,Sdw*/
               // *(Vdw[i][j]) = ((*(Vdw[i][j])) * (beta1 * momentum)) + ((*(dW[i][j])) * (1-beta1 * momentum));
                //*(Sdw[i][j]) = ((*(Sdw[i][j])) * beta2) + (dW[i][j]->square() * (1-beta2));
                /*Correcting first iterations*/
                // Matrix Vdw_corr = *(Vdw[i][j]) / (1 - pow(beta1, iteration+1));
                // Matrix Sdw_corr = *(Sdw[i][j]) / (1 - pow(beta2, iteration+1));
                 /*Updating parameters*/
                 //Matrix temp1 = Vdw_corr / (Sdw_corr.Sqrt() + epsilon);
                 //*(W[i][j]) = *(W[i][j]) - temp1 * alpha;
            //}

        /*
        grades.erase(CharGen("Vdw",  1));
        grades.put(CharGen("Vdw",  1), Vdw);
        grades.erase(CharGen("Vdb", 1));
        grades.put(CharGen("Vdb",  1), Vdb);
        grades.erase(CharGen("Sdw", i + 1));
        grades.put(CharGen("Sdw", i + 1), Sdw);
        grades.erase(CharGen("Sdb", i + 1));
        grades.put(CharGen("Sdb", i + 1), Sdb);
        parameters.replace(CharGen("W", 1),W);
        parameters.replace(CharGen("b", 1),b);
        /*Erasing dW, db*/
        /*
        grades.erase(CharGen("dW",  1));
        grades.erase(CharGen("db",  1));
        for(int i=0;i<W.size();i++)
        {
              Vdw[i].DELETE();
              Sdw[i].DELETE();
        }
        Sdb.DELETE();
        Vdb.DELETE();
    }
    /*END OF ADAM OPTIMIZER*/
}

#endif // BACKPROB2_H_INCLUDED
