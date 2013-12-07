#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "hdf5.h"
#include "mpi.h"
#include <math.h>
using namespace std;

string line;
string tmpstr;
int counter = 0;

int m_rows = 2;
int m_cols = 2;

int m1_rows;
int m1_cols;

int m2_rows;
int m2_cols;
int m_counter = 0;

int matrix_1_break;
int matrix_2_break;

int global_length;   
vector< double > matrix_1;
vector< double > matrix_2;
vector< double > result_matrix;

double* read_hdf(const std::string &filename, double* &data, int &m_rows, int &m_cols){

    hid_t file_id, dataset_id, space_id, property_id;
    herr_t status;

    //Create a new file using the default properties.
    file_id = H5Fopen (filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, "DATASET", H5P_DEFAULT);
    space_id = H5Dget_space(dataset_id);
    int length = H5Sget_simple_extent_npoints(space_id);
    hsize_t  dims[2];
    hsize_t  mdims[2];
    status = H5Sget_simple_extent_dims(space_id,dims,mdims);
    m_rows = dims[0];
    m_cols = dims[1];
    
    data = new double[length];
    global_length = length;
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
    H5P_DEFAULT, data);
    
    status = H5Sclose(space_id);
    status = H5Dclose(dataset_id);
    status = H5Fclose(file_id);
    return data;
}

void write_hdf(const std::string &filename, double* &data, int &m_rows, int &m_cols){

    hid_t file_id, dataset_id, space_id, property_id;
    herr_t status;

    hsize_t dims[2] = {m_rows,m_cols};


    //Create a new file using the default properties.
    file_id = H5Fcreate (filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    //Create dataspace.  Setting maximum size to NULL sets the maximum
    //size to be the current size.
    space_id = H5Screate_simple (2, dims, NULL);

    //Create the dataset creation property list, set the layout to compact.
    property_id = H5Pcreate (H5P_DATASET_CREATE);
    status = H5Pset_layout (property_id, H5D_CONTIGUOUS);

    // Create the dataset.
    dataset_id = H5Dcreate (file_id, "DATASET", H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, property_id, H5P_DEFAULT);

    //Write the data to the dataset.
    status = H5Dwrite (dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    status = H5Sclose(space_id);
    status = H5Dclose(dataset_id);
    status = H5Fclose(file_id);
    status = H5Pclose(property_id);
}

int index(int row, int col, int col_number){ return row*col_number + col;}

double * matrixMult(double * data_a, double * data_b, double * data_c, int nlocal){
  for (int i = 0; i < nlocal; i++){
    for (int j = 0; j < nlocal; j++){
      double sum = 0;
      for (int k = 0; k < nlocal; k++){
        sum = data_a[index(i, k, nlocal)] * data_b[index(k, j, nlocal)];
        data_c[index(i, j, nlocal)] += sum;  
      }
    }
  }
  return data_c;
}

int main(int argc, char* argv[]){
 MPI_Init(&argc, &argv);
 string filename_a = argv[1];
 string filename_b = argv[2];
 string filename_c = argv[3];

 int size, rank;
 double * data = 0;
 int i;

 read_hdf(filename_a, data, m_rows, m_cols);
 double data_a[global_length];
 m1_rows = m_rows;
 m1_cols = m_cols;

 read_hdf(filename_b, data, m_rows, m_cols);
 double data_b[global_length];
 m2_rows = m_rows;
 m2_cols = m_cols;
 MPI_Status status;
 MPI_Comm comm_cart;

 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 MPI_Comm_size(MPI_COMM_WORLD, &size);

 int n = m1_rows;     
 m_rows, m_cols = 0;          
 int dims[2];
 int nlocal;
 dims[0] = dims[1] = sqrt(size);         
 int periods[2];
 periods[0] = periods[1] = 1;

 int cartRank = 0;
 int cartCoords[2];

 int uprank, downrank, leftrank, rightrank;
 int shiftsource, shiftdest;

 MPI_Dims_create(size, 2, dims);    

 for(int x = 0; x < global_length; ++x){   
   data_a[x] = data[x];
 }

 for(int y = 0; y < global_length; ++y){
   data_b[y] = data[y];
 }
 MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &comm_cart); 
 MPI_Comm_rank(comm_cart, &cartRank);
 MPI_Cart_coords(comm_cart, cartRank, 2, cartCoords); 
 MPI_Cart_shift(comm_cart, 1, 1, &rightrank, &leftrank);
 MPI_Cart_shift(comm_cart, 0, 1, &downrank, &uprank);
 
 nlocal = n/dims[0];

 /* perform initial matrix alignment */
 double* a = new double[nlocal*nlocal];   
 double* b = new double[nlocal*nlocal];
 double* c = new double[nlocal*nlocal];
 /*distribute matrices*/
 int carta = cartCoords[0];
 int cartb = cartCoords[1];
 for (int i = 0; i < nlocal; i++){
   for (int j = 0; j < nlocal; j++){
     a[index(i, j, nlocal)] = data_a[index(nlocal*carta+i, nlocal*cartb+j, n)];
     b[index(i, j, nlocal)] = data_b[index(nlocal*carta+i, nlocal*cartb+j, n)];
   }
 }

 MPI_Cart_shift(comm_cart, 1, -cartCoords[0], &shiftsource, &shiftdest);
 MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE, shiftdest, 1, shiftsource, 1, comm_cart, &status);

 MPI_Cart_shift(comm_cart, 0, -cartCoords[1], &shiftsource, &shiftdest); 
 MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE, shiftdest, 1, shiftsource, 1, comm_cart, &status); 

 for (i = 0; i<dims[0]; i++) {
   c = matrixMult(a, b, c, nlocal);
   MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE, leftrank, 1, rightrank, 1, comm_cart, &status);
   MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE, uprank, 1, downrank, 1, comm_cart, &status);
  }


 global_length = ( m1_rows * m2_cols);
 double data_c[global_length];
 result_matrix.resize(global_length);        

 double * data_result = data_c;
 double * big_matrix = new double[n * n];
 if (rank == 0){
   for (int i = 0; i < nlocal; i++){
     for (int j = 0; j < nlocal; j++){
       big_matrix[index(carta*nlocal+i, cartb*nlocal+j, n)] = c[index(i, j, nlocal)];
     }
   }
   for ( int k = 1; k < size; k++){
     MPI_Recv(c, nlocal*nlocal, MPI_DOUBLE, k, 0, comm_cart, &status);
     MPI_Recv(&cartCoords[0], 1, MPI_INT, k, 0, comm_cart, &status);
     MPI_Recv(&cartCoords[1], 1, MPI_INT, k, 1, comm_cart, &status);
     carta = cartCoords[0];
     cartb = cartCoords[1]; 
     for (int i = 0; i < nlocal; i++){
       for (int j = 0; j < nlocal; j++){
         big_matrix[index(carta*nlocal+i, cartb*nlocal+j, n)] = c[index(i, j, nlocal)];
       }
     }

   }   
 }
 else {
   MPI_Send(c, nlocal*nlocal, MPI_DOUBLE, 0, 0, comm_cart);
   MPI_Send(&cartCoords[0], 1, MPI_INT, 0, 0, comm_cart);
   MPI_Send(&cartCoords[1], 1, MPI_INT, 0, 1, comm_cart);
 }
 if (rank == 0){
   write_hdf(filename_c, big_matrix, n, n);
 }

 MPI_Finalize();
 return 0;

}

