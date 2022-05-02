#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>

#define DEBUG 0

#define N 1024

int main(int argc, char *argv[] ) {

  int i, j, numprocs, rank, rows_sent,total_sent;
  float matrix[N][N];
  float vector[N];
  float result[N];
  struct timeval  tv1, tv2;


  MPI_Init (&argc, &argv);
  MPI_Comm_size ( MPI_COMM_WORLD , & numprocs );
  MPI_Comm_rank ( MPI_COMM_WORLD , & rank );

  for(i=0;i<N;i++) {
    vector[i] = i;
  }
//  /* Initialize Matrix and Vector */
//
  if(!rank){
    for(i=0;i<N;i++) {
      for(j=0;j<N;j++) {
        matrix[i][j] = i+j;
        //printf("%.2f  ",matrix[i][j]);
      }
      //printf("\n");
    }
  }
  rows_sent=N/numprocs;
  total_sent=rows_sent*N;
  float mymatrix[rows_sent][N];
  for(i=0;i<rows_sent;i++) {
    for(j=0;j<N;j++) {
      mymatrix[i][j] = i+j;
    }
  }
//
  MPI_Scatter(matrix,total_sent,MPI_INT,mymatrix,total_sent,MPI_INT,0,MPI_COMM_WORLD);
//
  gettimeofday(&tv1, NULL);
//
  for(i=0;i<rows_sent;i++) {
    result[i]=0;
    for(j=0;j<N;j++) {
      result[i] += mymatrix[i][j]*vector[j];
    }
  }
//
  gettimeofday(&tv2, NULL);
//  
  MPI_Gather(result,rows_sent,MPI_INT,vector,rows_sent,MPI_INT,0,MPI_COMM_WORLD);
//
  int microseconds = (tv2.tv_usec - tv1.tv_usec)+ 1000000 * (tv2.tv_sec - tv1.tv_sec);
  if(!rank){
    if (DEBUG){
      for(i=0;i<N;i++) {
        printf(" %f \t ",vector[i]);
      }
    } else {
      printf ("Time (seconds) = %lf\n", (double) microseconds/1E6);
    }    
  }
//
//
//
//  return 0;
  MPI_Finalize();
}
