#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>

#define DEBUG 0

#define N 1024

int main(int argc, char *argv[] ) {

  int i, j, numprocs, rank, rows_sent,total_sent,exact_div, extra_row;


  struct timeval  tv1, tv2;
  float result[N];

  MPI_Init (&argc, &argv);
  MPI_Comm_size ( MPI_COMM_WORLD , & numprocs );
  MPI_Comm_rank ( MPI_COMM_WORLD , & rank );
  exact_div= N%numprocs;
  rows_sent=N/numprocs;
  extra_row=exact_div ? 1:0;
  total_sent=rows_sent+extra_row*N;
  float vector[N+exact_div];


  for(i=0;i<N+exact_div;i++) {
    vector[i] = i;
  }
//  /* Initialize Matrix and Vector */
//
  if(!rank){

    float matrix[N+numprocs*extra_row][N];
    for(i=0;i<N+numprocs*extra_row;i++) {
      for(j=0;j<N;j++) {
        matrix[i][j] = i+j;
        //printf("%.2f  ",matrix[i][j]);
      }
      //printf("\n");
    }
  }

  float mymatrix[rows_sent+extra_row][N];
  for(i=0;i<rows_sent+extra_row;i++) {
    for(j=0;j<N;j++) {
      mymatrix[i][j] = i+j;
    }
  }
//
  MPI_Scatter(matrix,total_sent,MPI_INT,mymatrix,total_sent,MPI_INT,0,MPI_COMM_WORLD);
//
  gettimeofday(&tv1, NULL);
//
  for(i=0;i<rows_sent+extra_row;i++) {
    result[i]=0;
    for(j=0;j<N;j++) {
      result[i] += mymatrix[i][j]*vector[j];
    }
  }
//
  gettimeofday(&tv2, NULL);
//  
  MPI_Gather(result,rows_sent+extra_row,MPI_INT,vector,rows_sent+extra_row,MPI_INT,0,MPI_COMM_WORLD);
//
  int microseconds = (tv2.tv_usec - tv1.tv_usec)+ 1000000 * (tv2.tv_sec - tv1.tv_sec);
  if(!rank){
    if (DEBUG){
      for(i=0;i<N;i++) {
        if(i!=numprocs*(rows_sent+extra_row)){
          printf(" %f \t ",vector[i]);
        }else if (exact_div){
          exact_div--;
          printf(" %f \t ",vector[i]);
        }
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
