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
  struct timeval  t_comms_1, t_comms_2, t_comp_1, t_comp_2;


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
  gettimeofday(&t_comms_1, NULL);
  MPI_Scatter(matrix,total_sent,MPI_INT,mymatrix,total_sent,MPI_INT,0,MPI_COMM_WORLD);
//
  gettimeofday(&t_comp_1, NULL);
//
  for(i=0;i<rows_sent;i++) {
    result[i]=0;
    for(j=0;j<N;j++) {
      result[i] += mymatrix[i][j]*vector[j];
    }
  }
//
  gettimeofday(&t_comp_2, NULL);
//  
  MPI_Gather(result,rows_sent,MPI_INT,vector,rows_sent,MPI_INT,0,MPI_COMM_WORLD);
//
  gettimeofday(&t_comms_2, NULL);
  int comp = (t_comp_2.tv_usec - t_comp_1.tv_usec)+ 1000000 * (t_comp_2.tv_sec - t_comp_1.tv_sec);
  int comms = (t_comms_2.tv_usec - t_comms_1.tv_usec)+ 1000000 * (t_comms_2.tv_sec - t_comms_1.tv_sec);
  comms= comms-comp;
  if(!DEBUG){
    if(rank){
      MPI_Send(&comms,1,MPI_INT,0,0,MPI_COMM_WORLD);
      MPI_Send(&comp,1,MPI_INT,0,1,MPI_COMM_WORLD);
    }
  }

  if(!rank){
    if (DEBUG){
      for(i=0;i<N;i++) {
        printf(" %f \t ",vector[i]);
      }
    } else {
      printf ("Time (seconds) for the communications of process %d = %lf\n",rank,  (double) comms/1E6);
      printf ("Time (seconds) for the computation of process %d = %lf\n",rank,  (double) comp/1E6);
      for(i=1;i<numprocs;i++){
      MPI_Recv(&comms,1,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Recv(&comp,1,MPI_INT,i,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      printf ("Time (seconds) for the communications of process %d = %lf\n",i,  (double) comms/1E6);
      printf ("Time (seconds) for the computation of process %d = %lf\n",i,  (double) comp/1E6);
      }
    }    
  }
//
//
//
//  return 0;
  MPI_Finalize();
}
