#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define block_size 64

void mmul(float *A, float *B, float *C, int n)  {
    int rank, size, rows, i, j, k;

    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    rows = n / size;

    float A_local[rows * n];
    MPI_Scatter(A, rows * n, MPI_FLOAT, A_local, rows * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(rank != 0) {
        B = (float*)malloc(sizeof(float) * n * n);
    }
    MPI_Bcast(B, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    float C_local[rows * n];

    for(i = 0; i < rows; i++) {
       for(j = 0; j < n; j++) {
           C_local[i * n + j] = 0;
           for(k = 0; k < n; k++) {
               C_local[i * n + j] += A_local[i * n + k] * B[k * n + j];
           }
       }
    }

    MPI_Gather(C_local, rows * n, MPI_FLOAT, C, rows * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(rank != 0) {
        free(B);
    }
}

