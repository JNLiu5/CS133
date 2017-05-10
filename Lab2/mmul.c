#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define block_size 32


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

    for(i = 0; i < rows * n; i++) {
        C_local[i] = 0;
    }
    
    int bi, bj;
    int i_blocks = rows / block_size;
    int j_blocks = n / block_size;

    for(i = 0; i < i_blocks; i++) {
        for(j = 0; j < j_blocks; j++) {
            for(k = 0; k < n; k++) {
                for(bi = 0; bi < block_size; bi++) {
                    for(bj = 0; bj < block_size; bj++) {
                        C_local[(i * block_size + bi) * n + (j * block_size + bj)] += A_local[(i * block_size + bi) * n + k] * B[k * n + (j * block_size + bj)];
                    }
                }
            }
        }
    }

    MPI_Gather(C_local, rows * n, MPI_FLOAT, C, rows * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(rank != 0) {
        free(B);
    }
}

