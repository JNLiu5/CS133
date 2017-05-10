#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define block_size 64

void mmul(float *A, float *B, float *C, int n)  {
    int rank, size, rows, rows_offset, remainder;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //   master 
    if(rank == 0) {
        int num_workers = size - 1;
        rows = n / num_workers;
        remainder = n % num_workers;
        rows_offset = 0;
        //  send data to workers
        int worker;
        for(worker = 1; worker < size; worker++) {
            MPI_Send(&rows, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            MPI_Send(&rows_offset, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
            if(worker == size - 1) {
                //  if worker is the last one, also send them the remainder
                MPI_Send(&remainder, 1, MPI_INT, worker, 2, MPI_COMM_WORLD);
                MPI_Send(A + (rows_offset * n), (rows + remainder) * n, MPI_FLOAT, worker, 3, MPI_COMM_WORLD);
            }
            else {
                //  otherwise, send them 0
                int zero = 0;
                MPI_Send(&zero, 1, MPI_INT, worker, 2, MPI_COMM_WORLD);
                MPI_Send(A + (rows_offset * n), rows * n, MPI_FLOAT, worker, 3, MPI_COMM_WORLD);
            }
            MPI_Send(B, n * n, MPI_FLOAT, worker, 4, MPI_COMM_WORLD);
            rows_offset += rows;
        }

        //  gather data from workers
        for(worker = 1; worker < size; worker++) {
           MPI_Recv(&rows, 1, MPI_INT, worker, 0, MPI_COMM_WORLD, &status);
           MPI_Recv(&rows_offset, 1, MPI_INT, worker, 1, MPI_COMM_WORLD, &status);
           MPI_Recv(C + rows_offset * n, rows * n, MPI_FLOAT, worker, 2, MPI_COMM_WORLD, &status);
        }
    }
    //  worker
    else {
        MPI_Recv(&rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows_offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&remainder, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
        rows += remainder;

        float A_local[rows * n];
        float B_local[n * n];
        float C_local[rows * n];

        MPI_Recv(&A_local, rows * n, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &status);
        MPI_Recv(&B_local, n * n, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, &status);

        int i, j, k;
        for(i = 0; i < rows; i++) {
            for(j = 0; j < n; j++) {
                C_local[i * n + j] = 0;
                for(k = 0; k < n; k++) {
                    C_local[i * n + j] += A_local[i * n + k]*B_local[k * n + j];
                }
            }
        }

        MPI_Send(&rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&rows_offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(C_local, rows * n, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
}

