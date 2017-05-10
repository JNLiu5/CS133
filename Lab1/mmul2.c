#include "const.h"
#include <sys/time.h>

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define block_size 64

void mmul2(float A[ni][nk], float B[nk][nj], float C[ni][nj])
{
    omp_set_num_threads(32);
	int i, j, k;
    int bi, bj, bk;
    int i_blocks = ni/block_size;
    int j_blocks = nj/block_size;

    #pragma omp parallel for private(i, j, k, bi, bj)
    for(i = 0; i < i_blocks; i++) {
        for(j = 0; j < j_blocks; j++) {
            float temp[block_size][block_size] = {0};

            for(k = 0; k < nk; k++) {
                for(bi = 0; bi < block_size; bi++) {
                    for(bj = 0; bj < block_size; bj++) {
                        temp[bi][bj] += A[i*block_size + bi][k]*B[k][j*block_size + bj];
                    }
                }
            }

            for(bi = 0; bi < block_size; bi++) {
                for(bj = 0; bj < block_size; bj++) {
                    C[i*block_size + bi][j*block_size + bj] = temp[bi][bj];
                }
            }
        }
    }
}

