#include "const.h"

void mmul1(float A[ni][nk], float B[nk][nj], float C[ni][nj])
{
    omp_set_num_threads(omp_get_num_procs());
	int i, j, k;
    #pragma omp parallel for private(i)
    for (i = 0; i < ni; i++) {
        for(j = 0; j < nj; j++) {
            C[i][j] = 0;
        }
    }

    #pragma omp parallel for private(i)
    for (i = 0; i < ni; i++) {
        for(k = 0; k < nk; k++) {
            for(j = 0; j < nj; j++) {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

