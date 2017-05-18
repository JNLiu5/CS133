#include <stdio.h>
#include <stdlib.h>
#include "const.h"
#include <time.h>
#include <omp.h>

#include "psort.h"

/*
 * TODO: Make parallelization worth it!
 * Hint:
 * 1. Make sure that things aren't unnecessarily shared
 * 2. See if you can block computations
 */

// Comment this to disable merge parallelization
#define MPAR

// Comment this to disable mergesort parallelization
#define MSPAR

/*
 * The regular sequential merging procedure. Cannot be parallelized 
 * efficiently
 */
void merge(int* arr, int* aux, int left, int mid, int right) {
    int i = left;
    int j = mid;
    int k;

    for(k=left;k<right;++k) {
        if(i == mid) {
            aux[k] = arr[j++];
        }
        else if(j == right) {
            aux[k] = arr[i++];
        }
        else if(arr[i] < arr[j]) {
            aux[k] = arr[i++];
        }
        else {
            aux[k] = arr[j++];
        }
    }

}

void exchange(int* a, int* b) {
    int temp = *b;
    *b = *a;
    *a = temp;
}

/*
 * A faster parallel merge procedure
 * This attempts to merge A[i..j] and B[k..l] and places the result in aux[p..q]
 */
void pMerge(int *arr, int i, int j, int k, int l, int* aux, int p, int q) {
    int m = j - i;
    int n = l - k;

    if (m < n) {
        exchange(&i, &k);
        exchange(&j, &l);
        exchange(&m, &n);
    }

    if(m == 0){
        return;
    }
    else {
        int r = (i + j) / 2;
        int s = getRank(arr, k, l, arr[r]); 
        int t = p + (r - i) + (s - k);
        aux[t] = arr[r];

#pragma omp task firstprivate(i, r, k, s, p, t) 
        pMerge(arr, i, r, k, s, aux, p, t);

#pragma omp task firstprivate(r, j, s, l, t, q) 
        pMerge(arr, r + 1, j, s, l, aux, t + 1, q);
    }
}

/*
 * Sequential merge sort
 */
void mergeSort(int* arr, int *aux, int left, int right) {
    if(right - left <= 1) {
        return;
    }
    int k;

    int mid = (left + right) / 2;


    mergeSort(arr, aux, left, mid);
    mergeSort(arr, aux, mid, right);

    merge(arr, aux, left, mid, right);

    for(k=left;k<right;++k) {
        arr[k] = aux[k];
    }
}

/*
 * Parallel merge sort
 * Creates a new task for every step in the recursion
 */
void pMergeSort(int* arr, int *aux, int left, int right) {
    if(right - left <= 1) {
        return;
    }
    int k;

    int mid = (left + right) / 2;


#ifdef MSPAR
#pragma omp task firstprivate(arr, aux, left, mid, right)
    mergeSort(arr, aux, left, mid);

#pragma omp task firstprivate(arr, aux, left, mid, right)
    mergeSort(arr, aux, mid, right);

#pragma omp taskwait
#endif

#ifndef MSPAR
    mergeSort(arr, aux, left, mid);
    mergeSort(arr, aux, mid, right);
#endif

#ifdef MPAR
    if(right - left > 8192) {
        pMerge(arr, left, mid, mid, right, aux, left, right);
    }
    else {
        merge(arr, aux, left, mid, right);
    }
#endif

#ifndef MPAR
    merge(arr, aux, left, mid, right);
#endif

    for(k=left;k<right;++k) {
        arr[k] = aux[k];
    }
}

int main(int argc, char **argv) {
    double start, stop;

    int *arr = getRandomArray(SIZE, 1, 1000);
    int *pArr = dupArray(arr, SIZE);
    int *aux = (int *)malloc(SIZE * sizeof(int));

		omp_set_num_threads(32);
    /*
     * Run sequential mergesort
     */
    start = omp_get_wtime();
    mergeSort(arr, aux, 0, SIZE);
    stop = omp_get_wtime();
    printf("Sequential mergesort took %f seconds\n", stop - start);


    /*
     * Run parallel mergesort
     */
    start = omp_get_wtime();
    pMergeSort(pArr, aux, 0, SIZE);
    stop = omp_get_wtime();
    printf("Parallel mergesort took %f seconds\n", stop - start);
    printf("Diff was %d\n", diffArrays(pArr, arr, SIZE));

    return 0;
}

