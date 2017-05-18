#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/*
 * Generate an array of random numbers
 */
int* getRandomArray(int size, int min, int max) {
    int i;
    
    int *arr = (int *)malloc(sizeof(int) * size);
    for(i = 0;i<size;++i) {
        arr[i] = rand() % (max - min + 1) + min;
    }

    return arr;
}

/*
 * Display an array on STDOUT
 */
void displayArray(int *arr, int size) {
    int i;
    for(i=0;i<size;++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

/*
 * Duplicate an integer array
 */
int* dupArray(int *arr, int size) {
    int* dup = (int *)malloc(size * sizeof(int));
    memcpy(dup, arr, size * sizeof(int));

    return dup;
}

/*
 * Find the absolute error between two arrays
 */
int diffArrays(int *arr1, int* arr2, int size) {
    int diff = 0;
    int i;

    for(i=0;i<size;++i) {
        diff += abs(arr1[i] - arr2[i]);
    }
    
    return diff;
}


/*
 * This function returns the rank of an element in an array
 *
 */
int getRank(int* arr, int left, int right, int item) {
    while(left < right) {
        long mid = (left + right) / 2;

        if(item < arr[mid]) {
            right = mid;
        }
        else{
            left = mid + 1;
        }
    }

    return left;
} 
