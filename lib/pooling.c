#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "pooling.h"

double tabMax(double* tab, int size) {
    double max = tab[0];
    for (int i=0; i<size; i++) {
        if (tab[i] > max) {
            max = tab[i];
        }
    }
    return max;
}

double** poolingForward(double** input, int width, int height, int numFilters) {
    double** output = malloc(width * height * sizeof(double));
    assert(output != NULL);

    for (int i=0; i<width*height; i++) {
        output[i] = malloc(numFilters * sizeof(double));
        assert(output[i] != NULL);

        for (int k=0; k<numFilters; k++) {
            double* cell = malloc(4 * sizeof(double));
            assert(cell != NULL);

            cell[0] = input[2*i][k];
            cell[1] = input[2*i + 1][k];
            cell[2] = input[2*i + width][k];
            cell[3] = input[2*i + width + 1][k];

            output[i][k] = tabMax(cell, 4);
            free(cell);
        }
    }
    return output;
}