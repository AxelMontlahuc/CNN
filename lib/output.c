#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "dense.h"

double* softmax(double* input, int size) {
    double* output = malloc(size * sizeof(double));
    assert(output != NULL);

    double sum = 0.0;
    for (int i=0; i<size; i++) {
        sum += exp(input[i]);
    }

    for (int i=0; i<size; i++) {
        output[i] = exp(input[i]) / sum;
    }
    return output;
}

double loss(double* probs, int label) {
    return -log(probs[label]);
}

int accuracy(double* probs, int label, int size) {
    double max = probs[0];
    int index = 0;
    for (int i=0; i<size; i++) {
        if (probs[i] > max) {
            max = probs[i];
            index = i;
        }
    }
    
    if (index == label) return 1;
    else return 0;
}