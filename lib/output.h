#ifndef OUTPUT_H
#define OUTPUT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

double* softmax(double* input, int size);
double loss(double* probs, int label);
int accuracy(double* probs, int label, int size);

#endif