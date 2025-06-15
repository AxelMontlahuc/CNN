#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "import.h"
#include "convolution.h"
#include "pooling.h"
#include "dense.h"
#include "output.h"

#include "backprop.h"

double* dL_dprobs(double* probs, int size, int label) {
    double* grad = calloc(size, sizeof(double));
    grad[label] = -1.0 / probs[label];
    return grad;
}

double* drightProb_dtotals(double* totals, int size, int label) {
    double* grad = malloc(size * sizeof(double));

    double sum = 0.0;
    for (int i=0; i<size; i++) {
        sum += exp(totals[i]);
    }

    for (int i=0; i<size; i++) {
        if (i == label) {
            grad[i] = exp(totals[i]) * (sum - exp(totals[i])) / (sum * sum) ;
        } else {
            grad[i] = -exp(totals[label]) * exp(totals[i]) / (sum * sum);
        }
    }

    return grad;
}

double** dtotals_dweights(double* input, int size, int width, int height, int numFilters) {
    double** grad = malloc(size * sizeof(double*));
    assert(grad != NULL);

    for (int i=0; i<size; i++) {
        grad[i] = malloc(width * height * numFilters * sizeof(double));
        assert(grad[i] != NULL);

        for (int j=0; j<width*height*numFilters; j++) {
            grad[i][j] = input[j];
        }
    }

    return grad;
}

double* dtotals_dbiases(int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i=0; i<size; i++) {
        grad[i] = 1.0;
    }

    return grad;
}

double** dtotals_dinput(DenseLayer* denseLayer, int size, int width, int height, int numFilters) {
    double** grad = malloc(size * sizeof(double*));
    assert(grad != NULL);

    for (int i=0; i<size; i++) {
        grad[i] = malloc(width * height * numFilters * sizeof(double));
        assert(grad[i] != NULL);

        for (int j=0; j<width*height*numFilters; j++) {
            grad[i][j] = denseLayer->weights[i][j];
        }
    }

    return grad;
}

double* dL_dtotals(double* dL_dp, double* dp_dtot, int size, int label) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i=0; i<size; i++) {
        grad[i] = dL_dp[label] * dp_dtot[i];
    }

    return grad;
}

double** dL_dweights(double* dL_dtot, double** dtot_dw, int size, int width, int height, int numFilters) {
    double** grad = malloc(size * sizeof(double*));
    assert(grad != NULL);

    for (int i=0; i<size; i++) {
        grad[i] = malloc(width * height * numFilters * sizeof(double));
        assert(grad[i] != NULL);

        for (int j=0; j<width*height*numFilters; j++) {
            grad[i][j] = dL_dtot[i] * dtot_dw[i][j];
        }
    }

    return grad;
}

double* dL_dbiases(double* dL_dtot, double* dtot_db, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i=0; i<size; i++) {
        grad[i] = dL_dtot[i] * dtot_db[i];
    }

    return grad;
}

double** dL_dinput(double* dL_dtot, double** dtot_dinput, int size, int width, int height, int numFilters) {
    double** grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i=0; i<size; i++) {
        grad[i] = malloc(width * height * numFilters * sizeof(double));
        assert(grad[i] != NULL);

        for (int j=0; j<width*height*numFilters; j++) {
            grad[i][j] = dL_dtot[i] * dtot_dinput[i][j];
        }
    }
    return grad;
}

double** denseBackprop(DenseLayer* denseLayer, double* probs, double* totals, double* pooledImage, int width, int height, int numFilters, int label, double learningRate) {
    double* dL_dp = dL_dprobs(probs, denseLayer->size, label);
    double* dp_dtot = drightProb_dtotals(totals, denseLayer->size, label);
    double* dL_tot = dL_dtotals(dL_dp, dp_dtot, denseLayer->size, label);
    double** dtot_dw = dtotals_dweights(pooledImage, denseLayer->size, width, height, numFilters);
    double* dtot_db = dtotals_dbiases(denseLayer->size);
    double** dL_dw = dL_dweights(dL_tot, dtot_dw, denseLayer->size, width, height, numFilters);
    double* dL_db = dL_dbiases(dL_tot, dtot_db, denseLayer->size);
    double** dtot_din = dtotals_dinput(denseLayer, denseLayer->size, width, height, numFilters);
    double** dL_din = dL_dinput(dL_tot, dtot_din, denseLayer->size, width, height, numFilters);

    for (int i = 0; i < denseLayer->size; i++) {
        for (int j = 0; j < width * height * numFilters; j++) {
            denseLayer->weights[i][j] -= learningRate * dL_dw[i][j];
        }
        denseLayer->biases[i] -= learningRate * dL_db[i];
    }

    for (int i = 0; i < denseLayer->size; i++) {
        free(dtot_dw[i]);
        free(dL_dw[i]);
        free(dtot_din[i]);
    }
    free(dL_dp);
    free(dp_dtot);
    free(dL_tot);
    free(dtot_dw);
    free(dtot_db);
    free(dL_dw);
    free(dL_db);
    free(dtot_din);

    return dL_din;
}

double* backpropagation(ConvLayer* convLayer, DenseLayer* denseLayer, double** image, int width, int height, int divisor, int label, double learningRate) {
    double** convolutedImage = convolutionForward(convLayer, image, width, height, divisor);
    double* pooledImage = poolingForward(convolutedImage, (width-(divisor-1))/2, (height-(divisor-1))/2, convLayer->numFilters);
    double* totals = denseForward(denseLayer, pooledImage, (width-(divisor-1))/2, (height-(divisor-1))/2, convLayer->numFilters);
    double* probs = softmax(totals, denseLayer->size);
    double** dL_din = denseBackprop(denseLayer, probs, totals, pooledImage, (width-(divisor-1))/2, (height-(divisor-1))/2, convLayer->numFilters, label, learningRate);

    for (int i = 0; i < convLayer->numFilters; i++) {
        free(convolutedImage[i]);
    }
    for (int i = 0; i < denseLayer->size; i++) {
        free(dL_din[i]);
    }
    free(convolutedImage);
    free(pooledImage);
    free(totals);
    free(dL_din);
    return probs;
}