#ifndef IMPORT_H
#define IMPORT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

uint32_t swapEndian(uint32_t val);
int* readParameters(char* filename);
double** readImage(FILE* f, int width, int height);
double*** readImages(char* filename, int numImages, int width, int height);
int* readLabels(char* filename, int numImages);

#endif