#ifndef IMPORT_H
#define IMPORT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

int* readParameters(char* filename);
double*** readImages(char* filename);
int* readLabels(char* filename);

#endif