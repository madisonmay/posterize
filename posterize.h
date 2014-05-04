#ifndef POSTERIZE_H
#define POSTERIZE_H
#include <stdio.h>

__global__
void mode_kernel(const unsigned char* input, 
                 unsigned char* output, 
                 size_t cols, size_t rows, int channels,
                 int colors, unsigned int* hist);

__global__
void reduce_kernel(const unsigned char* input, 
                   unsigned char* output, 
                   size_t cols, size_t rows, int channels, int n);

char* processPosterize(char* image_rgb, size_t cols, size_t rows, int channels, int colors);
#endif