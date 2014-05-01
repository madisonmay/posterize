#ifndef POSTERIZE_H
#define POSTERIZE_H
#include <stdio.h>

__global__
void posterize(const unsigned char* const input, 
               unsigned char* const output, 
               size_t cols, size_t rows, int channels, int n);
__global__
void smooth(const unsigned char* const input, 
               unsigned char* const output, 
               size_t cols, size_t rows, int channels, int n,
               int *hist);

char* processPosterize(char* image_rgb, size_t cols, size_t rows, int channels, int colors);
#endif