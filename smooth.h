#ifndef SMOOTH_H
#define SMOOTH_H
__global__
void smooth(const unsigned char* input,
            unsigned char* output,
            size_t cols, size_t rows, int channels, int n,
            int *hist);

char* processSmooth(char* image_rgb, size_t cols, size_t rows, int channels, int colors);
#endif