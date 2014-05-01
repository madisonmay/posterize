#ifndef MODE_H
#define MODE_H
__global__
void mode(const unsigned char* input, unsigned char* output, size_t cols, size_t rows, int channels);
char* processMode(char* image_rgb, size_t cols, size_t rows, int channels, int colors);
#endif