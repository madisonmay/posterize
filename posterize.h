#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort);

__global__
void posterize(const unsigned char* const input, 
               unsigned char* const output, 
               size_t cols, size_t rows, int channels, int n);

char* process(char* image_rgb, size_t cols, size_t rows, int channels, int colors);
