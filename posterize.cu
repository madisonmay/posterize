#include <cuda.h>

__global__
void posterize(const unsigned char* const input, 
               unsigned char* const output, 
               size_t cols, size_t rows, int n)
{
  int i;
  int size = cols*rows*3;
  int w = 256/n;
  for (i=0; i<size; i++) {
    output[i] = (input[i]/w)*w+w/2;
  }
}