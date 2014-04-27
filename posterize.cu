#include "posterize.h"

__global__
void posterize(const unsigned char* input, 
               unsigned char* output, 
               size_t cols, size_t rows, int n)
{
  // int i;
  // int size = cols*rows*3;
  // int w = 256/n;
  // for (i=0; i<size; i++) {
  //   output[i] = (input[i]/w)*w+w/2;
  // }
  int x = blockDim.x*blockIdx.x+threadIdx.x;
  int y = blockDim.y*blockIdx.y+threadIdx.y;
  if (x >= cols || y >= rows) {
      return;
  }
  int idx = x+y*cols;
  output[idx] = input[idx]; //copy image
}