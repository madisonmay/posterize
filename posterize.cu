#include "posterize.h"

__global__
void posterize(const unsigned char* input, 
               unsigned char* output, 
               size_t cols, size_t rows, int n)
{
  int x = blockDim.x*blockIdx.x+threadIdx.x;
  int y = blockDim.y*blockIdx.y+threadIdx.y;
  if (x >= cols || y >= rows) {
      return;
  }

  int idx = y*cols+x;
  int w = 256/n;
  output[idx*3+0] = (input[idx*3+0]/w)*w+w/2;
  output[idx*3+1] = (input[idx*3+1]/w)*w+w/2;
  output[idx*3+2] = (input[idx*3+2]/w)*w+w/2;
}