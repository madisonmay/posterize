#include "utils.h"
#include "posterize.h"

__global__
void posterize(const unsigned char* input, 
               unsigned char* output, 
               size_t cols, size_t rows, int channels, int n)
{
  int x = blockDim.x*blockIdx.x+threadIdx.x;
  int y = blockDim.y*blockIdx.y+threadIdx.y;
  if (x >= cols || y >= rows) {
      return;
  }
  int idx = y*cols+x;
  int w = 256/n;
  output[idx*channels+0] = (input[idx*channels+0]/w)*w+w/2;
  output[idx*channels+1] = (input[idx*channels+1]/w)*w+w/2;
  output[idx*channels+2] = (input[idx*channels+2]/w)*w+w/2;
}

char* processPosterize(char* image_rgb, size_t cols, size_t rows, int channels, int colors)
{
  unsigned char *d_img_in;
  unsigned char *d_img_out;
  char *h_img_out;
  size_t image_data_size = sizeof(unsigned char)*cols*rows*channels;
  h_img_out = (char *)malloc(image_data_size);
  gpuErrchk(cudaMalloc(&d_img_in, image_data_size));
  gpuErrchk(cudaMalloc(&d_img_out, image_data_size));
  gpuErrchk(cudaMemcpy(d_img_in, image_rgb, image_data_size, cudaMemcpyHostToDevice));
  const dim3 blockSize(16,16,1);
  const dim3 gridSize(cols/blockSize.x+1,rows/blockSize.y+1,1);
  posterize<<<gridSize, blockSize>>>(d_img_in, d_img_out, cols, rows, channels, colors);
  gpuErrchk(cudaFree(d_img_in));
  gpuErrchk(cudaMemcpy(h_img_out, d_img_out, image_data_size, cudaMemcpyDeviceToHost));
  return h_img_out;
}
