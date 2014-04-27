#include "posterize.h"

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
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

__global__
void smooth(const unsigned char* input,
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

  output[idx*3+0] = input[idx*3+0];
  output[idx*3+1] = input[idx*3+1];
  output[idx*3+2] = input[idx*3+2];
}

char* process(char* image_rgb, size_t cols, size_t rows, int channels, int colors)
{
  unsigned char *d_img_in;
  unsigned char *d_img_out;
  unsigned char *d_smooth_out;
  char *h_img_out;
  size_t image_data_size = sizeof(unsigned char)*cols*rows*channels;
  h_img_out = (char *)malloc(image_data_size);
  gpuErrchk(cudaMalloc(&d_img_in, image_data_size));
  gpuErrchk(cudaMalloc(&d_img_out, image_data_size));
  gpuErrchk(cudaMalloc(&d_smooth_out, image_data_size));
  gpuErrchk(cudaMemcpy(d_img_in, image_rgb, image_data_size, cudaMemcpyHostToDevice));
  const dim3 blockSize(8,8,1);
  const dim3 gridSize(cols/blockSize.x+1,rows/blockSize.y+1,1);
  posterize<<<gridSize, blockSize>>>(d_img_in, d_img_out, cols, rows, channels, colors);
  gpuErrchk(cudaDeviceSynchronize());
  smooth<<<gridSize, blockSize>>>(d_img_out, d_smooth_out, cols, rows, channels, colors);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_img_out, d_smooth_out, image_data_size, cudaMemcpyDeviceToHost));
  return h_img_out;
}
