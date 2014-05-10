#include "utils.h"
#include "posterize.h"

__global__
void reduce_kernel(const unsigned char* input, 
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
void mode_kernel(const unsigned char* input,
                 unsigned char* output,
                 size_t cols, size_t rows, int channels, int dim)
{
  int x = blockDim.x*blockIdx.x+threadIdx.x;
  int y = blockDim.y*blockIdx.y+threadIdx.y;
  if (x >= cols || y >= rows) {
      return;
  }

  int idx = y*cols+x;

  int offset, Offset = 0;

  int count = 0, maxCount = 0;

  unsigned char mode = NULL;

  int i, j, k, J, K = 0;

  // for each channel...
  for (i = 0; i < channels; i++) {
    // for every pixel per channel...
    for (j = -dim/2; j <= dim/2; j++){
      count = 0;
      for (k = -dim/2; k <= dim/2; k++) {
        offset = idx+cols*j+k;
        if ((x+k >= cols || y+j >= rows) || (x+k < 0 || y+j < 0)) {
          continue;
        }

        // compare it to every other pixel
        for (J = -dim/2; J <= dim/2; J++){
          for (K = -dim/2; K <= dim/2; K++) {
            Offset = idx+cols*J+K;
            if ((x+K >= cols || y+J >= rows) || (x+K < 0 || y+J < 0)) {
              continue;
            }

            if (input[offset*channels + i] == input[Offset*channels + i]) {
              count++;
            }
          }
        }
        if (count > maxCount) {
          maxCount = count;
          mode = input[offset*channels + i];
        }
      }
    }

    if (maxCount > 1) {
      output[idx*channels + i] = mode;
    }
    else {
      output[idx*channels + i] = input[idx*channels + i];
    }
    maxCount = 0;
  }
}

char* processPosterize(char* image_rgb, size_t cols, size_t rows, int channels, int colors)
{
  unsigned char *d_img_in;
  unsigned char *d_img_reduce;
  unsigned char *d_img_mode;
  char *h_img_out;
  size_t image_data_size = sizeof(unsigned char)*cols*rows*channels;
  h_img_out = (char *)malloc(image_data_size);
  gpuErrchk(cudaMalloc(&d_img_in, image_data_size));
  gpuErrchk(cudaMalloc(&d_img_reduce, image_data_size));
  gpuErrchk(cudaMemcpy(d_img_in, image_rgb, image_data_size, cudaMemcpyHostToDevice));
  const dim3 blockSize(16,16,1);
  const dim3 gridSize(cols/blockSize.x+1,rows/blockSize.y+1,1);
  reduce_kernel<<<gridSize, blockSize>>>(d_img_in, d_img_reduce, cols, rows, channels, colors);
  gpuErrchk(cudaFree(d_img_in));
  gpuErrchk(cudaMalloc(&d_img_mode, image_data_size));
  mode_kernel<<<gridSize, blockSize>>>(d_img_reduce, d_img_mode, cols, rows, channels, 5);
  gpuErrchk(cudaFree(d_img_reduce));
  gpuErrchk(cudaMemcpy(h_img_out, d_img_mode, image_data_size, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_img_mode));
  return h_img_out;
}
