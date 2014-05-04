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
                 size_t cols, size_t rows, int channels,
                 int colors, unsigned int* hist)
{
  int x = blockDim.x*blockIdx.x+threadIdx.x;
  int y = blockDim.y*blockIdx.y+threadIdx.y;
  if (x >= cols || y >= rows) {
      return;
  }

  int idx = y*cols+x;

  int dim = 11;
  int offset;

  int i, j, k ;
  int w = 256/colors;
  int iter, base, maxColor, maxCount = 0; 

  // for each channel...
  for (i = 0; i < channels; i++) {
    // for every pixel per channel...
    base = idx*colors*channels+colors*i;
    for (iter = 0; iter < colors; iter++) {
      hist[base+iter] = 0;
    }

    maxColor = input[idx*channels+i]/w;

    for (j = -dim/2; j <= dim/2; j++){
      for (k = -dim/2; k <= dim/2; k++) {

        offset = idx+cols*j+k;
        if ((x+k >= cols || y+j >= rows) || (x+k < 0 || y+j < 0)) {
          continue;
        }

        unsigned char color = input[offset*channels+i]/w;
        hist[base+color]++;
      }
    }

    for (iter = 0; iter < colors; iter++) {
      if (hist[base+iter] > maxCount) {
        maxColor = iter;
        maxCount = hist[base+iter];
      }
    }

    output[idx*channels + i] = maxColor*w+w/2;

    maxCount = 0;
  }
}

char* processPosterize(char* image_rgb, size_t cols, size_t rows, int channels, int colors)
{
  unsigned char *d_img_in;
  unsigned char *d_img_reduce;
  unsigned char *d_img_mode;
  unsigned int  *d_hist;
  char *h_img_out;
  size_t image_data_size = sizeof(unsigned char)*cols*rows*channels;
  size_t hist_size = sizeof(unsigned int)*cols*rows*channels*colors;
  h_img_out = (char *)malloc(image_data_size);
  gpuErrchk(cudaMalloc(&d_img_in, image_data_size));
  gpuErrchk(cudaMalloc(&d_img_reduce, image_data_size));
  gpuErrchk(cudaMemcpy(d_img_in, image_rgb, image_data_size, cudaMemcpyHostToDevice));
  const dim3 blockSize(16,16,1);
  const dim3 gridSize(cols/blockSize.x+1,rows/blockSize.y+1,1);
  reduce_kernel<<<gridSize, blockSize>>>(d_img_in, d_img_reduce, cols, rows, channels, colors);
  gpuErrchk(cudaFree(d_img_in));
  gpuErrchk(cudaMalloc(&d_img_mode, image_data_size));
  gpuErrchk(cudaMalloc(&d_hist, hist_size));
  mode_kernel<<<gridSize, blockSize>>>(d_img_reduce, d_img_mode, cols, rows, channels, colors, d_hist);
  gpuErrchk(cudaFree(d_img_reduce));
  gpuErrchk(cudaMemcpy(h_img_out, d_img_mode, image_data_size, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_img_mode));
  return h_img_out;
}
