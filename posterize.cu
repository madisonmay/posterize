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
void mode(const unsigned char* input,
          unsigned char* output,
          size_t cols, size_t rows, int channels)
{
  int x = blockDim.x*blockIdx.x+threadIdx.x;
  int y = blockDim.y*blockIdx.y+threadIdx.y;
  if (x >= cols || y >= rows) {
      return;
  }

  int idx = y*cols+x;

  int dim = 9;

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

__global__
void smooth(const unsigned char* input,
            unsigned char* output,
            size_t cols, size_t rows, int channels, int n,
            int *hist)
{
  int x = blockDim.x*blockIdx.x+threadIdx.x;
  int y = blockDim.y*blockIdx.y+threadIdx.y;
  if (x >= cols || y >= rows) {
      return;
  }
  int id = y*cols+x;
  int w = 256/n;
  int i, j, windowSize = 5;
  int idx, idy;
  int size = n*n*n;
  int pixel_id;
  int r, g, b;

  for (i = 0; i < size; i++) {
    hist[i] = 0;   
  }

  int sum_r = 0;
  int sum_g = 0;
  int sum_b = 0;

  for (i = -windowSize/2; i<=windowSize/2; i++) {
    idy = min(max((y + i), 0), (int) rows);
    for (j = -windowSize/2; j<=windowSize/2; j++) {
      idx = min(max((x + j), 0), (int) cols);
      pixel_id = idy*cols + idx;
      r = input[pixel_id*channels+0]/w; sum_r += input[pixel_id*channels+0];
      g = input[pixel_id*channels+1]/w; sum_g += input[pixel_id*channels+1];
      b = input[pixel_id*channels+2]/w; sum_b += input[pixel_id*channels+2];
      hist[r*n*n + g*n + b]++;
    }
  }

  int max = 0;
  int max_index = 0;
  for (i = 0; i<size; i++) {
    if (hist[i] > max) {
      max_index = i;
      max = hist[i];
    }
  }

  // unsigned char mode_r = sum_r / (windowSize*windowSize);
  // unsigned char mode_g = sum_g / (windowSize*windowSize);
  // unsigned char mode_b = sum_b / (windowSize*windowSize);

  unsigned char mode_r = (unsigned char) (max_index/(n*n))*w+w/2;
  unsigned char mode_g = (unsigned char) ((max_index/n)%n)*w+w/2;
  unsigned char mode_b = (unsigned char) (max_index%(n*n))*w+w/2;

  output[id*channels+0] = mode_r;
  output[id*channels+1] = mode_g;
  output[id*channels+2] = mode_b;
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
  // posterize<<<gridSize, blockSize>>>(d_img_in, d_img_out, cols, rows, channels, colors);
  mode<<<gridSize, blockSize>>>(d_img_in, d_img_out, cols, rows, channels);
  gpuErrchk(cudaFree(d_img_in));
  gpuErrchk(cudaMemcpy(h_img_out, d_img_out, image_data_size, cudaMemcpyDeviceToHost));
  return h_img_out;
}
