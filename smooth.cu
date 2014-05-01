#include "utils.h"
#include "smooth.h"

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

char* processSmooth(char* image_rgb, size_t cols, size_t rows, int channels, int colors)
{
  unsigned char *d_img_in;
  unsigned char *d_img_out;
  char *h_img_out;
  size_t image_data_size = sizeof(unsigned char)*cols*rows*channels;
  h_img_out = (char *)malloc(image_data_size);
  int *hist;
  gpuErrchk(cudaMalloc(&d_img_in, image_data_size));
  gpuErrchk(cudaMalloc(&d_img_out, image_data_size));
  gpuErrchk(cudaMalloc(&hist, colors*colors*colors));
  gpuErrchk(cudaMemcpy(d_img_in, image_rgb, image_data_size, cudaMemcpyHostToDevice));
  const dim3 blockSize(16,16,1);
  const dim3 gridSize(cols/blockSize.x+1,rows/blockSize.y+1,1);
  smooth<<<gridSize, blockSize>>>(d_img_in, d_img_out, cols, rows, channels, colors, hist);
  gpuErrchk(cudaFree(d_img_in));
  gpuErrchk(cudaMemcpy(h_img_out, d_img_out, image_data_size, cudaMemcpyDeviceToHost));
  return h_img_out;
}