#include <string.h>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include "posterize.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

char* process(char* image_rgb, size_t cols, size_t rows, int colors)
{
  unsigned char *d_img_in;
  unsigned char *d_img_out;
  char *h_img_out;
  size_t image_data_size = sizeof(unsigned)*cols*rows*3;
  h_img_out = (char *)malloc(image_data_size);
  gpuErrchk(cudaMalloc(&d_img_in, image_data_size));
  gpuErrchk(cudaMalloc(&d_img_out, image_data_size));
  gpuErrchk(cudaMemcpy(d_img_in, image_rgb, image_data_size, cudaMemcpyHostToDevice));
  const dim3 blockSize(8,8,1);
  const dim3 gridSize(cols/blockSize.x+1,rows/blockSize.y+1,1);
  posterize<<<gridSize, blockSize>>>(d_img_in, d_img_out, cols, rows, colors);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaMemcpy(h_img_out, d_img_out, image_data_size, cudaMemcpyDeviceToHost));
  return h_img_out;
}

int main(int argc, char **argv)
{
  //uchar4 *h_image, *d_image;
  char* input_file;
  char* output_file;
  int colors;
  if (argc < 3) {
    printf("Provide input and output files.\n");
    exit(1);
  }
  if (argc == 4) {
    colors = atoi(argv[3]);
  } else {
    colors = 6;
  }
  input_file = argv[1];
  output_file = argv[2];
  IplImage* img = cvLoadImage(input_file, CV_LOAD_IMAGE_COLOR);
  IplImage* out_img = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
  cvCopy(img, out_img, NULL);
  size_t cols = img->width;
  size_t rows = img->height;
  char* image_rgb;
  image_rgb = img->imageData;
  char* out_image_rgb = process(image_rgb, cols, rows, colors);
  out_img->imageData = out_image_rgb;
  int p[3];
  p[0] = CV_IMWRITE_JPEG_QUALITY;
  p[1] = 95;
  p[2] = 0;
  cvSaveImage(output_file, out_img, p);
  cvReleaseImage(&img);
  return 0;
}