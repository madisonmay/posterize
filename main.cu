#include <string.h>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include "posterize.h"

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