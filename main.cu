#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include "posterize.h"
#include "serialposterize.h"

int main(int argc, char **argv)
{
  //uchar4 *h_image, *d_image;
  char* input_file;
  char* output_file;
  char* command;
  int colors;
  if (argc < 4) {
    printf("Provide command to run, input, and output files.\n");
    exit(1);
  }
  if (argc == 5) {
    colors = atoi(argv[3]);
  } else {
    colors = 6;
  }
  input_file = argv[1];
  output_file = argv[2];
  command = argv[3];
  IplImage* img = cvLoadImage(input_file, CV_LOAD_IMAGE_COLOR);
  IplImage* out_img = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
  cvCopy(img, out_img, NULL);
  size_t cols = img->width;
  size_t rows = img->height;
  int channels = img->nChannels;
  char* image_rgb;
  image_rgb = img->imageData;
  char* out_image_rgb;
  if (strcmp(command,"serial-posterize")) {
    out_image_rgb = processSerialPosterize(image_rgb, cols, rows, channels, colors);
  } else if (strcmp(command, "posterize")) {
    out_image_rgb = processPosterize(image_rgb, cols, rows, channels, colors);
  } else {
    printf("Command '%s' is not valid", command);
    exit(1);
  }
  out_img->imageData = out_image_rgb;
  int p[3];
  p[0] = CV_IMWRITE_JPEG_QUALITY;
  p[1] = 95;
  p[2] = 0;
  cvSaveImage(output_file, out_img, p);
  cvReleaseImage(&img);
  return 0;
}