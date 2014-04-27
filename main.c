#include <string.h>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv)
{
  //uchar4 *h_image, *d_image;
  char* input_file;
  char* output_file;
  if (argc < 3) {
    printf("Provide input and output files.\n");
    exit(1);
  }
  input_file = argv[1];
  output_file = argv[2];
  IplImage* img = cvLoadImage(input_file, CV_LOAD_IMAGE_COLOR);
  size_t cols = img->width;
  size_t rows = img->height;
  char* rgb_image = malloc(3*cols*rows);
  rgb_image = img->imageData;

  
  cvReleaseImage(&img);
  return 0;
}