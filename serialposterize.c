#include "serialposterize.h"

char* processSerialPosterize(char* image_rgb, size_t cols, size_t rows, int channels, int colors)
{
  char *image_out;
  size_t image_data_size = sizeof(unsigned char)*cols*rows*channels;
  image_out = (char *)malloc(image_data_size);
  int w = 256/colors;
  int i;
  for (i=0;i<rows*cols*3;i++) {
    image_out[i] = (image_rgb[i]/w)*w+w/2;
    // image_out[i*channels+1] = (image_rgb[i*channels+1]/w)*w+w/2;
    // image_out[i*channels+2] = (image_rgb[i*channels+2]/w)*w+w/2;
  }
  return image_out;
}