#include "serialmode.h"

char* processSerialMode(char* image_rgb, size_t cols, size_t rows, int channels, int colors)
{
  char *image_out;
  size_t image_data_size = sizeof(unsigned char)*cols*rows*channels;
  image_out = (char *)malloc(image_data_size);

  int dim = 9;
  int offset, Offset = 0;
  int count = 0, maxCount = 0;
  unsigned char mode = (char) NULL;
  int idx, x, y, i, j, k, J, K = 0;
  // for each pixel
  for (idx = 0; idx < rows*cols; idx++) {
    x = idx%cols;
    y = idx%rows;
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

              if (image_rgb[offset*channels + i] == image_rgb[Offset*channels + i]) {
                count++;
              }
            }
          }
          if (count > maxCount) {
            maxCount = count;
            mode = image_rgb[offset*channels + i];
          }
        }
      }

      if (maxCount > 1) {
        image_out[idx*channels + i] = mode;
      }
      else {
        image_out[idx*channels + i] = image_rgb[idx*channels + i];
      }
      maxCount = 0;
    }
  }
  return image_out;
}