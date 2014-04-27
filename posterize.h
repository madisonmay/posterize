
__global__
void posterize(const unsigned char* const input, 
               unsigned char* const output, 
               size_t cols, size_t rows, int n);