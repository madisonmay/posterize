OPENCV_CFLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`
main: main.cu
	nvcc $(OPENCV_CFLAGS) -g -o main main.cu $(OPENCV_LIBS)
