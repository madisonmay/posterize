OPENCV_CFLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`

main: main.o posterize.o
	nvcc $(OPENCV_CFLAGS) -g -o main main.o posterize.o $(OPENCV_LIBS)

main.o: main.cu
	nvcc $(OPENCV_CFLAGS) -g -c main.cu -o main.o

posterize.o: posterize.cu
	nvcc -g -c posterize.cu -o posterize.o
