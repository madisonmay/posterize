OPENCV_CFLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`

main: main.c posterize.o
	gcc $(OPENCV_CFLAGS) -g -o main main.c $(OPENCV_LIBS)

posterize.o: posterize.cu
	nvcc -c posterize.cu -o posterize.o
