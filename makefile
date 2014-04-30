OPENCV_CFLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`
OBJS = serialposterize.o posterize.o main.o

main: $(OBJS)
	nvcc $(OBJS) $(OPENCV_CFLAGS) -g -o $@ $(OPENCV_LIBS)

main.o: main.cu 
	nvcc $(OPENCV_CFLAGS) -g -c main.cu -o $@

posterize.o: posterize.cu
	nvcc -arch=sm_20 -g -c posterize.cu -o $@

serialposterize.o: serialposterize.c
	gcc -g -c serialposterize.c -o $@