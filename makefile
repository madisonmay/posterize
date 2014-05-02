OPENCV_CFLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`
OBJS = smooth.o mode.o serialposterize.o posterize.o main.o

main: $(OBJS)
	nvcc $(OBJS) $(OPENCV_CFLAGS) -g -o $@ $(OPENCV_LIBS)

main.o: main.cu 
	nvcc $(OPENCV_CFLAGS) -g -c $^ -o $@

posterize.o: posterize.cu
	nvcc -arch=sm_20 -g -c $^ -o $@

serialposterize.o: serialposterize.c
	gcc -g -c $^ -o $@

mode.o: mode.cu
	nvcc -g -c $^ -o $@

smooth.o: smooth.cu
	nvcc -g -c $^ -o $@

serialmode.o: serialmode.c
	gcc -g -c $^ -o $@