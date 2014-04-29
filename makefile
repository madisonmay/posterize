OPENCV_CFLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`

main: main.o posterize.o
	nvcc $(OPENCV_CFLAGS) -g -o main main.o posterize.o $(OPENCV_LIBS)

main.o: main.cu
	nvcc $(OPENCV_CFLAGS) -g -c main.cu -o main.o

posterize.o: posterize.cu
	nvcc -arch=sm_20 -g -c posterize.cu -o posterize.o

serial: serialmain.o serialposterize.o
	gcc $(OPENCV_CFLAGS) -g -o serialmain serialmain.o serialposterize.o $(OPENCV_LIBS)

serialmain.o: serialmain.c
	gcc -g -c serialmain.c -o serialmain.o

serialposterize: posterize.c
	gcc -g -c serialposterize.c -o serialmain.o	
