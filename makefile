OPENCV_CFLAGS = `pkg-config --cflags opencv`
OPENCV_LIBS = `pkg-config --libs opencv`
main: main.c
	gcc $(OPENCV_CFLAGS) -g -o main main.c $(OPENCV_LIBS)
