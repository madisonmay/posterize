#include <stdlib.h>
#include <stdio.h>

int main(){
	size_t free, total;
	printf("\n");
	cudaMemGetInfo(&free,&total); 
	printf("%d KB free of total %d KB\n",free/1024,total/1024);
}
