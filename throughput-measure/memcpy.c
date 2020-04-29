#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define BUFFER_SIZE 100663296 
#define COPY_SIZE 65536
#define ITERATION 1000
#define BEBOP_CORES_PER_NUMA 18 
int main(int argc, char *argv[]){
	int remote_num = atoi(argv[1]);
	int rank, psize;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);
	
	/* prepare allocate data at NUMA node 0 */
	int bind_core = rank % BEBOP_CORES_PER_NUMA;
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(bind_core, &mask);
	sched_setaffinity(getpid(), sizeof(cpu_set_t), &mask);
	
	char *buffer0 = (char*) malloc(BUFFER_SIZE);
	memset(buffer0, 0x2, BUFFER_SIZE);
	char *buffer1 = (char*) malloc(BUFFER_SIZE);
	memset(buffer1, 0x1, BUFFER_SIZE);
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(rank >= psize - remote_num){
			cpu_set_t mask;
			CPU_ZERO(&mask);
			CPU_SET(rank - (psize - remote_num) + BEBOP_CORES_PER_NUMA, &mask);
			sched_setaffinity(getpid(), sizeof(cpu_set_t), &mask);
	}

	int i, j;	
	struct timespec start, end;
	size_t offset = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	clock_gettime(CLOCK_MONOTONIC, &start);
	for(i = 0; i < ITERATION; ++i){
		memcpy(buffer0 + offset, buffer1 + offset, COPY_SIZE);
		offset = (offset + (COPY_SIZE << 1)) > BUFFER_SIZE ? 0 : offset + COPY_SIZE;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	
	/* keep copying to maintain memory load and store traffic */
	offset = 0;
	for(j=0;j<16;++j){
			for(i = 0; i < ITERATION; ++i){
					memcpy(buffer0 + offset, buffer1 + offset, COPY_SIZE);
					offset = (offset + (COPY_SIZE << 1)) > BUFFER_SIZE ? 0 : offset + COPY_SIZE;
			}
	}
	double time = ((double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1.0e9) / ITERATION;
	double local_bdw = COPY_SIZE / time / 1024.0 / 1024.0 / 1024.0 ;// GB/s
	double total_bdw;
	MPI_Reduce(&local_bdw, &total_bdw, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(rank == 0){
			printf("%d %.3lf\n", psize - remote_num, total_bdw);
	}

	MPI_Finalize();
	return 0;
}
