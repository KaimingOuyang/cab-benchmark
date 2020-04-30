#define _GNU_SOURCE
#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>

#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <unistd.h>

#define LEN 65536
#define BUFFER_SIZE 100663296
void output_affinity(int rank){
	cpu_set_t mask;
	CPU_ZERO(&mask);
	sched_getaffinity(0, sizeof(cpu_set_t), &mask);
	int nproc = sysconf(_SC_NPROCESSORS_ONLN);
	int i;
	char buffer[64];
	for (i = 0; i < nproc; i++) {
		sprintf(buffer+i, "%d", CPU_ISSET(i, &mask));
	}
	printf("rank %d -  %s\n", rank, buffer);
	fflush(stdout);
}

#define IMB_P2P_CACHE_LINE_LEN 64
static inline void touch_send_buff(size_t size, char * send_buffer) {
		static size_t dummy = 0x11223344;
		size_t i = 0;
		while (i < size) {
				send_buffer[i] = (char)dummy;
				dummy++;
				i += IMB_P2P_CACHE_LINE_LEN;
		}
}

static inline void touch_recv_buff(size_t size, char * recv_buffer) {
		static size_t dummy = 0x11223344;
		size_t i = 0;
		//int *tmp = (int*) recv_buffer;
		//int cnt = size / sizeof(int);
		while (i < size) {
				dummy += recv_buffer[i];
				i += IMB_P2P_CACHE_LINE_LEN;
		}
}

int main(int argc, char* argv[])
{
	int rank, psize;
	long i,j;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);
	const int iter = atoi(argv[1]);
	const size_t msg_sz = atol(argv[2]);
	const int out_flag = atol(argv[3]);
	//const int bind_core = atoi(argv[4]);
	/* outflag = 1 -> process.out
	 * outflag = 0 -> message_psize.out
	 * */
	char *data_send = NULL;
	char *data_recv = NULL;
	if(rank == 0 || rank == 1){
		data_send = (char*) malloc(BUFFER_SIZE);
		data_recv = (char*) malloc(BUFFER_SIZE);
		memset(data_send, 0, BUFFER_SIZE);
		memset(data_recv, 0, BUFFER_SIZE);
	}
	
	size_t offset = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	offset = 0;
	double time = 0;
	for(i=0;i<iter;++i){
			char *cur_send_buf = data_send + offset;
			char *cur_recv_buf = data_recv + offset;
			if(rank == 0){
					MPI_Send(cur_send_buf, msg_sz, MPI_CHAR, 1, 777, MPI_COMM_WORLD);
					MPI_Recv(cur_recv_buf, msg_sz, MPI_CHAR, 1, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					time -= MPI_Wtime();
					touch_recv_buff(msg_sz, cur_recv_buf);
					time += MPI_Wtime();
			}else if (rank == 1){
					MPI_Recv(cur_recv_buf, msg_sz, MPI_CHAR, 0, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					time -= MPI_Wtime();
					touch_recv_buff(msg_sz, cur_recv_buf);
					time += MPI_Wtime();
					MPI_Send(cur_send_buf, msg_sz, MPI_CHAR, 0, 777, MPI_COMM_WORLD);
			}
			offset = (offset + (msg_sz << 1)) > BUFFER_SIZE ? 0 : (offset + msg_sz);
	}
	time /= iter;
	double max_time;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	if(rank == 0){
		if(out_flag == 0){
			// output data psize 
			printf("%ld %.3lf\n", msg_sz, max_time * 1e6);
			fflush(stdout);
		}else if(out_flag == 1){
			// output process number 
			printf("%ld %.3lf\n", psize, max_time * 1e6);
			fflush(stdout);
		}else if(out_flag == 2){
			// output iteration 
			printf("%ld %.3lf\n", iter, max_time * 1e6);
			fflush(stdout);
		}
	} 
	MPI_Finalize();
	return 0;
}
