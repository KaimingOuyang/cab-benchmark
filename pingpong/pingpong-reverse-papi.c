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

#include <papi.h>

#define LEN 65536
#define BUFFER_SIZE 100663296
void output_affinity(int rank) {
	cpu_set_t mask;
	CPU_ZERO(&mask);
	sched_getaffinity(0, sizeof(cpu_set_t), &mask);
	int nproc = sysconf(_SC_NPROCESSORS_ONLN);
	int i;
	char buffer[64];
	for (i = 0; i < nproc; i++) {
		sprintf(buffer + i, "%d", CPU_ISSET(i, &mask));
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

#define NUM_EVENT 3
double global_values[NUM_EVENT] = {0.0, 0.0, 0.0};

static inline void touch_recv_buff(size_t size, char * recv_buffer) {
	static size_t dummy = 0x11223344;
	size_t i = 0;
	int error;
    int events[NUM_EVENT] = {PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM};
    long long local_values[NUM_EVENT];
	error = PAPI_start_counters(events, NUM_EVENT);
	while (i < size) {
		dummy += recv_buffer[i];
		i += IMB_P2P_CACHE_LINE_LEN;
	}
	error = PAPI_stop_counters(local_values, NUM_EVENT);
	if (error != PAPI_OK) {
		printf("%s\n", PAPI_strerror(error));
		fflush(stdout);
	}

	for (i = 0; i < NUM_EVENT; i++)
		global_values[i] += (double) local_values[i];

}

int main(int argc, char* argv[])
{
	int rank, psize;
	long i, j;
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
	if (rank == 0 || rank == 1) {
		data_send = (char*) malloc(BUFFER_SIZE);
		data_recv = (char*) malloc(BUFFER_SIZE);
		memset(data_send, 0, BUFFER_SIZE);
		memset(data_recv, 0, BUFFER_SIZE);
	}

	size_t offset = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	offset = 0;
	double time = MPI_Wtime();
	for (i = 0; i < iter; ++i) {
		char *cur_send_buf = data_send + offset;
		char *cur_recv_buf = data_recv + offset;
		if (rank == 0) {
			MPI_Send(cur_send_buf, msg_sz, MPI_CHAR, 1, 777, MPI_COMM_WORLD);
			MPI_Recv(cur_recv_buf, msg_sz, MPI_CHAR, 1, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			touch_recv_buff(msg_sz, cur_recv_buf);
		} else if (rank == 1) {
			MPI_Recv(cur_recv_buf, msg_sz, MPI_CHAR, 0, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			touch_recv_buff(msg_sz, cur_recv_buf);
			MPI_Send(cur_send_buf, msg_sz, MPI_CHAR, 0, 777, MPI_COMM_WORLD);
		}
		offset = (offset + (msg_sz << 1)) > BUFFER_SIZE ? 0 : (offset + msg_sz);
	}
	time = (MPI_Wtime() - time) / iter;
	
	MPI_Barrier(MPI_COMM_WORLD);
	double max_events[NUM_EVENT];
	MPI_Reduce(global_values, max_events, NUM_EVENT, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(rank == 0){
		printf("%ld ", msg_sz);
		for(i=0;i<NUM_EVENT;++i){
		    if(i != NUM_EVENT - 1)
				printf("%.0lf ", global_values[i] / iter / 1000.0);
			else
				printf("%.0lf", global_values[i] / iter / 1000.0);
				
		}
		printf("\n");
		fflush(stdout);	
	}
	MPI_Finalize();
	return 0;
}
