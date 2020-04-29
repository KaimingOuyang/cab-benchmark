#include "mpi.h"
#include "stdio.h"
#include <stdlib.h>
#include <string.h>
#if defined TEST
#define TOTLE_TILE_SIZE 1048576 //1MB
#else
#define TOTLE_TILE_SIZE 1073741824 //1GB
#endif


int main(int argc, char *argv[])
{
    int rank, nprocs;
    MPI_Win win;
    int errs = 0;

    MPI_Aint i, j, k;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int tile_dim = atoi(argv[1]);
    int iterations = atoi(argv[2]);
    int total_elem_tile = TOTLE_TILE_SIZE / sizeof(double);
    int elem_tile = tile_dim * tile_dim;

    double *dest_mx_buf, *src_mx_buf;
    int target_rank = rank ^ 1;
    
    int warm_up_iter = 100;
    
    if(rank == 0 || rank == 1){
        src_mx_buf = (double*) malloc(TOTLE_TILE_SIZE);
        MPI_Win_allocate(TOTLE_TILE_SIZE, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, (void*) &dest_mx_buf, &win);   
        memset(src_mx_buf, 0x01, TOTLE_TILE_SIZE);
        memset(dest_mx_buf, 0x01, TOTLE_TILE_SIZE);
    }else{
        MPI_Win_allocate(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, (void*) &dest_mx_buf, &win);
    }
    
	int iter = total_elem_tile / elem_tile;
    double max_time;
    /* warm up */
	if (rank == 0)
	{  
			for (k = 0; k < warm_up_iter; ++k) {
					//MPI_Win_fence(0, win);
					MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, win);
					for (i = 0; i < iter; i += 1){
							MPI_Accumulate(src_mx_buf + i * elem_tile, elem_tile, MPI_DOUBLE, target_rank, i * elem_tile, elem_tile, MPI_DOUBLE, MPI_SUM, win);
							MPI_Win_flush(target_rank, win);
					}
					MPI_Win_unlock(target_rank, win);
			}
			//MPI_Win_fence(0, win);
	}

    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();
	if (rank == 0)
	{  
			for (k = 0; k < iterations; ++k) {
					MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, win);
					for (i = 0; i < iter; i += 1){
							MPI_Accumulate(src_mx_buf + i * elem_tile, elem_tile, MPI_DOUBLE, target_rank, i * elem_tile, elem_tile, MPI_DOUBLE, MPI_SUM, win);
							MPI_Win_flush(target_rank, win);
					}
					MPI_Win_unlock(target_rank, win);
			}
	}
    MPI_Barrier(MPI_COMM_WORLD);
    
    time = MPI_Wtime() - time;
    time = time * 1e6 / iterations / iter; //ms
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%d %lf\n", nprocs, max_time);
        fflush(stdout);
    }
    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
