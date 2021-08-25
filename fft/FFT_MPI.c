#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>
#include "fft-complex.h"
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// Private function prototypes
static size_t reverse_bits(size_t val, int width);


bool Fft_transform(double complex vec[], size_t n, bool inverse,int argc, char *argv[]) {
	if (n == 0)
		return true;
	else if ((n & (n - 1)) == 0)  // Is power of 2
		return Fft_transformRadix2(vec, n, inverse,argc,argv);
	else{
        printf("input of size %ld is not a power of 2",n+1);
        return 1;
    }  // More complicated algorithm for arbitrary sizes

		
}


bool Fft_transformRadix2(double complex vec[], size_t n, bool inverse,int argc, char *argv[]) {
	// Length variables
	int levels = 0;  // Compute levels = floor(log2(n))
	for (size_t temp = n; temp > 1U; temp >>= 1)
		levels++;
	if ((size_t)1U << levels != n)
		return false;  // n is not a power of 2
	
	// Trigonometric tables
	if (SIZE_MAX / sizeof(double complex) < n / 2)
		return false;
	double complex *exptable = malloc((n / 2) * sizeof(double complex));
	if (exptable == NULL)
		return false;
	for (size_t i = 0; i < n / 2; i++)
		exptable[i] = cexp((inverse ? 2 : -2) * M_PI * i / n * I);
	
	// Bit-reversed addressing permutation
	for (size_t i = 0; i < n; i++) {
		size_t j = reverse_bits(i, levels);
		if (j > i) {
			double complex temp = vec[i];
			vec[i] = vec[j];
			vec[j] = temp;
		}
	}
	
	// Cooley-Tukey decimation-in-time radix-2 FFT
	/*for (size_t size = 2; size <= n; size *= 2) {
		size_t halfsize = size / 2;
		size_t tablestep = n / size;
		for (size_t i = 0; i < n; i += size) {
			for (size_t j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
				size_t l = j + halfsize;
				double complex temp = vec[l] * exptable[k];
				vec[l] = vec[j] - temp;
				vec[j] += temp;
			}
		}
        
        
		if (size == n)  // Prevent overflow in 'size *= 2'
			break;
	}*/

    int tasks, iam;
    struct timeval* tval_before,* tval_after,* tval_result;
    tval_before = (struct timeval*)malloc(sizeof(struct timeval));
    tval_after = (struct timeval*)malloc(sizeof(struct timeval));
    tval_result = (struct timeval*)malloc(sizeof(struct timeval));
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &iam);
    MPI_Status status;
    double complex AllVecs[n*tasks];
    int IndexToBeProcessed [n/2];
    int FromIndex = (floor((n/2)/tasks) * iam);
    int ToIndex = (floor((n/2)/tasks) * (iam+1))-1;
    if (iam == tasks-1)ToIndex=(n/2)-1;
    gettimeofday(tval_before, NULL);
    // Cooley-Tukey decimation-in-time radix-2 FFT
    for (size_t size = 2; size <= n; size *= 2) {
        size_t halfsize = size / 2;
        size_t tablestep = n / size;
        IndexToBeProcessed[0]= 0;
        for (int x = 1;x < n/2;x++){
            
            IndexToBeProcessed[x] = (IndexToBeProcessed[x-1]+1)%halfsize==0 ? IndexToBeProcessed[x-1] + halfsize + 1 :  IndexToBeProcessed[x-1] + 1;
            //printf("index %d to be processed: %d\n",x,IndexToBeProcessed[x]);

            if (x == n/2) break;
            
            
        }
        
        
        for(int j = FromIndex;j<=ToIndex;j++){
            size_t k = tablestep*(IndexToBeProcessed[j]%size);
            size_t l = IndexToBeProcessed[j] + halfsize;
            double complex temp = vec[l] * exptable[k];
            vec[l] = vec[IndexToBeProcessed[j]] - temp;
            vec[IndexToBeProcessed[j]] += temp;
            
        }
        //printf("before update, task number %d, vec: %f + %fi  %f + %fi  %f + %fi  %f + %fi\n",iam,creal(vec[0]), cimag(vec[0]),creal(vec[1]), cimag(vec[1]),creal(vec[2]), cimag(vec[2]),creal(vec[3]), cimag(vec[3]));
        MPI_Gather(vec,n,MPI_C_DOUBLE_COMPLEX,AllVecs,n,MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if(iam==0){
            
            int ProcessThatCalculatedI = 0;
            for(long i = 0;i<n;i++){
                for(long j = 0 ; j < n/2;j++){
                    if (IndexToBeProcessed[j] == i || IndexToBeProcessed[j] == i-halfsize){
                        ProcessThatCalculatedI = MIN(floor(j/((ToIndex-FromIndex+1))), tasks-1);
                        break;
                    }
                }
                vec[i]=AllVecs[ (ProcessThatCalculatedI*n) + i];
            }

        }

        MPI_Bcast(vec,n, MPI_C_DOUBLE_COMPLEX,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        //printf("after update, task number %d, vec: %f + %fi  %f + %fi  %f + %fi  %f + %fi\n",iam,creal(vec[0]), cimag(vec[0]),creal(vec[1]), cimag(vec[1]),creal(vec[2]), cimag(vec[2]),creal(vec[3]), cimag(vec[3]));
        if (size == n)break;
    }


    //if(iam == 0)for(int i = 0;i < n;i++)printf("%d: %f + %fi\n",i, creal(vec[i]), cimag(vec[i]));
    if(iam == 0)gettimeofday(tval_after, NULL);
	if(iam == 0)timersub(tval_after, tval_before, tval_result);
    if(iam == 0)printf("%ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
	
    
    MPI_Finalize();
	free(exptable);
	return true;
}


static size_t reverse_bits(size_t val, int width) {
	size_t result = 0;
	for (int i = 0; i < width; i++, val >>= 1)
		result = (result << 1) | (val & 1U);
	return result;
}


int main(int argc, char *argv[]){
    
    /*
    size_t n = 3;
    double complex array[n+1];
    array[0] = 2;
    array[1] = 3;
    array[2] = -1;
    array[3] = 1;
    
    array[4] = 2;
    array[5] = 3;
    array[6] = -1;
    array[7] = 1;
    */
    
    size_t n = 16383;
    double complex array[n+1];
    for(int i=0;i<n+1;i++){
        array[i]=rand()%100;
    }
    
    Fft_transform(array,n+1,false,argc,argv);
    /*
    for(int i = 0;i <= n;i++){

      printf("%f + %fi\n", creal(array[i]), cimag(array[i]));
    }           
    */
   
    return 0;
}
