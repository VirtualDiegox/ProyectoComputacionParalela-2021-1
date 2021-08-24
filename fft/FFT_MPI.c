#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>
#include "fft-complex.h"


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
    int IndexToBeProcessed [n/2];
    int FromIndex = (floor((n/2)/tasks) * iam);
    int ToIndex = (floor((n/2)/tasks) * (iam+1))-1;
    //printf("task %d FromIndex: %d ToIndex: %d\n",iam,FromIndex,ToIndex);
        gettimeofday(tval_before, NULL);
        // Cooley-Tukey decimation-in-time radix-2 FFT
        for (size_t size = 2; size <= n; size *= 2) {
            size_t halfsize = size / 2;
            size_t tablestep = n / size;
            
            if(iam == 0){
                IndexToBeProcessed[0]= 0;
                for (int x = 1;x < n/2;x++){
                    
                    IndexToBeProcessed[x] = (IndexToBeProcessed[x-1]+1)%halfsize==0 ? IndexToBeProcessed[x-1] + halfsize + 1 :  IndexToBeProcessed[x-1] + 1;
                    //printf("index %d to be processed: %d\n",x,IndexToBeProcessed[x]);

                    if (x == n/2) break;
                    
                    
                }
                  
            }    
            
            
           
            MPI_Barrier(MPI_COMM_WORLD);
            
            MPI_Bcast(IndexToBeProcessed,n/2, MPI_INT,0,MPI_COMM_WORLD);
            
            for(int j = FromIndex;j<=ToIndex;j++){
                size_t k = tablestep*(IndexToBeProcessed[j]%size);
                size_t l = IndexToBeProcessed[j] + halfsize;
				double complex temp = vec[l] * exptable[k];
				vec[l] = vec[IndexToBeProcessed[j]] - temp;
				vec[IndexToBeProcessed[j]] += temp;
                
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            for (int x = 0;x < (int)n/2;x++){
                
                double real_temp,imag_temp;

                int sender=floor(x/((n/2)/tasks));
                

                //Envio y recepcion del primer complejo
                if( iam == sender && iam != 0){
                    real_temp = creal(vec[IndexToBeProcessed[x]]);
                    MPI_Ssend(&real_temp, 1,  MPI_DOUBLE, 0, sender, MPI_COMM_WORLD);
                }
                if(iam == 0 && sender != 0){
                    MPI_Recv(&real_temp, 1, MPI_DOUBLE, sender,sender, MPI_COMM_WORLD, &status);
                }
                if( iam == sender && iam != 0){
                    imag_temp = cimag(vec[IndexToBeProcessed[x]]);
                    MPI_Ssend(&imag_temp, 1,  MPI_DOUBLE, 0, sender, MPI_COMM_WORLD);
                }
                if(iam == 0 && sender != 0){
                    MPI_Recv(&imag_temp, 1, MPI_DOUBLE, sender, sender, MPI_COMM_WORLD, &status);
                    vec[IndexToBeProcessed[x]] =  real_temp + imag_temp * I;
                }
                MPI_Barrier(MPI_COMM_WORLD);
                //Envio y recepcion del segundo complejo
                
                if( iam == sender && iam != 0){
                    real_temp = creal(vec[IndexToBeProcessed[x]+halfsize]);
                    MPI_Ssend(&real_temp, 1,  MPI_DOUBLE, 0, sender, MPI_COMM_WORLD);
                }
                if(iam == 0 && sender != 0){
                    MPI_Recv(&real_temp, 1, MPI_DOUBLE, sender, sender, MPI_COMM_WORLD, &status);
                }
                if( iam == sender && iam != 0){
                    imag_temp = cimag(vec[IndexToBeProcessed[x]+halfsize]);
                    MPI_Ssend(&imag_temp, 1,  MPI_DOUBLE, 0, sender, MPI_COMM_WORLD);
                }
                if(iam == 0 && sender != 0){
                    MPI_Recv(&imag_temp, 1, MPI_DOUBLE, sender, sender, MPI_COMM_WORLD, &status);
                    vec[IndexToBeProcessed[x]+halfsize] =  real_temp + imag_temp * I;
                }
                
                
                
                
                MPI_Barrier(MPI_COMM_WORLD);

                
               
              
               
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(vec,n, MPI_C_COMPLEX,0,MPI_COMM_WORLD);
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
    //size_t n = 7;
    
    //double complex array[n+1];
    //array[0] = 2;
    //array[1] = 3;
    //array[2] = -1;
    //array[3] = 1;
    
    //array[4] = 2;
    //array[5] = 3;
    //array[6] = -1;
    //array[7] = 1;

    size_t n = 65535;
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
