#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "fft-complex.h"
#include <sys/time.h>

// Private function prototypes
static size_t reverse_bits(size_t val, int width);
static void *memdup(const void *src, size_t n);


bool Fft_transform(double complex vec[], size_t n, bool inverse) {
	if (n == 0)
		return true;
	else if ((n & (n - 1)) == 0) {
		return Fft_transformRadix2(vec, n, inverse);

	} // Is power of 2	
	else  // More complicated algorithm for arbitrary sizes
		return printf("NO es potencia de 2");
}


bool Fft_transformRadix2(double complex vec[], size_t n, bool inverse) {
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
	for (size_t size = 2; size <= n; size *= 2) {
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
	}
	
	free(exptable);
	return true;
}



static size_t reverse_bits(size_t val, int width) {
	size_t result = 0;
	for (int i = 0; i < width; i++, val >>= 1)
		result = (result << 1) | (val & 1U);
	return result;
}



int main(){
    //size_t n = 3;
    
	struct timeval* tval_before,* tval_after,* tval_result;
    tval_before = (struct timeval*)malloc(sizeof(struct timeval));
    tval_after = (struct timeval*)malloc(sizeof(struct timeval));
    tval_result = (struct timeval*)malloc(sizeof(struct timeval));


    //double complex array[n];
    //array[0] = 2;
    //array[1] = 3;
    //array[2] = -1;
    //array[3] = 1;

	size_t n = 4194304;
    double complex array[n+1];
    for(int i=0;i<n+1;i++){
        array[i]=rand()%100;
    }
	


	gettimeofday(tval_before, NULL);
    Fft_transform(array,n+1,false);
	gettimeofday(tval_after, NULL);
	timersub(tval_after, tval_before, tval_result);
    printf("%ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
		
    //for(int i = 0;i <= n;i++){
      //printf("%f + %fi\n", creal(array[i]), cimag(array[i]));
    //}
    return 0;
}

