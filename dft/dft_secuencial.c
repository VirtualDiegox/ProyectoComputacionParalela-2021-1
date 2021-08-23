#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/time.h>

#define e 2.71828182846
#define pi 3.14159265359

//basado en https://www.nayuki.io/page/how-to-implement-the-discrete-fourier-transform

void compute_dft_complex(const double complex input[], double complex output[], size_t n) {
	for (size_t k = 0; k < n; k++) {  // For each output element
		double complex sum = 0.0;
		for (size_t t = 0; t < n; t++) {  // For each input element
			double angle = 2 * M_PI * t * k / n;
			sum += input[t] * cexp(-angle * I);
		}
		output[k] = sum;
	}
}


int main(){
    struct timeval* tval_before,* tval_after,* tval_result;
    tval_before = (struct timeval*)malloc(sizeof(struct timeval));
    tval_after = (struct timeval*)malloc(sizeof(struct timeval));
    tval_result = (struct timeval*)malloc(sizeof(struct timeval));
    
    
    int n = 2047;
    double complex array[n];
    for(int i=0;i<n;i++){
        array[i]=rand()%100;
    }
    //array[0] = 2;
    //array[1] = 3;
    //array[2] = -1;
    //array[3] = 1;

    double complex outArray[n];
    gettimeofday(tval_before, NULL);
    compute_dft_complex(array,outArray,n+1);
    gettimeofday(tval_after, NULL);
    timersub(tval_after, tval_before, tval_result);
    printf("%ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
    /*for(int i = 0;i <= n;i++){
      printf("%f + %fi\n", creal(outArray[i]), cimag(outArray[i]));
    }*/
    return 0;
}