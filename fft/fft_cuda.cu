#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <sys/time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "FftComplex.hpp"

using std::complex;
using std::cout;
using std::endl;
using std::vector;
using std::size_t;
using std::uintmax_t;


__global__ void transformCUDA(int *bloques,size_t *n,size_t *size,size_t *halfsize,size_t *tablestep, complex<double> *vec,complex<double> *expTable){
	int index = threadIdx.x + blockIdx.x * blockDim.x; 
    int hilos = blockDim.x * (*bloques);

	for (size_t i = 0; i < *n; i += *size) {
		for (size_t j = i, k = 0; j < i + *halfsize; j++, k += *tablestep) {
			complex<double> temp = vec[j + *halfsize] * expTable[k];
			vec[j + *halfsize] = vec[j] - temp;
			vec[j] += temp;
		}
	}


}


// Private function prototypes
static size_t reverseBits(size_t val, int width);
static void testFft(int n,int hilos,int bloques);

static thrust::host_vector<complex<double> > randomComplexes(int n);


void Fft::transform(thrust::host_vector<complex<double> > &vec, bool inverse,int hilos,int bloques) {
	size_t n = vec.size();
	if (n == 0)
		return;
	else if ((n & (n - 1)) == 0)  // Is power of 2
		Fft::transformRadix2(vec, inverse , hilos,bloques);
	else  // More complicated algorithm for arbitrary sizes
		printf("is not power of 2\n");
}


void Fft::transformRadix2(thrust::host_vector<complex<double> > &vec, bool inverse,int hilos,int bloques ) {
	// Length variables
	size_t n = vec.size();
	int levels = 0;  // Compute levels = floor(log2(n))
	for (size_t temp = n; temp > 1U; temp >>= 1)
		levels++;
	if (static_cast<size_t>(1U) << levels != n)
		throw std::domain_error("Length is not a power of 2");
	
	// Trigonometric table
	thrust::host_vector<complex<double> > expTable(n / 2);
	for (size_t i = 0; i < n / 2; i++)
		expTable[i] = std::polar(1.0, (inverse ? 2 : -2) * M_PI * i / n);
	
	// Bit-reversed addressing permutation
	for (size_t i = 0; i < n; i++) {
		size_t j = reverseBits(i, levels);
		if (j > i)
			std::swap(vec[i], vec[j]);
	}


	//CUDA

	int *d_bloques;
	size_t *d_n;
	cudaMalloc((void **)&d_bloques, sizeof(int));
	cudaMemcpy(d_bloques, &bloques, sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_n, sizeof(size_t));
	cudaMemcpy(d_n, &n, sizeof(size_t), cudaMemcpyHostToDevice);
	thrust::device_vector<complex<double> > d_expTable = expTable;
	thrust::device_vector<complex<double> > d_vec = vec;
	complex<double> * d_expTable_pointer = thrust::raw_pointer_cast(&d_expTable[0]);
	complex<double> * d_vec_pointer = thrust::raw_pointer_cast(&d_vec[0]);



	


	
	// Cooley-Tukey decimation-in-time radix-2 FFT
	for (size_t size = 2; size <= n; size *= 2) {
		size_t halfsize = size / 2;
		size_t tablestep = n / size;
		size_t *d_halfsize,*d_tablestep,*d_size;
		cudaMalloc((void **)&d_halfsize, sizeof(size_t));
		cudaMalloc((void **)&d_tablestep, sizeof(size_t));
		cudaMalloc((void **)&d_size, sizeof(size_t));

		cudaMemcpy(d_halfsize, &halfsize, sizeof(size_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_tablestep, &tablestep, sizeof(size_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_size, &size, sizeof(size_t), cudaMemcpyHostToDevice);

		cudaStream_t stream;
    	cudaStreamCreate(&stream);
		transformCUDA<<<bloques,hilos,0,stream>>>(d_bloques,d_n,d_size,d_halfsize,d_tablestep,d_vec_pointer,d_expTable_pointer);
		cudaStreamSynchronize(stream);
		
		for (size_t i = 0; i < n; i += size) {
			for (size_t j = i, k = 0; j < i + halfsize; j++, k += tablestep) {
				complex<double> temp = vec[j + halfsize] * expTable[k];
				vec[j + halfsize] = vec[j] - temp;
				vec[j] += temp;
			}
		}
		if (size == n)  // Prevent overflow in 'size *= 2'
			break;
	}
	cudaFree(d_n);
	cudaFree(d_bloques);
}




static size_t reverseBits(size_t val, int width) {

	size_t result = 0;
	for (int i = 0; i < width; i++, val >>= 1)
		result = (result << 1) | (val & 1U);
	return result;
}





int main(int argc, char *argv[]) {

	int hilos = atoi(argv[1]);
	int bloques = atoi(argv[2]);
	
	
	size_t n = 65536;
	
	
	testFft(n,hilos, bloques);
	
	
	return 0;
}


static void testFft(int n,int hilos,int bloques) {
	struct timeval* tval_before,* tval_after,* tval_result;
    tval_before = (struct timeval*)malloc(sizeof(struct timeval));
    tval_after = (struct timeval*)malloc(sizeof(struct timeval));
    tval_result = (struct timeval*)malloc(sizeof(struct timeval));
	const thrust::host_vector<complex<double> > input = randomComplexes(n);
	
	thrust::host_vector<complex<double> > actual = input;
	gettimeofday(tval_before, NULL);
	Fft::transform(actual, false,hilos, bloques);
	gettimeofday(tval_after, NULL);
	timersub(tval_after, tval_before, tval_result);
    printf("%ld.%06ld\n", (long int)tval_result->tv_sec, (long int)tval_result->tv_usec);
	
}








/*---- Utility functions ----*/


static thrust::host_vector<complex<double> > randomComplexes(int n) {
	std::uniform_real_distribution<double> valueDist(-1.0, 1.0);
	thrust::host_vector<complex<double> > result;
	for (int i = 0; i < n; i++)
		result.push_back(complex<double>(rand()%100, rand()%100));
	return result;
}