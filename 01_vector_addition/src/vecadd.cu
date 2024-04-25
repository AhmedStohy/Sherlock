// This program computes the sum of two vectors of length N
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void
vectorAdd(const int *__restrict a, const int *__restrict b, int *__restrict c, int N)
{
	// Calculate global thread ID
	int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
	// Boundary check
	if (thread_id < N)
		c[thread_id] = a[thread_id] + b[thread_id];
}

// Check vector add result
void
verify_result(const int *a, const int *b, const int *c, int N)
{
	for (int i = 0; i < N; i++)
		assert(a[i] + b[i] == c[i]);
}

int
main(int argc, char const *argv[])
{
	// Array size of 2^16 (65536 elements)
	constexpr int N = 1 << 16;
	constexpr size_t bytes = sizeof(int) * N;

	// Vectors for holding the host-side (CPU-side) data
	int a_h[N], b_h[N], c_h[N];
	// Initialize random numbers in each array
	for (int i = 0; i < N; i++)
	{
		a_h[i] = rand() % 100;
		b_h[i] = rand() % 100;
	}

	// Allocate memory on the device
	int *a_d, *b_d, *c_d;
	cudaMalloc((void **)&a_d, bytes);
	cudaMalloc((void **)&b_d, bytes);
	cudaMalloc((void **)&c_d, bytes);

	// Copy data from the host to the device (CPU -> GPU)
	cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice);

	// Threads per CTA (1024)
	int NUM_THREADS = 1 << 10;

	// CTAs per Grid
	// We need to launch at LEAST as many threads as we have elements
	// This equation pads an extra CTA to the grid if N cannot evenly be divided
	// by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	// Launch the kernel on the GPU
	// Kernel calls are asynchronous (the CPU program continues execution after
	// call, but no necessarily before the kernel finishes)
	vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(a_d, b_d, c_d, N);
	std::cout << "COMPLETED  ADDING SUCCESSFULLY !" << std::endl;

	// Copy sum vector from device to host
	// cudaMemcpy is a synchronous operation, and waits for the prior kernel
	// launch to complete (both go to the default stream in this case).
	// Therefore, this cudaMemcpy acts as both a memcpy and synchronization
	// barrier.
	cudaMemcpy(c_h, c_d, bytes, cudaMemcpyDeviceToHost);

	// Check result for errors
	verify_result(a_h, b_h, c_h, N);

	// Free memory on device
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	std::cout << "CODE COMPLETED SUCCESSFULLY !" << std::endl;

	return 0;
}