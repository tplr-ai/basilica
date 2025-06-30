extern "C" {

__global__ void matrix_multiply(
    const double* A,
    const double* B,
    double* C,
    int rowsA,
    int colsA,
    int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsB) {
        double sum = 0.0;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

__device__ unsigned int lcg_random(unsigned long long* seed) {
    // Linear congruential generator
    *seed = (*seed * 1664525ULL + 1013904223ULL) & 0xFFFFFFFFULL;
    return (unsigned int)(*seed);
}

__global__ void init_rng(unsigned long long* state, unsigned long long seed) {
    *state = seed;
}

__global__ void generate_random(double* data, unsigned long long* state, unsigned int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        unsigned long long local_seed = *state + idx;
        unsigned int rand_val = lcg_random(&local_seed);
        // Convert to double in range [0, 1)
        data[idx] = (double)rand_val / 4294967296.0;
    }
}

}