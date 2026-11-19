#include "kernels/cuda_compat.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

namespace cc = qe::cuda_compat;

#define CHECK_CUDA(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        return EXIT_FAILURE; \
    } \
} while (0)

__host__ __device__ int left(int row, int k) { return (row * 7 + k * 3 + k / 5) % 11 - 5; }
__host__ __device__ int right(int k, int col) { return (k * 5 + col * 3 + k / 3) % 9 - 4; }

enum class Format { FP16, FP16Packed, BF16, INT8, MixedINT8 };

template<Format F>
constexpr bool is_int = F == Format::INT8 || F == Format::MixedINT8;

template<Format F, bool A>
__device__ uint32_t pack(int row, int col) {
    uint32_t result = 0;
    constexpr int width = is_int<F> ? 8 : 16;
    for (int i = 0; i < 32 / width; ++i) {
        const int v = A ? left(row, col + i) : right(row + i, col);
        uint32_t bits;
        if constexpr (is_int<F>) {
            bits = uint8_t(F == Format::MixedINT8 && !A ? v + 193 : v);
        } else if constexpr (F == Format::BF16) {
            const float x = v * (A ? 0x1p24f : 0x1p-24f);
            bits = __bfloat16_as_ushort(__float2bfloat16_rn(x));
        } else {
            bits = __half_as_ushort(__int2half_rn(v));
        }
        result |= bits << (width * i);
    }
    return result;
}

template<Format F>
__global__ void matrix(float* out) {
    const int group = threadIdx.x / 4;
    const int thread = threadIdx.x % 4;
    constexpr int pack_size = is_int<F> ? 4 : 2;
    const int k0 = thread * pack_size;
    const int k1 = k0 + 4 * pack_size;
    const uint32_t a0 = pack<F, true>(group, k0);
    const uint32_t a1 = pack<F, true>(group + 8, k0);
    const uint32_t a2 = pack<F, true>(group, k1);
    const uint32_t a3 = pack<F, true>(group + 8, k1);
    const uint32_t b0 = pack<F, false>(k0, group);
    const uint32_t b1 = pack<F, false>(k1, group);
    float c0 = 1, c1 = 2, c2 = 3, c3 = 4;
    if constexpr (is_int<F>) {
        int i0 = 1, i1 = 2, i2 = 3, i3 = 4;
        cc::mma_int8<F == Format::MixedINT8>(i0, i1, i2, i3, a0, a1, a2, a3, b0, b1);
        c0 = i0; c1 = i1; c2 = i2; c3 = i3;
    } else if constexpr (F == Format::BF16) {
        cc::mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    } else if constexpr (F == Format::FP16Packed) {
        uint32_t top = 0x40003c00u, bottom = 0x44004200u;  // (1, 2) and (3, 4)
        cc::mma_fp16_packed(top, bottom, a0, a1, a2, a3, b0, b1);
        c0 = __half2float(__ushort_as_half(uint16_t(top)));
        c1 = __half2float(__ushort_as_half(uint16_t(top >> 16)));
        c2 = __half2float(__ushort_as_half(uint16_t(bottom)));
        c3 = __half2float(__ushort_as_half(uint16_t(bottom >> 16)));
    } else {
        cc::mma_fp16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }
    out[group * 8 + thread * 2] = c0;
    out[group * 8 + thread * 2 + 1] = c1;
    out[(group + 8) * 8 + thread * 2] = c2;
    out[(group + 8) * 8 + thread * 2 + 1] = c3;
}

template<Format F>
int check_matrix(float* out) {
    matrix<F><<<1, 32>>>(out);
    float values[128];
    CHECK_CUDA(cudaMemcpy(values, out, sizeof(values), cudaMemcpyDeviceToHost));
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 8; ++col) {
            int expected = 1 + col % 2 + (row >= 8 ? 2 : 0);
            for (int k = 0; k < (is_int<F> ? 32 : 16); ++k) {
                expected += left(row, k) * (right(k, col) + (F == Format::MixedINT8 ? 193 : 0));
            }
            if (values[row * 8 + col] != float(expected)) {
                std::fprintf(stderr, "format %d row %d col %d: expected %d, got %g\n",
                             int(F), row, col, expected, values[row * 8 + col]);
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

__global__ void copy_and_arrive(const uint4* input, int* errors) {
    __shared__ uint4 rows[64];
    __shared__ alignas(8) uint32_t state[2];
    const unsigned addr = __cvta_generic_to_shared(state);
    const int thread = threadIdx.x;
    if (thread == 0) cc::barrier_init(addr, 64);
    __syncthreads();
    for (unsigned phase = 0; phase < 4; ++phase) {
        const int src = (thread + phase) % 64;
        cc::copy_to_shared<16>(__cvta_generic_to_shared(rows + thread), input + src, thread % 3 ? 16 : 0);
        cc::barrier_arrive_after_copy(addr);
        while (!cc::barrier_ready(addr, phase & 1u)) __nanosleep(32);
        const int peer = (thread + 32) % 64;
        const uint4 value = rows[peer];
        const unsigned x = 4u * ((peer + phase) % 64);
        if (peer % 3 ? value.x != x || value.y != x + 1 || value.z != x + 2 || value.w != x + 3
                     : value.x || value.y || value.z || value.w) {
            atomicAdd(errors, 1);
        }
        __syncthreads();
    }
    cc::barrier_arrive(addr);
    while (!cc::barrier_ready(addr, 0)) __nanosleep(32);
}

int main() {
    int count = 0;
    const cudaError_t devices = cudaGetDeviceCount(&count);
    if (devices == cudaErrorNoDevice || devices == cudaErrorInsufficientDriver || (devices == cudaSuccess && !count)) {
        std::puts("CUDA device unavailable");
        return 77;
    }
    CHECK_CUDA(devices);
    float* out;
    CHECK_CUDA(cudaMalloc(&out, 128 * sizeof(float)));
    const int failed = check_matrix<Format::FP16>(out) || check_matrix<Format::FP16Packed>(out) ||
                       check_matrix<Format::BF16>(out) || check_matrix<Format::INT8>(out) ||
                       check_matrix<Format::MixedINT8>(out);
    CHECK_CUDA(cudaFree(out));
    if (failed) return EXIT_FAILURE;

    uint4 input[64];
    for (unsigned i = 0; i < 64; ++i) input[i] = make_uint4(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3);
    uint4* gpu_input;
    int* errors;
    CHECK_CUDA(cudaMalloc(&gpu_input, sizeof(input)));
    CHECK_CUDA(cudaMalloc(&errors, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(gpu_input, input, sizeof(input), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(errors, 0, sizeof(int)));
    copy_and_arrive<<<1, 64>>>(gpu_input, errors);
    int mismatches;
    CHECK_CUDA(cudaMemcpy(&mismatches, errors, sizeof(mismatches), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(gpu_input));
    CHECK_CUDA(cudaFree(errors));
    if (mismatches) std::fprintf(stderr, "shared copy: %d mismatches\n", mismatches);
    return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
