#pragma once

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

#include <cuda/atomic>
#include <cuda_fp16.h>
#include <cstdint>

namespace qe::sm75 {

__device__ __forceinline__ void mma_fp16(float& c0, float& c1, float& c2, float& c3,
                                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
    // The Ampere fragment is two consecutive Turing K=8 fragments.
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a2), "r"(a3), "r"(b1));
}

__device__ __forceinline__ void mma_fp16_packed(uint32_t& c01, uint32_t& c23,
                                               uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1) {
    const float2 top = __half22float2(__half2_raw{uint16_t(c01), uint16_t(c01 >> 16)});
    const float2 bottom = __half22float2(__half2_raw{uint16_t(c23), uint16_t(c23 >> 16)});
    float c0 = top.x, c1 = top.y, c2 = bottom.x, c3 = bottom.y;
    mma_fp16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    const __half2_raw packed_top = __floats2half2_rn(c0, c1);
    const __half2_raw packed_bottom = __floats2half2_rn(c2, c3);
    c01 = uint32_t(packed_top.x) | (uint32_t(packed_top.y) << 16);
    c23 = uint32_t(packed_bottom.x) | (uint32_t(packed_bottom.y) << 16);
}

__device__ __forceinline__ float bf16_dot_pair(float acc, uint32_t a, uint32_t b) {
    acc = fmaf(__uint_as_float(a << 16), __uint_as_float(b << 16), acc);
    return fmaf(__uint_as_float(a & 0xffff0000u), __uint_as_float(b & 0xffff0000u), acc);
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
    // Evaluate the same fragment with FP32 FMAs to preserve the BF16 exponent range.
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    const int a_base = lane & ~3;
    const int b_base = 8 * (lane & 3);
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t a_top = __shfl_sync(0xffffffff, k < 4 ? a0 : a2, a_base + (k & 3));
        const uint32_t a_bottom = __shfl_sync(0xffffffff, k < 4 ? a1 : a3, a_base + (k & 3));
        const uint32_t b = k < 4 ? b0 : b1;
        const uint32_t b_left = __shfl_sync(0xffffffff, b, b_base + (k & 3));
        const uint32_t b_right = __shfl_sync(0xffffffff, b, b_base + 4 + (k & 3));
        c0 = bf16_dot_pair(c0, a_top, b_left);
        c1 = bf16_dot_pair(c1, a_top, b_right);
        c2 = bf16_dot_pair(c2, a_bottom, b_left);
        c3 = bf16_dot_pair(c3, a_bottom, b_right);
    }
}

template<bool UnsignedB, typename Acc>
__device__ __forceinline__ void mma_int8_tile(Acc& c0, Acc& c1, uint32_t a, uint32_t b) {
    if constexpr (UnsignedB) {
        asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32 "
                     "{%0,%1}, {%2}, {%3}, {%0,%1};"
                     : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
    } else {
        asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                     "{%0,%1}, {%2}, {%3}, {%0,%1};"
                     : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
    }
}

template<bool UnsignedB, typename Acc>
__device__ __forceinline__ void mma_int8(Acc& c0, Acc& c1, Acc& c2, Acc& c3,
                                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
    mma_int8_tile<UnsignedB>(c0, c1, a0, b0);
    mma_int8_tile<UnsignedB>(c2, c3, a1, b0);
    mma_int8_tile<UnsignedB>(c0, c1, a2, b1);
    mma_int8_tile<UnsignedB>(c2, c3, a3, b1);
}

template<int Bytes, typename Address>
__device__ __forceinline__ void copy_to_shared(unsigned dest, Address source, unsigned valid = Bytes) {
    auto* dst = reinterpret_cast<uint32_t*>(__cvta_shared_to_generic(dest));
    const auto* src = reinterpret_cast<const uint32_t*>(source);
    #pragma unroll
    for (int offset = 0; offset < Bytes; offset += sizeof(uint32_t)) {
        dst[offset / sizeof(uint32_t)] =
            valid > unsigned(offset) ? src[offset / sizeof(uint32_t)] : 0;
    }
}

using Barrier = cuda::atomic_ref<uint32_t, cuda::thread_scope_block>;

// The callers reserve 8 bytes per barrier. Use the low word for the current
// phase, fixed arrival count and remaining arrivals in this phase.
__device__ __forceinline__ Barrier barrier_at(unsigned addr) {
    return Barrier(*reinterpret_cast<uint32_t*>(__cvta_shared_to_generic(addr)));
}

__device__ __forceinline__ void barrier_init(unsigned addr, unsigned count) {
    barrier_at(addr).store((count << 16) | count, cuda::memory_order_relaxed);
}

__device__ __forceinline__ void barrier_arrive(unsigned addr) {
    constexpr uint32_t phase = 0x80000000u;
    auto barrier = barrier_at(addr);
    uint32_t state = barrier.load(cuda::memory_order_relaxed);
    uint32_t next;
    do {
        next = state - 1;
        if ((state & 0xffffu) == 1) {
            next = ((state ^ phase) & 0xffff0000u) | ((state >> 16) & 0x7fffu);
        }
    } while (!barrier.compare_exchange_weak(state, next, cuda::memory_order_acq_rel,
                                             cuda::memory_order_relaxed));
}

__device__ __forceinline__ bool barrier_ready(unsigned addr, unsigned parity) {
    return (barrier_at(addr).load(cuda::memory_order_acquire) >> 31) != parity;
}

}  // namespace qe::sm75

#endif
