#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "sm75_compat.cuh"

namespace qe::cuda_compat {

__device__ __forceinline__ void mma_fp16(float& c0, float& c1, float& c2, float& c3,
                                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    sm75::mma_fp16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_fp16_packed(uint32_t& c01, uint32_t& c23,
                                               uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    sm75::mma_fp16_packed(c01, c23, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(c01), "+r"(c23)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    sm75::mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

template<bool UnsignedB, typename Acc>
__device__ __forceinline__ void mma_int8(Acc& c0, Acc& c1, Acc& c2, Acc& c3,
                                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    sm75::mma_int8<UnsignedB>(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
    if constexpr (UnsignedB) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
#endif
}

template<int Bytes, typename Address>
__device__ __forceinline__ void copy_to_shared(unsigned dest, Address source, unsigned valid = Bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    sm75::copy_to_shared<Bytes>(dest, source, valid);
#else
    if constexpr (Bytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                     :: "r"(dest), "l"(source), "r"(valid) : "memory");
    } else {
        static_assert(Bytes == 4 || Bytes == 8);
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;"
                     :: "r"(dest), "l"(source), "n"(Bytes), "r"(valid) : "memory");
    }
#endif
}

__device__ __forceinline__ void copy_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("" ::: "memory");
#else
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template<int Groups>
__device__ __forceinline__ void copy_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("" ::: "memory");
#else
    asm volatile("cp.async.wait_group %0;" :: "n"(Groups) : "memory");
#endif
}

__device__ __forceinline__ void copy_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("" ::: "memory");
#else
    asm volatile("cp.async.wait_all;" ::: "memory");
#endif
}

template<bool Max, typename T>
__device__ __forceinline__ T warp_extreme(T value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    #pragma unroll
    for (int bit = 16; bit > 0; bit /= 2) {
        const T peer = __shfl_xor_sync(0xffffffffu, value, bit);
        value = Max ? max(value, peer) : min(value, peer);
    }
    return value;
#else
    if constexpr (Max) return __reduce_max_sync(0xffffffffu, value);
    else return __reduce_min_sync(0xffffffffu, value);
#endif
}

template<typename Address>
__device__ __forceinline__ void prefetch_l2(Address address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("prefetch.global.L2 [%0];" :: "l"(address));
#else
    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(address));
#endif
}

__device__ __forceinline__ void barrier_init(unsigned addr, unsigned count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    sm75::barrier_init(addr, count);
#else
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
#endif
}

__device__ __forceinline__ void barrier_arrive(unsigned addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    sm75::barrier_arrive(addr);
#else
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(addr) : "memory");
#endif
}

__device__ __forceinline__ void barrier_arrive_after_copy(unsigned addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    sm75::barrier_arrive(addr);
#else
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(addr) : "memory");
#endif
}

__device__ __forceinline__ bool barrier_ready(unsigned addr, unsigned parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return sm75::barrier_ready(addr, parity);
#else
    unsigned ready;
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; "
                 "selp.u32 %0, 1, 0, p; }"
                 : "=r"(ready) : "r"(addr), "r"(parity) : "memory");
    return ready != 0;
#endif
}

}  // namespace qe::cuda_compat
