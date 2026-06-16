// Compatibility shim for __HIPCC_RTC__ mode on ROCm 7.2+
// __HIPCC_RTC__ suppresses implicit includes, hiding __syncthreads / int64_t.
// This header provides minimal definitions so kernel TUs compile.
#pragma once

#ifdef __HIPCC_RTC__
#ifdef __HIP_DEVICE_COMPILE__

#ifndef __syncthreads
__device__ inline void __syncthreads() { __builtin_amdgcn_s_barrier(); }
#endif

#ifndef _INT64_T
#define _INT64_T
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif

#endif // __HIP_DEVICE_COMPILE__
#endif // __HIPCC_RTC__
