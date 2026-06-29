// GDN K2-OUT instantiation (split-path parallel-over-chunks output)
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_defs.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gdn_k2_out_kernel(gdn_k2_kargs kargs) {}
#else
#include "opus_gdn/gdn_k2_out_kernel_template.hpp"
#endif
template __global__ void gdn_k2_out_kernel<gdn_k2_traits<64, 128, 128, 32, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_out_kernel<gdn_k2_traits<64, 128, 128, 32, 8>>(gdn_k2_kargs);
// larger V-tiles cut redundant q/k re-reads + intra q@k^T recompute (out kernel is HBM-BW bound)
template __global__ void gdn_k2_out_kernel<gdn_k2_traits<64, 128, 128, 64, 8>>(gdn_k2_kargs);
template __global__ void gdn_k2_out_kernel<gdn_k2_traits<64, 128, 128, 64, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_out_kernel<gdn_k2_traits<64, 128, 128, 128, 8>>(gdn_k2_kargs);
template __global__ void gdn_k2_out_kernel<gdn_k2_traits<64, 128, 128, 128, 4>>(gdn_k2_kargs);
