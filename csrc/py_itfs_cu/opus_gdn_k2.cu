// GDN K2 instantiation (h update + output)
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_defs.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gdn_k2_kernel(gdn_k2_kargs kargs) {}
#else
#include "opus_gdn/gdn_k2_kernel_template.hpp"
#endif
template __global__ void gdn_k2_kernel<gdn_k2_traits<64, 128, 128, 64, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<64, 128, 128, 32, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<64, 128, 128, 64, 8>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<64, 128, 128, 32, 8>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<32, 128, 128, 64, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_kernel<gdn_k2_traits<16, 128, 128, 64, 4>>(gdn_k2_kargs);
