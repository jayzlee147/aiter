// GDN Wavefront H-scan instantiation
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_defs.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gdn_wf_h_kernel(gdn_wf_h_kargs kargs) {}
#else
#include "opus_gdn/gdn_wf_h_kernel_template.hpp"
#endif
// BT=64, K=V=128, BV=64, 4 warps
template __global__ void gdn_wf_h_kernel<gdn_k2_traits<64, 128, 128, 64, 4>>(gdn_wf_h_kargs);
// BT=64, K=V=128, BV=64, 8 warps
template __global__ void gdn_wf_h_kernel<gdn_k2_traits<64, 128, 128, 64, 8>>(gdn_wf_h_kargs);
// BT=128, K=V=128, BV=64, 4 warps
template __global__ void gdn_wf_h_kernel<gdn_k2_traits<128, 128, 128, 64, 4>>(gdn_wf_h_kargs);
