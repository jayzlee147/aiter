// GDN K2 split-mode instantiation: scan-only + parallel output
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_defs.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gdn_k2_scan_kernel(gdn_k2_kargs kargs) {}
template<typename Traits> __global__ void gdn_k2_output_kernel(gdn_k2_output_kargs kargs) {}
#else
#include "opus_gdn/gdn_k2_scan_kernel_template.hpp"
#include "opus_gdn/gdn_k2_output_kernel_template.hpp"
#endif
// BT=64, K=V=128, BV=64, nw=8 — target configuration
template __global__ void gdn_k2_scan_kernel<gdn_k2_traits<64, 128, 128, 64, 8>>(gdn_k2_kargs);
template __global__ void gdn_k2_output_kernel<gdn_k2_traits<64, 128, 128, 64, 8>>(gdn_k2_output_kargs);
// BT=64, K=V=128, BV=64, nw=4 — fallback
template __global__ void gdn_k2_scan_kernel<gdn_k2_traits<64, 128, 128, 64, 4>>(gdn_k2_kargs);
template __global__ void gdn_k2_output_kernel<gdn_k2_traits<64, 128, 128, 64, 4>>(gdn_k2_output_kargs);
