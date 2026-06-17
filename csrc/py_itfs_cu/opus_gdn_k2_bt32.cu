// GDN K2 BT=32 instantiation (h update + output)
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_defs.h"
#ifndef __HIP_DEVICE_COMPILE__
template<typename Traits> __global__ void gdn_k2_bt32_kernel(gdn_k2_kargs kargs) {}
#else
#include "opus_gdn/gdn_k2_bt32_kernel_template.hpp"
#endif
template __global__ void gdn_k2_bt32_kernel<gdn_k2_traits<32, 128, 128, 64, 4>>(gdn_k2_kargs);
