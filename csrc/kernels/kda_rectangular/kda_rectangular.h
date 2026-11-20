// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>
#include <optional>

namespace aiter::kda_rectangular {

// Keep PyTorch headers out of the HIP device translation unit.
enum NativeDType
{
    NATIVE_BF16,
    NATIVE_FLOAT,
    NATIVE_INT32,
    NATIVE_INT64,
    NATIVE_BOOL
};
struct TensorDesc
{
    void* data;
    int64_t sizes[4], strides[4];
    int ndim, dtype;
    int64_t size(int i) const { return sizes[i]; }
    int64_t stride(int i) const { return strides[i]; }
    int64_t numel() const
    {
        int64_t n = 1;
        for(int i = 0; i < ndim; i++)
            n *= sizes[i];
        return n;
    }
    void* data_ptr() const { return data; }
    template <class U>
    U* data_ptr() const
    {
        return static_cast<U*>(data);
    }
};
using NativeOpt = std::optional<TensorDesc>;
bool overlaps_bf16(const TensorDesc&, const TensorDesc&);
bool native_conv_half(
    TensorDesc, TensorDesc, NativeOpt, TensorDesc, NativeOpt, TensorDesc, int, hipStream_t);
bool native_conv(
    TensorDesc, TensorDesc, NativeOpt, TensorDesc, NativeOpt, TensorDesc, bool, bool, hipStream_t);
void native_prepare(TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    TensorDesc,
                    double,
                    double,
                    bool,
                    bool,
                    bool,
                    hipStream_t);
void native_walk(TensorDesc,
                 TensorDesc,
                 TensorDesc,
                 TensorDesc,
                 TensorDesc,
                 TensorDesc,
                 TensorDesc,
                 TensorDesc,
                 NativeOpt,
                 NativeOpt,
                 TensorDesc,
                 hipStream_t);
void native_norm(TensorDesc, TensorDesc, TensorDesc, TensorDesc, double, hipStream_t);
void native_norm_live(
    TensorDesc, TensorDesc, TensorDesc, TensorDesc, TensorDesc, double, hipStream_t);
void native_decode(TensorDesc,
                   TensorDesc,
                   TensorDesc,
                   TensorDesc,
                   TensorDesc,
                   TensorDesc,
                   TensorDesc,
                   TensorDesc,
                   TensorDesc,
                   TensorDesc,
                   TensorDesc,
                   TensorDesc,
                   NativeOpt,
                   NativeOpt,
                   TensorDesc,
                   double,
                   double,
                   hipStream_t);

void native_walk_part(TensorDesc,
                      TensorDesc,
                      TensorDesc,
                      TensorDesc,
                      TensorDesc,
                      TensorDesc,
                      TensorDesc,
                      TensorDesc,
                      NativeOpt,
                      NativeOpt,
                      TensorDesc,
                      int,
                      hipStream_t);

} // namespace aiter::kda_rectangular
