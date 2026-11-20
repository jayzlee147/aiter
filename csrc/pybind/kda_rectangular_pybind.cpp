// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "kda_rectangular.h"
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <torch/csrc/utils/pybind.h>

namespace {
using namespace aiter::kda_rectangular;
using T   = at::Tensor;
using Opt = c10::optional<T>;
TensorDesc d(T t)
{
    TORCH_CHECK(t.is_cuda(), "HIP KDA expects GPU tensors");
    TORCH_CHECK(t.dim() <= 4, "HIP KDA supports at most four dimensions");
    TensorDesc r{};
    r.data = t.data_ptr();
    r.ndim = t.dim();
    switch(t.scalar_type())
    {
    case at::kBFloat16: r.dtype = NATIVE_BF16; break;
    case at::kFloat: r.dtype = NATIVE_FLOAT; break;
    case at::kInt: r.dtype = NATIVE_INT32; break;
    case at::kLong: r.dtype = NATIVE_INT64; break;
    case at::kBool: r.dtype = NATIVE_BOOL; break;
    default: TORCH_CHECK(false, "Unsupported HIP KDA dtype");
    }
    for(int i = 0; i < t.dim(); i++)
    {
        r.sizes[i]   = t.size(i);
        r.strides[i] = t.stride(i);
    }
    return r;
}
NativeOpt od(Opt t) { return t ? NativeOpt(d(*t)) : std::nullopt; }
hipStream_t stream() { return c10::hip::getCurrentHIPStream().stream(); }
bool dispatch_conv(T x, T weight, Opt cache, T cu, Opt has, T out, bool for_mixer)
{
    c10::DeviceGuard guard(x.device());
    auto input = d(x), coeffs = d(weight);
    auto history = od(cache);
    if(history && overlaps_bf16(*history, input))
    {
        // Cache writes can overwrite input needed by other channels or sequences.
        // Preserve inactive histories and commit only after every input read.
        at::NoGradGuard no_grad;
        T staged = cache->clone();
        native_conv(input, coeffs, d(staged), d(cu), od(has), d(out), false, false, stream());
        cache->copy_(staged);
        return false;
    }
    // When only the weights alias the cache, the separate commit kernel runs
    // after all weight reads. The usual tiled path needs no extra cache copy.
    const bool allow_tiled = !history || !overlaps_bf16(*history, coeffs);
    return native_conv(
        input, coeffs, history, d(cu), od(has), d(out), for_mixer, allow_tiled, stream());
}
void conv(T x, T weight, Opt cache, T cu, Opt has, T out)
{
    dispatch_conv(x, weight, cache, cu, has, out, false);
}
bool conv_half(T x, T weight, Opt cache, T cu, Opt has, T out, int half)
{
    c10::DeviceGuard guard(x.device());
    TORCH_CHECK(half == 0 || half == 1, "Expected first or second convolution half");
    return native_conv_half(d(x), d(weight), od(cache), d(cu), od(has), d(out), half, stream());
}
bool conv_for_mixer(T x, T weight, Opt cache, T cu, Opt has, T out)
{
    return dispatch_conv(x, weight, cache, cu, has, out, true);
}
template <bool ForWalk, bool AllowStreaming = true, bool Blocked = false>
void prepare(T q,
             T k,
             T v,
             T g,
             T beta,
             T a,
             T dt,
             T cu,
             T ci,
             T qg,
             T w,
             T u,
             T kg,
             T aq,
             T decay,
             double lb,
             double scale)
{
    c10::DeviceGuard guard(q.device());
    native_prepare(d(q),
                   d(k),
                   d(v),
                   d(g),
                   d(beta),
                   d(a),
                   d(dt),
                   d(cu),
                   d(ci),
                   d(qg),
                   d(w),
                   d(u),
                   d(kg),
                   d(aq),
                   d(decay),
                   lb,
                   scale,
                   ForWalk,
                   AllowStreaming,
                   Blocked,
                   stream());
}
void walk(T qg, T w, T u, T kg, T aq, T decay, T cu, T offsets, Opt initial, Opt final, T out)
{
    c10::DeviceGuard guard(out.device());
    native_walk(d(qg),
                d(w),
                d(u),
                d(kg),
                d(aq),
                d(decay),
                d(cu),
                d(offsets),
                od(initial),
                od(final),
                d(out),
                stream());
}
void walk_part(
    T qg, T w, T u, T kg, T aq, T decay, T cu, T offsets, Opt initial, Opt final, T out, int begin)
{
    c10::DeviceGuard guard(out.device());
    native_walk_part(d(qg),
                     d(w),
                     d(u),
                     d(kg),
                     d(aq),
                     d(decay),
                     d(cu),
                     d(offsets),
                     od(initial),
                     od(final),
                     d(out),
                     begin,
                     stream());
}
void norm(T x, T gate, T weight, T out, double eps)
{
    c10::DeviceGuard guard(x.device());
    native_norm(d(x), d(gate), d(weight), d(out), eps, stream());
}
void norm_live(T x, T gate, T weight, T out, T cu, double eps)
{
    c10::DeviceGuard guard(x.device());
    native_norm_live(d(x), d(gate), d(weight), d(out), d(cu), eps, stream());
}
void decode(T x,
            T conv,
            T weight,
            T gate,
            T beta,
            T og,
            T a,
            T dt,
            T state,
            T indices,
            T cu,
            T nw,
            Opt accepted,
            Opt ci,
            T out,
            double lb,
            double eps)
{
    c10::DeviceGuard guard(x.device());
    native_decode(d(x),
                  d(conv),
                  d(weight),
                  d(gate),
                  d(beta),
                  d(og),
                  d(a),
                  d(dt),
                  d(state),
                  d(indices),
                  d(cu),
                  d(nw),
                  od(accepted),
                  od(ci),
                  d(out),
                  lb,
                  eps,
                  stream());
}
} // namespace

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    m.def("conv", &conv);
    m.def("conv_half", &conv_half);
    m.def("conv_for_mixer", &conv_for_mixer);
    m.def("prepare", &prepare<false>);
    m.def("prepare_for_walk", &prepare<true>);
    m.def("prepare_for_blocked", &prepare<true, true, true>);
    m.def("prepare_for_core", &prepare<true, false>);
    m.def("walk", &walk);
    m.def("walk_part", &walk_part);
    m.def("norm", &norm);
    m.def("norm_live", &norm_live);
    m.def("decode", &decode);
}
