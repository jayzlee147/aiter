// GDN Prefill K1 Kernel — BT=16, Neumann Series
// Step 1: g_cumsum + KKT Gram matrix
// Step 2: Triangular inverse (I+A)^{-1} via Neumann series C = Σ (-A)^n
//         + WY factor assembly (w_bar, u_bar)
//
// Grid: (NT, B*H)   Block: (BLOCK_SIZE = 256)
// Target: gfx942 (MI300X), MFMA bf16 16×16×16
//
// A is 16×16 strictly lower triangular → nilpotent of order 16.
// Neumann series: (I+A)^{-1} = I - A + A² - A³ + … + (-A)^{15}
// Each (-A)^n = (-A)^{n-1} × (-A), computable via MFMA 16×16×16.
// In practice A^n = 0 for n ≥ 16, but we truncate at the first zero power.
#pragma once

#include <opus/opus.hpp>
#include "opus_gdn/gdn_defs.h"

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, 2)
gdn_k1_kernel(gdn_k1_kargs kargs) {
    using namespace opus;
    using T = Traits;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    static_assert(T::BT == 16, "This template is for BT=16 only");

    const int i_t  = blockIdx.x;   // chunk index
    const int i_bh = blockIdx.y;   // batch*H index
    const int i_b  = i_bh / kargs.H;
    const int i_h  = i_bh % kargs.H;

    const int tid     = threadIdx.x;
    const int warp_id = tid / T::WARP_SIZE;
    const int lane_id = tid % T::WARP_SIZE;

    const int BT = T::BT;  // 16
    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;

    const int chunk_start = i_t * BT;
    const int bos = i_b * kargs.T;

    // =====================================================================
    // Shared memory
    // =====================================================================
    extern __shared__ char smem_buf[];

    // Phase 1: g[BT] + beta[BT] + k[BT×K]
    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);        // 16 fp32
    D_ACC*  s_beta = s_g + BT;                                  // 16 fp32
    D_ATTN* s_k    = reinterpret_cast<D_ATTN*>(s_beta + BT);   // 16×K bf16

    // Phase 2: C[BT×BT] fp32 = 1KB (where s_k was — s_k is only 16*128*2=4KB)
    D_ACC*  s_C = reinterpret_cast<D_ACC*>(s_k);  // 16×16 fp32

    // Neumann temp: neg_A_pow[BT×BT] fp32 = 1KB (beyond s_C)
    D_ACC*  s_neg_A_pow = s_C + BT * BT;          // 16×16 fp32

    // =====================================================================
    // Phase 1a: Load g and beta, compute prefix sum
    // =====================================================================
    const D_ACC* g_base    = reinterpret_cast<const D_ACC*>(kargs.ptr_g)
                             + (bos + chunk_start) * H + i_h;
    const D_ACC* beta_base = reinterpret_cast<const D_ACC*>(kargs.ptr_beta)
                             + (bos + chunk_start) * H + i_h;

    for (int i = tid; i < BT; i += T::BLOCK_SIZE) {
        int global_t = chunk_start + i;
        if (global_t < kargs.T) {
            s_g[i]    = g_base[i * H];
            s_beta[i] = beta_base[i * H];
        } else {
            s_g[i]    = 0.0f;
            s_beta[i] = 0.0f;
        }
    }
    __syncthreads();

    // Hillis-Steele prefix sum (4 steps for BT=16)
    for (int stride = 1; stride < BT; stride <<= 1) {
        for (int i = tid; i < BT; i += T::BLOCK_SIZE) {
            if (i >= stride)
                s_g[i] += s_g[i - stride];
        }
        __syncthreads();
    }

    // Write g_cumsum to HBM
    D_ACC* g_cumsum_base = reinterpret_cast<D_ACC*>(kargs.ptr_g_cumsum)
                           + (bos + chunk_start) * H + i_h;
    for (int i = tid; i < BT; i += T::BLOCK_SIZE) {
        if (chunk_start + i < kargs.T)
            g_cumsum_base[i * H] = s_g[i];
    }
    __syncthreads();

    // =====================================================================
    // Phase 1b: Load k[BT, K] into LDS
    // =====================================================================
    const D_ATTN* k_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                           + ((bos + chunk_start) * H + i_h) * K;

    // BT*K = 16*128 = 2048 elements, one pass with 256 threads
    for (int i = tid; i < BT * K; i += T::BLOCK_SIZE) {
        int row = i / K;
        int col = i % K;
        if (chunk_start + row < kargs.T)
            s_k[i] = k_base[row * H * K + col];
        else
            s_k[i] = static_cast<D_ATTN>(0);
    }
    __syncthreads();

    // =====================================================================
    // Phase 1c: KKT GEMM — A[BT, BT] = k[BT, K] × k^T[K, BT]
    //
    // For BT=16, this is a single 16×16 output tile.
    // K=128 → inner dimension = K/16 = 8 MFMA steps.
    // =====================================================================
    auto mma_kkt = make_tiled_mma<bf16_t, bf16_t, fp32_t>(
        seq<T::KKT_E_M, T::KKT_E_N, T::KKT_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    auto p_coord = make_tuple(number<lane_id>{}, number<warp_id>{});
    auto u_ka = mma_kkt.layout_a(make_tuple(number<K>{}, number<1>{}), p_coord);
    auto u_kb = mma_kkt.layout_b(make_tuple(number<K>{}, number<1>{}), p_coord);

    auto s_k_handle = make_smem(reinterpret_cast<D_ATTN*>(s_k));
    auto v_ka = s_k_handle.load<T::VEC_KV>(u_ka);
    auto v_kb = s_k_handle.load<T::VEC_KV>(u_kb);

    auto v_kkt = mma_kkt.template init_c<D_ACC>();
    v_kkt = mma_kkt(v_ka, v_kb, v_kkt);

    __syncthreads();

    // =====================================================================
    // Phase 1d: Gate scaling + lower-triangular mask → A (strict lower tri)
    // Store scaled A into s_C[BT×BT] fp32 (reusing s_k region)
    // =====================================================================
    auto u_c = mma_kkt.layout_c(make_tuple(number<BT>{}, number<1>{}), p_coord);
    auto y_shape_c = mma_kkt.y_shape_c();
    constexpr int c_elems = decltype(y_shape_c)::size();

    static_for<c_elems>([&](auto i) {
        int offset = u_c(number<i>{});
        int row = offset / BT;
        int col = offset % BT;
        D_ACC val = v_kkt[i];
        if (row > col && row < BT && col < BT) {
            D_ACC g_diff = s_g[row] - s_g[col];
            val = s_beta[row] * __expf(g_diff) * val;
        } else {
            val = 0.0f;
        }
        s_C[row * BT + col] = val;
    });
    __syncthreads();

    // =====================================================================
    // Phase 2a: Neumann series — C = (I + A)^{-1} = Σ_{n=0}^{15} (-A)^n
    //
    // A is 16×16 strictly lower triangular → nilpotent, series is exact.
    // Algorithm:
    //   neg_A_pow_0 = I       (n=0 term)
    //   C = I                 (accumulator, starts with n=0)
    //   for n = 1 .. 15:
    //     neg_A_pow_n = neg_A_pow_{n-1} @ (-A)    (16×16 matmul)
    //     C += neg_A_pow_n
    //     if neg_A_pow_n == 0: break   (early exit)
    //
    // For 16×16 strict lower tri, powers go zero by n=16 at latest.
    // Effective terms: typically 5-8 are nonzero.
    // =====================================================================

    // Negate A in place: s_C now holds -A (strict lower tri part) + zeros elsewhere
    for (int i = tid; i < BT * BT; i += T::BLOCK_SIZE) {
        int r = i / BT;
        int c = i % BT;
        if (r > c)
            s_C[i] = -s_C[i];
        // diagonal and upper are already 0
    }
    __syncthreads();

    // s_C = -A.  We need: C_result = I + (-A) + (-A)^2 + (-A)^3 + ...
    // Initialize neg_A_pow = I, C_result in separate buffer

    // We'll use s_C for -A (read-only), s_neg_A_pow for current power,
    // and accumulate C_result into s_C itself after we're done reading -A.
    // Problem: we need -A throughout for matmuls but also accumulate into C.
    //
    // Solution: keep -A in s_C, use s_neg_A_pow for the running power,
    // accumulate C in registers (each thread owns BT*BT/BS = 1 element).

    // BT*BT = 256 = BS, so each thread owns exactly one element of C
    int my_r = tid / BT;
    int my_c = tid % BT;

    D_ACC c_accum = (my_r == my_c) ? 1.0f : 0.0f;  // n=0 term: I

    // Initialize neg_A_pow = I
    s_neg_A_pow[tid] = (my_r == my_c) ? 1.0f : 0.0f;
    __syncthreads();

    // Iterate: neg_A_pow_n = neg_A_pow_{n-1} @ (-A), C += neg_A_pow_n
    for (int n = 1; n < BT; n++) {
        // new_pow[r, c] = Σ_j old_pow[r, j] * (-A)[j, c]
        // Since (-A) is strict lower tri, (-A)[j, c] != 0 only for j > c
        D_ACC new_val = 0.0f;
        for (int j = 0; j < BT; j++) {
            new_val += s_neg_A_pow[my_r * BT + j] * s_C[j * BT + my_c];
        }
        __syncthreads();
        s_neg_A_pow[tid] = new_val;
        __syncthreads();

        c_accum += new_val;
    }

    // Write C_result to s_C (overwriting -A, which is no longer needed)
    s_C[tid] = c_accum;
    __syncthreads();

    // =====================================================================
    // Phase 2c: WY factors
    //   u_bar = C @ (v * beta)
    //   w_bar = C @ (k * beta * exp(g_cumsum))
    //
    // C is 16×16 in s_C. Process K/V in 64-wide subtiles.
    // =====================================================================
    D_ATTN* w_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_w_bar)
                         + ((bos + chunk_start) * H + i_h) * K;
    D_ATTN* u_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_u_bar)
                         + ((bos + chunk_start) * H + i_h) * V;

    // Temp buffer for scaled k/v subtiles: BT*BK_SUB bf16 = 16*64*2 = 2KB
    // Place after s_neg_A_pow (which is 1KB beyond s_C)
    D_ATTN* s_kv_sub = reinterpret_cast<D_ATTN*>(s_neg_A_pow + BT * BT);

    // --- u_bar = C @ (v * beta) ---
    const D_ATTN* v_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_v)
                           + ((bos + chunk_start) * H + i_h) * V;

    for (int iv = 0; iv < T::N_V_ITERS; iv++) {
        int v_offset = iv * T::BV_SUB;

        // Load v subtile [BT, BV_SUB=64] scaled by beta
        for (int i = tid; i < BT * T::BV_SUB; i += T::BLOCK_SIZE) {
            int row = i / T::BV_SUB;
            int col = i % T::BV_SUB;
            if (chunk_start + row < kargs.T) {
                D_ACC val = static_cast<D_ACC>(v_base[row * H * V + v_offset + col]);
                s_kv_sub[i] = static_cast<D_ATTN>(val * s_beta[row]);
            } else {
                s_kv_sub[i] = static_cast<D_ATTN>(0);
            }
        }
        __syncthreads();

        // u_bar[BT, 64] = C[BT, BT] @ v_scaled[BT, 64]
        for (int idx = tid; idx < BT * T::BV_SUB; idx += T::BLOCK_SIZE) {
            int row = idx / T::BV_SUB;
            int col = idx % T::BV_SUB;
            D_ACC acc = 0.0f;
            for (int j = 0; j < BT; j++)
                acc += s_C[row * BT + j]
                     * static_cast<D_ACC>(s_kv_sub[j * T::BV_SUB + col]);
            if (chunk_start + row < kargs.T)
                u_bar_base[row * H * V + v_offset + col]
                    = static_cast<D_ATTN>(acc);
        }
        __syncthreads();
    }

    // --- w_bar = C @ (k * beta * exp(g_cumsum)) ---
    for (int ik = 0; ik < T::N_K_ITERS; ik++) {
        int k_offset = ik * T::BK_SUB;

        // Load k subtile [BT, BK_SUB=64] scaled by beta * exp(g_cumsum)
        for (int i = tid; i < BT * T::BK_SUB; i += T::BLOCK_SIZE) {
            int row = i / T::BK_SUB;
            int col = i % T::BK_SUB;
            if (chunk_start + row < kargs.T) {
                D_ACC val = static_cast<D_ACC>(k_base[row * H * K + k_offset + col]);
                s_kv_sub[i] = static_cast<D_ATTN>(
                    val * s_beta[row] * __expf(s_g[row]));
            } else {
                s_kv_sub[i] = static_cast<D_ATTN>(0);
            }
        }
        __syncthreads();

        // w_bar[BT, 64] = C[BT, BT] @ k_scaled[BT, 64]
        for (int idx = tid; idx < BT * T::BK_SUB; idx += T::BLOCK_SIZE) {
            int row = idx / T::BK_SUB;
            int col = idx % T::BK_SUB;
            D_ACC acc = 0.0f;
            for (int j = 0; j < BT; j++)
                acc += s_C[row * BT + j]
                     * static_cast<D_ACC>(s_kv_sub[j * T::BK_SUB + col]);
            if (chunk_start + row < kargs.T)
                w_bar_base[row * H * K + k_offset + col]
                    = static_cast<D_ATTN>(acc);
        }
        __syncthreads();
    }
}
