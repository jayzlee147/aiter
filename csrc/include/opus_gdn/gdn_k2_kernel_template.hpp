// GDN Prefill K2 Kernel — Fused Step 3 + Step 4
// Step 3: Hidden state update (retrieve, gate-scale, decay, accumulate)
// Step 4: Output (cross-chunk QH, intra-chunk causal attention)
//
// Grid: (cdiv(V, BV), N*H)   Block: (BLOCK_SIZE = 256)
// h state: register-resident, N_K × [BK_SUB, BV] fp32
// Chunks: serial iteration within each workgroup
//
// GEMMs: scalar first-pass (MFMA optimization deferred)
// h is spilled to LDS for retrieve/cross-chunk GEMMs, remains in registers
// for decay/accumulate. v_new kept in LDS untouched for AV matmul.
//
// Target: gfx942 (MI300X)
#pragma once

#include <opus/opus.hpp>
#include "opus_gdn/gdn_defs.h"

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, 2)
gdn_k2_kernel(gdn_k2_kargs kargs) {
    using T = Traits;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    static_assert(T::BK_SUB == T::BV,
        "First-pass requires BK_SUB == BV for LDS aliasing");

    constexpr int BT     = T::BT;
    constexpr int BK_SUB = T::BK_SUB;     // 64
    constexpr int BV     = T::BV;          // 64
    constexpr int N_K    = T::N_K;         // 2 for K=128
    constexpr int BS     = T::BLOCK_SIZE;  // 256

    // Per-thread element counts
    // h[BK_SUB, BV] distributed across BS threads
    constexpr int H_ELEMS = BK_SUB * BV / BS;  // 16
    // Output tile [BT, BV] distributed across BS threads
    constexpr int O_ELEMS = BT * BV / BS;      // 16
    // A_intra tile [BT, BT] distributed across BS threads
    constexpr int A_ELEMS = BT * BT / BS;      // 16

    // Thread identity
    const int i_v  = blockIdx.x;
    const int i_nh = blockIdx.y;
    const int i_n  = i_nh / kargs.H;
    const int i_h  = i_nh % kargs.H;
    const int tid  = threadIdx.x;

    const int v_off = i_v * BV;
    const int bos   = i_n * kargs.T;

    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;
    const int NT = kargs.NT;

    // [B, T, H, K/V] strides
    const int stride_k = H * K;
    const int stride_v = H * V;
    const int stride_g = H;

    // =====================================================================
    // Register-resident h state
    //
    // Thread distribution: element i maps to flat = tid + i * BS
    //   row = flat / BV,  col = flat % BV
    // =====================================================================
    D_ACC h1[H_ELEMS];
    D_ACC h2[H_ELEMS];
    for (int i = 0; i < H_ELEMS; i++) {
        h1[i] = 0.0f;
        if constexpr (N_K >= 2) h2[i] = 0.0f;
    }

    if (kargs.ptr_h0 != nullptr) {
        const D_ACC* h0 = reinterpret_cast<const D_ACC*>(kargs.ptr_h0)
                          + (i_n * H + i_h) * K * V;
        for (int i = 0; i < H_ELEMS; i++) {
            int flat = tid + i * BS;
            int r = flat / BV;
            int c = flat % BV;
            if (v_off + c < V) {
                h1[i] = h0[r * V + v_off + c];
                if constexpr (N_K >= 2)
                    h2[i] = h0[(BK_SUB + r) * V + v_off + c];
            }
        }
    }

    // =====================================================================
    // Shared memory layout
    //
    // Persistent:
    //   s_g[BT]              fp32       256B
    //
    // Pool (reinterpreted per phase):
    //   Phase B/C: s_h[BK_SUB*BV] fp32 (16KB) + s_sub[BT*BK_SUB] bf16 (8KB)
    //   Phase D:   s_k at s_h pos bf16 (8KB) + s_v at s_sub pos bf16 (8KB)
    //   Phase E:   s_qk at s_h pos bf16 (16KB) then s_A at s_h pos fp32 (16KB)
    //              s_v at s_sub pos bf16 (8KB) — untouched since Phase B'
    //
    // Peak: 256 + 16384 + 8192 = 24832 bytes
    // =====================================================================
    extern __shared__ char smem_buf[];

    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);
    D_ACC*  s_pool = s_g + BT;

    // Pool sub-regions (aliased across phases)
    D_ACC*   s_h   = s_pool;                                        // [BK_SUB*BV] fp32
    D_ATTN*  s_sub = reinterpret_cast<D_ATTN*>(s_h + BK_SUB * BV); // [BT*BK_SUB] bf16

    // s_v aliases s_sub (same position, same size since BK_SUB==BV)
    D_ATTN*  s_v   = s_sub;

    // =====================================================================
    // HBM base pointers (batch + head offset applied)
    // =====================================================================
    const D_ATTN* q_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_q)
                          + (bos * H + i_h) * K;
    const D_ATTN* k_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                          + (bos * H + i_h) * K;
    const D_ATTN* w_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_w_bar)
                          + (bos * H + i_h) * K;
    const D_ATTN* u_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_u_bar)
                          + (bos * H + i_h) * V;
    const D_ACC*  g_hbm = reinterpret_cast<const D_ACC*>(kargs.ptr_g_cumsum)
                          + bos * H + i_h;
    D_ATTN* o_hbm  = reinterpret_cast<D_ATTN*>(kargs.ptr_o)
                     + (bos * H + i_h) * V;
    D_ATTN* vn_hbm = reinterpret_cast<D_ATTN*>(kargs.ptr_v_new)
                     + (bos * H + i_h) * V;

    // =====================================================================
    // Chunk-serial main loop
    // =====================================================================
    for (int i_t = 0; i_t < NT; i_t++) {
        const int t0 = i_t * BT;

        // --- Load g_cumsum[BT] for this chunk ---
        for (int i = tid; i < BT; i += BS) {
            s_g[i] = (t0 + i < kargs.T) ? g_hbm[(t0 + i) * stride_g] : 0.0f;
        }
        __syncthreads();

        int last_valid = (t0 + BT <= kargs.T) ? (BT - 1)
                                               : (kargs.T - t0 - 1);
        D_ACC g_last = s_g[last_valid];

        // =================================================================
        // (a) Store h_snapshot to HBM (pre-update state, for backward)
        // =================================================================
        if (kargs.ptr_h_snap != nullptr) {
            D_ACC* snap = reinterpret_cast<D_ACC*>(kargs.ptr_h_snap)
                + ((int64_t)i_n * NT * H + (int64_t)i_t * H + i_h) * K * V;
            for (int i = 0; i < H_ELEMS; i++) {
                int flat = tid + i * BS;
                int r = flat / BV;
                int c = flat % BV;
                if (v_off + c < V) {
                    snap[r * V + v_off + c] = h1[i];
                    if constexpr (N_K >= 2)
                        snap[(BK_SUB + r) * V + v_off + c] = h2[i];
                }
            }
        }

        // =================================================================
        // (b) Retrieve  +  (c) Cross-chunk
        //
        // retrieve[BT, BV] = Σ_bk  w_bar_sub[BT, 64] @ h[64, BV]
        // o_cross [BT, BV] = Σ_bk  q_sub    [BT, 64] @ h[64, BV]
        //
        // h is spilled to s_h for GEMM, then stays valid in registers.
        // =================================================================
        D_ACC r_retrieve[O_ELEMS];
        D_ACC r_o_cross[O_ELEMS];
        for (int i = 0; i < O_ELEMS; i++) {
            r_retrieve[i] = 0.0f;
            r_o_cross[i]  = 0.0f;
        }

        for (int bk = 0; bk < N_K; bk++) {
            D_ACC* h_cur = (bk == 0) ? h1 : h2;
            int k_off = bk * BK_SUB;

            // Spill h[bk] to s_h[BK_SUB*BV] fp32
            for (int i = 0; i < H_ELEMS; i++) {
                int flat = tid + i * BS;
                s_h[flat] = h_cur[i];
            }
            __syncthreads();

            // Load w_bar subtile [BT, BK_SUB] → s_sub
            for (int i = tid; i < BT * BK_SUB; i += BS) {
                int row = i / BK_SUB;
                int col = i % BK_SUB;
                s_sub[i] = (t0 + row < kargs.T)
                    ? w_hbm[(t0 + row) * stride_k + k_off + col]
                    : static_cast<D_ATTN>(0);
            }
            __syncthreads();

            // retrieve[s,c] += Σ_j w[s,j] * h[j,c]
            for (int i = 0; i < O_ELEMS; i++) {
                int flat = tid + i * BS;
                int s = flat / BV;
                int c = flat % BV;
                D_ACC acc = 0.0f;
                for (int j = 0; j < BK_SUB; j++) {
                    acc += static_cast<D_ACC>(s_sub[s * BK_SUB + j])
                         * s_h[j * BV + c];
                }
                r_retrieve[i] += acc;
            }
            __syncthreads();

            // Load q subtile [BT, BK_SUB] → s_sub (reuse)
            for (int i = tid; i < BT * BK_SUB; i += BS) {
                int row = i / BK_SUB;
                int col = i % BK_SUB;
                s_sub[i] = (t0 + row < kargs.T)
                    ? q_hbm[(t0 + row) * stride_k + k_off + col]
                    : static_cast<D_ATTN>(0);
            }
            __syncthreads();

            // o_cross[s,c] += Σ_j q[s,j] * h[j,c]
            for (int i = 0; i < O_ELEMS; i++) {
                int flat = tid + i * BS;
                int s = flat / BV;
                int c = flat % BV;
                D_ACC acc = 0.0f;
                for (int j = 0; j < BK_SUB; j++) {
                    acc += static_cast<D_ACC>(s_sub[s * BK_SUB + j])
                         * s_h[j * BV + c];
                }
                r_o_cross[i] += acc;
            }
            __syncthreads();
        }

        // Gate-scale o_cross: o_cross[s,:] *= exp(g_cumsum[s])
        for (int i = 0; i < O_ELEMS; i++) {
            int flat = tid + i * BS;
            int s = flat / BV;
            r_o_cross[i] *= __expf(s_g[s]);
        }

        // =================================================================
        // (b') v_new = u_bar - retrieve
        // Store to s_v (LDS, for Phase e AV) AND HBM (for backward).
        // s_v aliases s_sub — safe since s_sub is no longer needed.
        // =================================================================
        for (int i = 0; i < O_ELEMS; i++) {
            int flat = tid + i * BS;
            int s = flat / BV;
            int c = flat % BV;
            D_ACC u_val = 0.0f;
            if (t0 + s < kargs.T && v_off + c < V)
                u_val = static_cast<D_ACC>(
                    u_hbm[(t0 + s) * stride_v + v_off + c]);
            D_ACC v_new_val = u_val - r_retrieve[i];
            s_v[flat] = static_cast<D_ATTN>(v_new_val);
            if (t0 + s < kargs.T && v_off + c < V)
                vn_hbm[(t0 + s) * stride_v + v_off + c]
                    = static_cast<D_ATTN>(v_new_val);
        }
        __syncthreads();
        // r_retrieve[] freed logically

        // =================================================================
        // (d) Decay h + Accumulate h += k^T @ v_hat
        //
        // v_hat[s,c] = v_new[s,c] * exp(g_last - g[s])
        // Applied on-the-fly during accumulate so s_v retains v_new for Phase e.
        // =================================================================

        D_ACC decay = __expf(g_last);
        for (int i = 0; i < H_ELEMS; i++) {
            h1[i] *= decay;
            if constexpr (N_K >= 2) h2[i] *= decay;
        }

        for (int bk = 0; bk < N_K; bk++) {
            D_ACC* h_cur = (bk == 0) ? h1 : h2;
            int k_off = bk * BK_SUB;

            // Load k subtile [BT, BK_SUB] into s_h region (recast to bf16)
            D_ATTN* s_k = reinterpret_cast<D_ATTN*>(s_h);
            for (int i = tid; i < BT * BK_SUB; i += BS) {
                int row = i / BK_SUB;
                int col = i % BK_SUB;
                s_k[i] = (t0 + row < kargs.T)
                    ? k_hbm[(t0 + row) * stride_k + k_off + col]
                    : static_cast<D_ATTN>(0);
            }
            __syncthreads();

            // h[j,c] += Σ_s k[s,j] * v_new[s,c] * exp(g_last - g[s])
            for (int i = 0; i < H_ELEMS; i++) {
                int flat = tid + i * BS;
                int j = flat / BV;
                int c = flat % BV;
                D_ACC acc = 0.0f;
                for (int s = 0; s < BT; s++) {
                    D_ACC gate = (t0 + s < kargs.T)
                        ? __expf(g_last - s_g[s]) : 0.0f;
                    acc += static_cast<D_ACC>(s_k[s * BK_SUB + j])
                         * static_cast<D_ACC>(s_v[s * BV + c])
                         * gate;
                }
                h_cur[i] += acc;
            }
            __syncthreads();
        }

        // =================================================================
        // (e) Intra-chunk causal attention
        //
        // A_intra[BT,BT] = Σ_bk q_sub @ k_sub^T
        // A_intra[s,r] *= exp(g[s]-g[r]) if s>=r, else 0
        // o_intra[BT,BV] = A_intra @ v_new
        // =================================================================

        // QK^T: accumulate in registers
        D_ACC r_A[A_ELEMS];
        for (int i = 0; i < A_ELEMS; i++) r_A[i] = 0.0f;

        for (int bk = 0; bk < N_K; bk++) {
            int k_off = bk * BK_SUB;

            // s_h region holds 16KB fp32 = 32KB raw, fits 2× BT*BK_SUB bf16 (16KB)
            D_ATTN* s_q = reinterpret_cast<D_ATTN*>(s_h);
            D_ATTN* s_k = s_q + BT * BK_SUB;

            for (int i = tid; i < BT * BK_SUB; i += BS) {
                int row = i / BK_SUB;
                int col = i % BK_SUB;
                D_ATTN qv = static_cast<D_ATTN>(0);
                D_ATTN kv = static_cast<D_ATTN>(0);
                if (t0 + row < kargs.T) {
                    qv = q_hbm[(t0 + row) * stride_k + k_off + col];
                    kv = k_hbm[(t0 + row) * stride_k + k_off + col];
                }
                s_q[i] = qv;
                s_k[i] = kv;
            }
            __syncthreads();

            // A[s,r] += Σ_j q[s,j] * k[r,j]
            for (int i = 0; i < A_ELEMS; i++) {
                int flat = tid + i * BS;
                int s = flat / BT;
                int r = flat % BT;
                D_ACC acc = 0.0f;
                for (int j = 0; j < BK_SUB; j++) {
                    acc += static_cast<D_ACC>(s_q[s * BK_SUB + j])
                         * static_cast<D_ACC>(s_k[r * BK_SUB + j]);
                }
                r_A[i] += acc;
            }
            __syncthreads();
        }

        // Gate + causal mask
        for (int i = 0; i < A_ELEMS; i++) {
            int flat = tid + i * BS;
            int s = flat / BT;
            int r = flat % BT;
            if (s >= r && t0 + s < kargs.T && t0 + r < kargs.T)
                r_A[i] *= __expf(s_g[s] - s_g[r]);
            else
                r_A[i] = 0.0f;
        }

        // Store A_intra to LDS (reuse s_h region as fp32)
        D_ACC* s_A = s_pool;
        for (int i = 0; i < A_ELEMS; i++) {
            int flat = tid + i * BS;
            s_A[flat] = r_A[i];
        }
        __syncthreads();

        // AV matmul + combine with cross-chunk + store output
        // s_v still holds v_new from Phase B' (untouched by Phases D and E/QKT)
        for (int i = 0; i < O_ELEMS; i++) {
            int flat = tid + i * BS;
            int s = flat / BV;
            int c = flat % BV;
            D_ACC o_intra = 0.0f;
            for (int r = 0; r < BT; r++) {
                o_intra += s_A[s * BT + r]
                         * static_cast<D_ACC>(s_v[r * BV + c]);
            }
            D_ACC o_val = kargs.scale * (r_o_cross[i] + o_intra);
            if (t0 + s < kargs.T && v_off + c < V)
                o_hbm[(t0 + s) * stride_v + v_off + c]
                    = static_cast<D_ATTN>(o_val);
        }
        __syncthreads();

    }  // end chunk loop

    // =====================================================================
    // Epilogue: store final h → ptr_ht[N, H, K, V]
    // =====================================================================
    if (kargs.ptr_ht != nullptr) {
        D_ACC* ht = reinterpret_cast<D_ACC*>(kargs.ptr_ht)
                    + (i_n * H + i_h) * K * V;
        for (int i = 0; i < H_ELEMS; i++) {
            int flat = tid + i * BS;
            int r = flat / BV;
            int c = flat % BV;
            if (v_off + c < V) {
                ht[r * V + v_off + c] = h1[i];
                if constexpr (N_K >= 2)
                    ht[(BK_SUB + r) * V + v_off + c] = h2[i];
            }
        }
    }
}
