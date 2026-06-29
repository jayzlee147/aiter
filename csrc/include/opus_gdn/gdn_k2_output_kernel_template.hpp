// GDN Prefill K2 Output Kernel — phases c (cross-chunk) + e (intra-chunk)
// Embarrassingly parallel: one workgroup per (chunk, v_slice) pair.
// Reads h_snap (fp32, [K,V] layout) and v_new (bf16) from scan kernel.
//
// Grid: (NT * cdiv(V, BV), B*H)   Block: (BLOCK_SIZE)
// No serial dependency — all chunks are independent.
//
// h_snap layout: [B, NT, H, K, V] fp32  (K-major, matches scan kernel output)
#pragma once

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_mfma_utils.h"

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, 1)
gdn_k2_output_kernel(gdn_k2_output_kargs kargs) {
    using namespace gdn_mfma;
    using T = Traits;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    constexpr int BT     = T::BT;
    constexpr int BK_SUB = T::BK_SUB;
    constexpr int BV     = T::BV;
    constexpr int N_K    = T::N_K;
    constexpr int BS     = T::BLOCK_SIZE;
    constexpr int WS     = T::WARP_SIZE;
    constexpr int PAD    = T::SMEM_PAD;
    constexpr int W      = 16;

    // Output GEMM tiling: C[BT, BV]
    constexpr bool BT_LARGE = (BT >= 32);
    constexpr int O_T_M = BT_LARGE
        ? ((T::NUM_WARPS < BT / W) ? T::NUM_WARPS : BT / W) : 1;
    constexpr int O_T_N = T::NUM_WARPS / O_T_M;
    constexpr int O_E_M = BT / (W * O_T_M);
    constexpr int O_E_N = BV / (W * O_T_N);
    constexpr int O_E_K = BK_SUB / W;
    static_assert(O_E_N > 0);

    // QK^T GEMM tiling: C[BT, BT]
    constexpr int QKT_T_M = BT_LARGE
        ? ((T::NUM_WARPS < BT / W) ? T::NUM_WARPS : BT / W) : 1;
    constexpr int QKT_T_N = BT_LARGE ? (T::NUM_WARPS / QKT_T_M) : T::NUM_WARPS;
    constexpr int QKT_E_M = BT / (W * QKT_T_M);
    constexpr int QKT_E_N = BT / (W * QKT_T_N);
    constexpr int QKT_E_K = BK_SUB / W;

    constexpr int STRIDE_BK = BK_SUB + PAD;
    constexpr int STRIDE_BT = BT + PAD;

    // Decode grid index: blockIdx.x = i_t * NV + i_v
    const int i_tv = blockIdx.x;
    const int i_t  = i_tv / kargs.NV;
    const int i_v  = i_tv % kargs.NV;
    const int i_nh = blockIdx.y;
    const int i_n  = i_nh / kargs.H;
    const int i_h  = i_nh % kargs.H;
    const int tid  = threadIdx.x;
    const int warp_id = tid / WS;
    const int lane_id = tid % WS;

    const int v_off = i_v * BV;
    const int t0 = i_t * BT;
    const int bos = i_n * kargs.T;
    const int K = kargs.K;
    const int V = kargs.V;
    const int H = kargs.H;
    const int NT = kargs.NT;

    const int stride_k = H * K;
    const int stride_v = H * V;
    const int stride_g = H;

    int o_m_base, o_n_base;
    if constexpr (BT_LARGE) {
        o_m_base = (warp_id / O_T_N) * (O_E_M * W);
        o_n_base = (warp_id % O_T_N) * (O_E_N * W);
    } else {
        o_m_base = 0;
        o_n_base = warp_id * W;
    }

    int qkt_m_base, qkt_n_base;
    if constexpr (BT_LARGE) {
        qkt_m_base = (warp_id / QKT_T_N) * (QKT_E_M * W);
        qkt_n_base = (warp_id % QKT_T_N) * (QKT_E_N * W);
    } else {
        qkt_m_base = 0;
        qkt_n_base = warp_id * W;
    }

    const int T_rem = kargs.T - t0;
    const bool full_chunk = (T_rem >= BT);
    const bool v_full = (v_off + BV <= V);

    // Shared memory layout:
    //   s_g[BT] fp32                   — gate cumsum (persistent)
    //   s_v_T[BV, STRIDE_BT] bf16     — v_new transposed (persistent for AV)
    //   pool:  phase c: s_h_T[BV, STRIDE_BK] + s_sub[BT, STRIDE_BK]
    //          phase e: s_k[BT, STRIDE_BK] (reuses s_h_T area)
    //                   s_A[BT, STRIDE_BT] (reuses s_sub area or pool)
    constexpr int smem_g_bytes  = BT * (int)sizeof(D_ACC);
    constexpr int smem_vT_bytes = BV * STRIDE_BT * (int)sizeof(D_ATTN);

    extern __shared__ char smem_buf[];
    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);
    D_ATTN* s_v_T  = reinterpret_cast<D_ATTN*>(smem_buf + smem_g_bytes);
    D_ATTN* s_pool = reinterpret_cast<D_ATTN*>(smem_buf + smem_g_bytes + smem_vT_bytes);

    // Load g_cumsum
    const D_ACC* g_hbm = reinterpret_cast<const D_ACC*>(kargs.ptr_g_cumsum)
                        + (bos + t0) * stride_g + i_h;
    for (int i = tid; i < BT; i += BS)
        s_g[i] = (i < T_rem) ? g_hbm[i * stride_g] : 0.0f;

    // Load v_new → s_v_T (transposed: [BV, STRIDE_BT])
    const D_ATTN* vn_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_v_new)
                          + ((int64_t)(bos + t0) * H + i_h) * V;
    constexpr int VN_ELEMS = BT * BV;
    for (int i = tid; i < VN_ELEMS; i += BS) {
        int s = i / BV;
        int c = i % BV;
        D_ATTN val = {};
        if ((full_chunk || s < T_rem) && (v_full || v_off + c < V))
            val = vn_hbm[s * stride_v + v_off + c];
        s_v_T[c * STRIDE_BT + s] = val;
    }
    __syncthreads();

    // =================================================================
    // Phase (c): Cross-chunk output via MFMA
    // o_cross[BT, BV] = Σ_bk q[BT, BK_SUB] × h_snap[BK_SUB, BV]
    // h_snap is fp32 [K, V] layout → load and convert to bf16 in LDS
    // =================================================================
    constexpr int C_ELEMS = O_E_M * O_E_N;
    v4f32_t r_o_cross[C_ELEMS];
    clear_v4f32<C_ELEMS>(r_o_cross);

    D_ATTN* s_h_T  = s_pool;                    // [BV, STRIDE_BK] — h transposed
    D_ATTN* s_sub  = s_pool + BV * STRIDE_BK;   // [BT, STRIDE_BK] — q or k

    const D_ATTN* q_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_q)
                         + ((int64_t)(bos + t0) * H + i_h) * K;
    const D_ATTN* k_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                         + ((int64_t)(bos + t0) * H + i_h) * K;
    const D_ACC* h_snap = reinterpret_cast<const D_ACC*>(kargs.ptr_h_snap)
                        + ((int64_t)i_n * NT * H + (int64_t)i_t * H + i_h) * K * V;

    // Prefetch q[bk=0]
    constexpr int PF_VEC   = 4;
    constexpr int PF_NVEC  = BK_SUB / PF_VEC;
    constexpr int PF_ELEMS = BT * PF_NVEC;
    constexpr int PF_LOADS = (PF_ELEMS + BS - 1) / BS;

    v4bf16_t pf_a[PF_LOADS];  // carries q (phase c) then k (phase e)
    #pragma unroll
    for (int li = 0; li < PF_LOADS; li++) {
        int i = tid + li * BS;
        pf_a[li] = {};
        if (i < PF_ELEMS) {
            int row = i / PF_NVEC;
            int col = (i % PF_NVEC) * PF_VEC;
            if (full_chunk || row < T_rem)
                pf_a[li] = *reinterpret_cast<const v4bf16_t*>(
                    &q_hbm[row * stride_k + col]);
        }
    }

    for (int bk = 0; bk < N_K; bk++) {
        // Load h_snap[bk] from HBM (fp32) → s_h_T as bf16 transposed
        // h_snap layout: [K, V] fp32, we need s_h_T[BV, BK_SUB+PAD] bf16
        constexpr int H_ELEMS = BK_SUB * BV;
        for (int i = tid; i < H_ELEMS; i += BS) {
            int k_idx = i / BV;       // row in h (K dimension)
            int v_idx = i % BV;       // col in h (V dimension)
            D_ACC val = 0.0f;
            if (v_full || v_off + v_idx < V)
                val = h_snap[(bk * BK_SUB + k_idx) * V + (v_off + v_idx)];
            s_h_T[v_idx * STRIDE_BK + k_idx] = static_cast<D_ATTN>(val);
        }

        // Install q[bk] → s_sub
        #pragma unroll
        for (int li = 0; li < PF_LOADS; li++) {
            int i = tid + li * BS;
            if (i < PF_ELEMS) {
                int row = i / PF_NVEC;
                int col = (i % PF_NVEC) * PF_VEC;
                *reinterpret_cast<v4bf16_t*>(&s_sub[row * STRIDE_BK + col]) = pf_a[li];
            }
        }
        __syncthreads();

        // Prefetch q[bk+1] or k[bk=0] for phase e
        if (bk + 1 < N_K) {
            int k_off = (bk + 1) * BK_SUB;
            #pragma unroll
            for (int li = 0; li < PF_LOADS; li++) {
                int i = tid + li * BS;
                pf_a[li] = {};
                if (i < PF_ELEMS) {
                    int row = i / PF_NVEC;
                    int col = (i % PF_NVEC) * PF_VEC;
                    if (full_chunk || row < T_rem)
                        pf_a[li] = *reinterpret_cast<const v4bf16_t*>(
                            &q_hbm[row * stride_k + k_off + col]);
                }
            }
        } else {
            // Last bk of phase c: prefetch k[bk=0] for phase e
            #pragma unroll
            for (int li = 0; li < PF_LOADS; li++) {
                int i = tid + li * BS;
                pf_a[li] = {};
                if (i < PF_ELEMS) {
                    int row = i / PF_NVEC;
                    int col = (i % PF_NVEC) * PF_VEC;
                    if (full_chunk || row < T_rem)
                        pf_a[li] = *reinterpret_cast<const v4bf16_t*>(
                            &k_hbm[row * stride_k + col]);
                }
            }
        }

        // Cross-chunk GEMM: o_cross += q × h_T
        tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
            r_o_cross, s_sub,  o_m_base, STRIDE_BK,
                       s_h_T,  o_n_base, STRIDE_BK, lane_id);
        __syncthreads();
    }

    // Gate-scale o_cross: *= exp(g_cumsum[s])
    for (int i = 0; i < C_ELEMS; i++) {
        int s;
        if constexpr (BT_LARGE)
            s = o_m_base + (i / O_E_N) * W + (lane_id >> 4) * 4;
        else
            s = (lane_id >> 4) * 4;
        for (int p = 0; p < 4; p++) {
            int sp = s + p;
            int row = BT_LARGE ? sp : ((i / O_E_N) * W + sp);
            r_o_cross[i][p] *= fast_exp(s_g[row]);
        }
    }

    // =================================================================
    // Phase (e): Intra-chunk causal attention
    // QK^T → gate + mask → AV
    // =================================================================
    {
        D_ATTN* s_k4 = s_pool;
        D_ATTN* s_A5 = s_pool;

        // Persistent q: reload q[bk=0] from HBM for QK^T
        // (pf_a currently holds k[bk=0] from the end of phase c)
        // We need q for QK^T. Load q[bk=0] into s_sub (reuse), then use pf_a for k.
        // Actually, let's load q into a persistent region and k into s_k4.

        // Strategy: use s_h_T area for persistent q (same size: BT × STRIDE_BK),
        // and s_k4 = s_pool for k. But s_h_T and s_pool overlap...
        // Better: use s_pool bottom for q (persistent across bk), top for k
        D_ATTN* s_q_e  = s_pool;                     // [BT, STRIDE_BK] — q for QK^T
        D_ATTN* s_k_e  = s_pool + BT * STRIDE_BK;    // [BT, STRIDE_BK] — k for QK^T

        if constexpr (BT >= 32) {
            v4f32_t r_A[QKT_E_M * QKT_E_N];
            clear_v4f32<QKT_E_M * QKT_E_N>(r_A);

            for (int bk = 0; bk < N_K; bk++) {
                // Load q[bk] from HBM → s_q_e
                int q_off = bk * BK_SUB;
                for (int i = tid; i < PF_ELEMS; i += BS) {
                    int row = i / PF_NVEC;
                    int col = (i % PF_NVEC) * PF_VEC;
                    v4bf16_t val = {};
                    if (full_chunk || row < T_rem)
                        val = *reinterpret_cast<const v4bf16_t*>(
                            &q_hbm[row * stride_k + q_off + col]);
                    *reinterpret_cast<v4bf16_t*>(&s_q_e[row * STRIDE_BK + col]) = val;
                }

                // Install pf_a (k[bk]) → s_k_e
                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    if (i < PF_ELEMS) {
                        int row = i / PF_NVEC;
                        int col = (i % PF_NVEC) * PF_VEC;
                        *reinterpret_cast<v4bf16_t*>(&s_k_e[row * STRIDE_BK + col]) = pf_a[li];
                    }
                }
                __syncthreads();

                // Prefetch k[bk+1]
                if (bk + 1 < N_K) {
                    int k_off = (bk + 1) * BK_SUB;
                    #pragma unroll
                    for (int li = 0; li < PF_LOADS; li++) {
                        int i = tid + li * BS;
                        pf_a[li] = {};
                        if (i < PF_ELEMS) {
                            int row = i / PF_NVEC;
                            int col = (i % PF_NVEC) * PF_VEC;
                            if (full_chunk || row < T_rem)
                                pf_a[li] = *reinterpret_cast<const v4bf16_t*>(
                                    &k_hbm[row * stride_k + k_off + col]);
                        }
                    }
                }

                // QK^T GEMM: r_A += s_q_e × s_k_e
                tiled_gemm_mfma<QKT_E_M, QKT_E_N, QKT_E_K>(
                    r_A, s_q_e, qkt_m_base, STRIDE_BK,
                         s_k_e, qkt_n_base, STRIDE_BK, lane_id);
                __syncthreads();
            }

            // Gate + causal mask
            for (int i = 0; i < QKT_E_M * QKT_E_N; i++) {
                int en = i % QKT_E_N;
                for (int p = 0; p < 4; p++) {
                    int s, r;
                    if constexpr (BT_LARGE) {
                        s = qkt_m_base + (i / QKT_E_N) * W + (lane_id >> 4) * 4 + p;
                        r = qkt_n_base + en * W + (lane_id & 15);
                    } else {
                        s = (lane_id >> 4) * 4 + p;
                        r = qkt_n_base + en * W + (lane_id & 15);
                    }
                    if (s >= r && (full_chunk || (s < T_rem && r < T_rem)))
                        r_A[i][p] *= fast_exp(s_g[s] - s_g[r]);
                    else
                        r_A[i][p] = 0.0f;
                }
            }

            // Store A_intra as bf16 to s_A5[BT, STRIDE_BT]
            D_ATTN* s_A = s_pool;
            for (int i = 0; i < QKT_E_M * QKT_E_N; i++) {
                int en = i % QKT_E_N;
                int row_base, col_base;
                if constexpr (BT_LARGE) {
                    row_base = qkt_m_base + (i / QKT_E_N) * W + (lane_id >> 4) * 4;
                    col_base = qkt_n_base + en * W + (lane_id & 15);
                } else {
                    row_base = (lane_id >> 4) * 4;
                    col_base = qkt_n_base + en * W + (lane_id & 15);
                }
                for (int p = 0; p < 4; p++)
                    s_A[(row_base + p) * STRIDE_BT + col_base] =
                        static_cast<D_ATTN>(r_A[i][p]);
            }

        } else {
            // BT < 32: single MFMA tile
            v4f32_t r_A[1];
            clear_v4f32<1>(r_A);

            for (int bk = 0; bk < N_K; bk++) {
                int q_off = bk * BK_SUB;
                for (int i = tid; i < PF_ELEMS; i += BS) {
                    int row = i / PF_NVEC;
                    int col = (i % PF_NVEC) * PF_VEC;
                    v4bf16_t val = {};
                    if (full_chunk || row < T_rem)
                        val = *reinterpret_cast<const v4bf16_t*>(
                            &q_hbm[row * stride_k + q_off + col]);
                    *reinterpret_cast<v4bf16_t*>(&s_q_e[row * STRIDE_BK + col]) = val;
                }
                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    if (i < PF_ELEMS) {
                        int row = i / PF_NVEC;
                        int col = (i % PF_NVEC) * PF_VEC;
                        *reinterpret_cast<v4bf16_t*>(&s_k_e[row * STRIDE_BK + col]) = pf_a[li];
                    }
                }
                __syncthreads();
                if (bk + 1 < N_K) {
                    int k_off = (bk + 1) * BK_SUB;
                    #pragma unroll
                    for (int li = 0; li < PF_LOADS; li++) {
                        int i = tid + li * BS;
                        pf_a[li] = {};
                        if (i < PF_ELEMS) {
                            int row = i / PF_NVEC;
                            int col = (i % PF_NVEC) * PF_VEC;
                            if (full_chunk || row < T_rem)
                                pf_a[li] = *reinterpret_cast<const v4bf16_t*>(
                                    &k_hbm[row * stride_k + k_off + col]);
                        }
                    }
                }
                if (warp_id == 0) {
                    tiled_gemm_mfma<1, 1, QKT_E_K>(
                        r_A, s_q_e, 0, STRIDE_BK,
                             s_k_e, 0, STRIDE_BK, lane_id);
                }
                __syncthreads();
            }

            D_ATTN* s_A = s_pool;
            if (warp_id == 0) {
                for (int p = 0; p < 4; p++) {
                    int s = (lane_id >> 4) * 4 + p;
                    int r = lane_id & 15;
                    if (s >= r && (full_chunk || (s < T_rem && r < T_rem)))
                        r_A[0][p] *= fast_exp(s_g[s] - s_g[r]);
                    else
                        r_A[0][p] = 0.0f;
                    s_A[s * STRIDE_BT + r] = static_cast<D_ATTN>(r_A[0][p]);
                }
            }
        }
        __syncthreads();

        // AV GEMM: o_intra = A × v_new
        constexpr int AV_E_K = BT / W;
        v4f32_t r_o_intra[C_ELEMS];
        clear_v4f32<C_ELEMS>(r_o_intra);

        D_ATTN* s_A_rd = s_pool;
        tiled_gemm_mfma<O_E_M, O_E_N, AV_E_K>(
            r_o_intra,
            s_A_rd, o_m_base, STRIDE_BT,
            s_v_T,  o_n_base, STRIDE_BT,
            lane_id);

        // Combine: o = scale * (o_cross + o_intra)
        D_ATTN* o_hbm = reinterpret_cast<D_ATTN*>(kargs.ptr_o)
                       + ((int64_t)(bos + t0) * H + i_h) * V;
        for (int i = 0; i < C_ELEMS; i++) {
            int en = i % O_E_N;
            for (int p = 0; p < 4; p++) {
                int s, c;
                if constexpr (BT_LARGE) {
                    s = o_m_base + (i / O_E_N) * W + (lane_id >> 4) * 4 + p;
                    c = o_n_base + en * W + (lane_id & 15);
                } else {
                    s = (lane_id >> 4) * 4 + p;
                    c = o_n_base + en * W + (lane_id & 15);
                }
                D_ACC o_val = kargs.scale * (r_o_cross[i][p] + r_o_intra[i][p]);
                if ((full_chunk || s < T_rem) && (v_full || v_off + c < V))
                    o_hbm[s * stride_v + v_off + c] = static_cast<D_ATTN>(o_val);
            }
        }
    }
}
