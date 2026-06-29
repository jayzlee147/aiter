// GDN Prefill K2-OUT Kernel — parallel-over-chunks output (fwd_o analog)
//
// Companion to gdn_k2_kernel<...,SCAN_ONLY=true> (the serial scan). The scan
// materializes, per chunk, the pre-update hidden state h_snap[B,NT,H,V,K] (bf16,
// stored transposed [V,K]) and v_new[B,T,H,V]. This kernel then computes the
// output for every chunk INDEPENDENTLY:
//
//   o[t] = scale * ( q[t] @ H_{t-1}                       (cross-chunk)
//                  + causal_gated(q[t] @ k[t]^T) @ v_new[t] )   (intra-chunk)
//
// Grid: (cdiv(V,BV), NT, B*H)  Block: (BLOCK_SIZE)
// The NT grid dimension is what restores GPU utilization on long single
// sequences (low B*H) where the fused k2 starves the device.
//
// Target: gfx942 (MI300X) / gfx950 (MI350), MFMA bf16 16x16x16
#pragma once

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_mfma_utils.h"

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE)
gdn_k2_out_kernel(gdn_k2_kargs kargs) {
    using namespace gdn_mfma;
    using T = Traits;
    using D_ATTN = typename T::D_ATTN;   // bf16
    using D_ACC  = typename T::D_ACC;    // fp32

    constexpr int BT     = T::BT;
    constexpr int BK_SUB = T::BK_SUB;    // 64
    constexpr int BV     = T::BV;        // 64
    constexpr int N_K    = T::N_K;       // K / BK_SUB
    constexpr int BS     = T::BLOCK_SIZE;
    constexpr int WS     = T::WARP_SIZE; // 64
    constexpr int PAD    = T::SMEM_PAD;
    constexpr int W      = 16;

    // Output GEMM (o_cross / o_intra) tiling: C[BT, BV]
    constexpr bool BT_LARGE = (BT >= 32);
    constexpr int O_T_M = BT_LARGE
        ? ((T::NUM_WARPS < BT / W) ? T::NUM_WARPS : BT / W) : 1;
    constexpr int O_T_N = T::NUM_WARPS / O_T_M;
    constexpr int O_E_M = BT / (W * O_T_M);
    constexpr int O_E_N = BV / (W * O_T_N);
    constexpr int O_E_K = BK_SUB / W;

    // QK^T GEMM tiling: C[BT, BT]
    constexpr int QKT_T_M = BT_LARGE
        ? ((T::NUM_WARPS < BT / W) ? T::NUM_WARPS : BT / W) : 1;
    constexpr int QKT_T_N = BT_LARGE ? (T::NUM_WARPS / QKT_T_M) : T::NUM_WARPS;
    constexpr int QKT_E_M = BT / (W * QKT_T_M);
    constexpr int QKT_E_N = BT / (W * QKT_T_N);
    constexpr int QKT_E_K = BK_SUB / W;

    constexpr int STRIDE_BK = BK_SUB + PAD;
    constexpr int STRIDE_BT = BT + PAD;

    const int i_v  = blockIdx.x;
    const int i_t  = blockIdx.y;
    const int i_nh = blockIdx.z;
    const int i_n  = i_nh / kargs.H;
    const int i_h  = i_nh % kargs.H;
    const int tid  = threadIdx.x;
    const int warp_id = tid / WS;
    const int lane_id = tid % WS;

    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;
    const int NT = kargs.NT;
    const int v_off = i_v * BV;
    const int bos   = i_n * kargs.T;
    const int t0    = i_t * BT;
    const int stride_k = H * K;
    const int stride_v = H * V;
    const int stride_g = H;

    const int T_rem = kargs.T - t0;
    const bool full_chunk = (T_rem >= BT);
    const bool v_full = (v_off + BV <= V);

    // Warp tile bases (same convention as fused k2)
    int o_m_base, o_n_base, qkt_m_base, qkt_n_base;
    if constexpr (BT_LARGE) {
        o_m_base   = (warp_id / O_T_N)   * (O_E_M * W);
        o_n_base   = (warp_id % O_T_N)   * (O_E_N * W);
        qkt_m_base = (warp_id / QKT_T_N) * (QKT_E_M * W);
        qkt_n_base = (warp_id % QKT_T_N) * (QKT_E_N * W);
    } else {
        o_m_base = 0; o_n_base = warp_id * W;
        qkt_m_base = 0; qkt_n_base = warp_id * W;
    }

    // ---- LDS layout ----
    extern __shared__ char smem_buf[];
    D_ACC*  s_g   = reinterpret_cast<D_ACC*>(smem_buf);
    D_ATTN* s_q   = reinterpret_cast<D_ATTN*>(smem_buf + BT * sizeof(D_ACC));   // [N_K, BT, STRIDE_BK]
    D_ATTN* s_kh  = s_q + N_K * BT * STRIDE_BK;                                 // [BV or BT, STRIDE_BK] (h^T / k, per bk)
    D_ATTN* s_vT  = s_kh + (BV > BT ? BV : BT) * STRIDE_BK;                     // [BV, STRIDE_BT] (v_new^T)
    D_ATTN* s_A5  = s_vT + BV * STRIDE_BT;                                      // [BT, STRIDE_BT]

    // ---- HBM base pointers ----
    const D_ATTN* q_ch = reinterpret_cast<const D_ATTN*>(kargs.ptr_q)
                         + ((bos + t0) * H + i_h) * K;
    const D_ATTN* k_ch = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                         + ((bos + t0) * H + i_h) * K;
    const D_ATTN* vn_ch = reinterpret_cast<const D_ATTN*>(kargs.ptr_v_new)
                          + ((bos + t0) * H + i_h) * V;
    const D_ACC* g_ch = reinterpret_cast<const D_ACC*>(kargs.ptr_g_cumsum)
                        + (bos + t0) * H + i_h;
    // h_snap: [B, NT, H, V, K] (stored transposed [V,K]); subtile bk lives at +bk*BK_SUB in K
    const D_ATTN* hsnap = reinterpret_cast<const D_ATTN*>(kargs.ptr_h_snap)
                          + ((int64_t)(i_n * NT + i_t) * H + i_h) * V * K;
    D_ATTN* o_ch = reinterpret_cast<D_ATTN*>(kargs.ptr_o)
                   + ((bos + t0) * H + i_h) * V;

    constexpr int VEC = 4;
    constexpr int K_NVEC = T::K / VEC;

    // ---- Load g_cumsum[BT] ----
    for (int i = tid; i < BT; i += BS)
        s_g[i] = (i < T_rem) ? g_ch[i * stride_g] : 0.0f;

    // ---- Load q[BT,K] into s_q (all N_K subtiles, [BT, STRIDE_BK] each) ----
    for (int i = tid; i < BT * K_NVEC; i += BS) {
        int row = i / K_NVEC;
        int colv = (i % K_NVEC) * VEC;
        int bk = colv / BK_SUB;
        int col = colv % BK_SUB;
        v4bf16_t val = {};
        if (full_chunk || row < T_rem)
            val = *reinterpret_cast<const v4bf16_t*>(&q_ch[row * stride_k + colv]);
        *reinterpret_cast<v4bf16_t*>(&s_q[bk * BT * STRIDE_BK + row * STRIDE_BK + col]) = val;
    }
    // PREFETCH v_new from HBM into registers now, so its latency overlaps the
    // bk-loop GEMMs instead of stalling the A@v_new stage (num_stages-style).
    constexpr int V_NVEC = BV / VEC;
    constexpr int VN_ITERS = (BT * V_NVEC + BS - 1) / BS;
    v4bf16_t vn_reg[VN_ITERS];
    #pragma unroll
    for (int it = 0; it < VN_ITERS; it++) {
        int i = tid + it * BS;
        vn_reg[it] = v4bf16_t{};
        if (i < BT * V_NVEC) {
            int row = i / V_NVEC, cc = (i % V_NVEC) * VEC;
            if ((full_chunk || row < T_rem) && (v_full || v_off + cc < V))
                vn_reg[it] = *reinterpret_cast<const v4bf16_t*>(&vn_ch[row * stride_v + v_off + cc]);
        }
    }
    __syncthreads();

    // =====================================================================
    // o_cross[BT,BV] = Sum_bk q[:,bk] @ H[bk,:]      (H^T resident as [V,K])
    // A[BT,BT]       = Sum_bk q[:,bk] @ k[:,bk]^T
    // =====================================================================
    constexpr int C_ELEMS = O_E_M * O_E_N;
    v4f32_t r_o[C_ELEMS];
    v4f32_t r_A[QKT_E_M * QKT_E_N];
    clear_v4f32<C_ELEMS>(r_o);
    clear_v4f32<QKT_E_M * QKT_E_N>(r_A);

    // Software-pipelined bk loop (num_stages=2): prefetch bk+1's h/k from HBM
    // into registers while bk's GEMMs run, hiding load latency. de-aliased
    // h^T -> s_kh, k -> s_A5 so both sub-GEMMs run after ONE barrier.
    constexpr int HN = (BV * (BK_SUB / VEC) + BS - 1) / BS;
    constexpr int KN = (BT * (BK_SUB / VEC) + BS - 1) / BS;
    v4bf16_t hr[HN], kr[KN];
    auto load_hk = [&](int bk) {
        int koff = bk * BK_SUB;
        #pragma unroll
        for (int it = 0; it < HN; it++) {
            int i = tid + it * BS; hr[it] = v4bf16_t{};
            if (i < BV * (BK_SUB / VEC)) {
                int col = i / (BK_SUB / VEC), kk = (i % (BK_SUB / VEC)) * VEC;
                if (v_full || v_off + col < V)
                    hr[it] = *reinterpret_cast<const v4bf16_t*>(&hsnap[(v_off + col) * K + koff + kk]);
            }
        }
        #pragma unroll
        for (int it = 0; it < KN; it++) {
            int i = tid + it * BS; kr[it] = v4bf16_t{};
            if (i < BT * (BK_SUB / VEC)) {
                int row = i / (BK_SUB / VEC), kk = (i % (BK_SUB / VEC)) * VEC;
                if (full_chunk || row < T_rem)
                    kr[it] = *reinterpret_cast<const v4bf16_t*>(&k_ch[row * stride_k + koff + kk]);
            }
        }
    };
    load_hk(0);                                    // prologue
    for (int bk = 0; bk < N_K; bk++) {
        #pragma unroll
        for (int it = 0; it < HN; it++) { int i = tid + it * BS;
            if (i < BV * (BK_SUB / VEC)) { int col = i/(BK_SUB/VEC), kk=(i%(BK_SUB/VEC))*VEC;
                *reinterpret_cast<v4bf16_t*>(&s_kh[col * STRIDE_BK + kk]) = hr[it]; } }
        #pragma unroll
        for (int it = 0; it < KN; it++) { int i = tid + it * BS;
            if (i < BT * (BK_SUB / VEC)) { int row = i/(BK_SUB/VEC), kk=(i%(BK_SUB/VEC))*VEC;
                *reinterpret_cast<v4bf16_t*>(&s_A5[row * STRIDE_BK + kk]) = kr[it]; } }
        __syncthreads();
        if (bk + 1 < N_K) load_hk(bk + 1);         // overlaps the GEMMs below
        tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
            r_o, s_q + bk * BT * STRIDE_BK, o_m_base, STRIDE_BK,
                 s_kh,                       o_n_base, STRIDE_BK, lane_id);
        tiled_gemm_mfma<QKT_E_M, QKT_E_N, QKT_E_K>(
            r_A, s_q + bk * BT * STRIDE_BK, qkt_m_base, STRIDE_BK,
                 s_A5,                       qkt_n_base, STRIDE_BK, lane_id);
        __syncthreads();
    }

    // ---- gate o_cross: o_cross[s,:] *= exp(g[s]) ----
    for (int i = 0; i < C_ELEMS; i++) {
        int s_base = BT_LARGE ? (o_m_base + (i / O_E_N) * W + (lane_id >> 4) * 4)
                              : ((lane_id >> 4) * 4);
        for (int p = 0; p < 4; p++) {
            int row = BT_LARGE ? (s_base + p) : ((i / O_E_N) * W + s_base + p);
            r_o[i][p] *= fast_exp(s_g[row]);
        }
    }

    // ---- gate + causal mask A, store to s_A5[BT, STRIDE_BT] ----
    for (int i = 0; i < QKT_E_M * QKT_E_N; i++) {
        int en = i % QKT_E_N;
        int row_base = BT_LARGE ? (qkt_m_base + (i / QKT_E_N) * W + (lane_id >> 4) * 4)
                                : ((lane_id >> 4) * 4);
        int col = qkt_n_base + en * W + (lane_id & 15);
        for (int p = 0; p < 4; p++) {
            int s = row_base + p;
            D_ACC a = 0.0f;
            if (s >= col && (full_chunk || (s < T_rem && col < T_rem)))
                a = r_A[i][p] * fast_exp(s_g[s] - s_g[col]);
            s_A5[s * STRIDE_BT + col] = static_cast<D_ATTN>(a);
        }
    }

    // ---- store prefetched v_new transposed -> s_vT[BV, STRIDE_BT] (no HBM wait) ----
    #pragma unroll
    for (int it = 0; it < VN_ITERS; it++) {
        int i = tid + it * BS;
        if (i < BT * V_NVEC) {
            int row = i / V_NVEC;            // s (time)
            int cc  = (i % V_NVEC) * VEC;    // v within tile
            for (int q = 0; q < VEC; q++)
                s_vT[(cc + q) * STRIDE_BT + row] = vn_reg[it][q];
        }
    }
    __syncthreads();

    // ---- o_intra = A @ v_new ; o = scale*(o_cross + o_intra) ----
    constexpr int AV_E_K = BT / W;
    v4f32_t r_oi[C_ELEMS];
    clear_v4f32<C_ELEMS>(r_oi);
    tiled_gemm_mfma<O_E_M, O_E_N, AV_E_K>(
        r_oi, s_A5, o_m_base, STRIDE_BT,
              s_vT, o_n_base, STRIDE_BT, lane_id);

    for (int i = 0; i < C_ELEMS; i++) {
        int en = i % O_E_N;
        for (int p = 0; p < 4; p++) {
            int s = BT_LARGE ? (o_m_base + (i / O_E_N) * W + (lane_id >> 4) * 4 + p)
                             : ((i / O_E_N) * W + (lane_id >> 4) * 4 + p);
            int c = o_n_base + en * W + (lane_id & 15);
            D_ACC val = kargs.scale * (r_o[i][p] + r_oi[i][p]);
            if ((full_chunk || s < T_rem) && (v_full || v_off + c < V))
                o_ch[s * stride_v + v_off + c] = static_cast<D_ATTN>(val);
        }
    }
}
