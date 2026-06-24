// GDN Prefill K2 Kernel — MFMA bf16 16×16×16 optimized
// Step 3: Hidden state update (retrieve, gate-scale, decay, accumulate)
// Step 4: Output (cross-chunk QH, intra-chunk causal attention)
//
// Grid: (cdiv(V, BV), N*H)   Block: (BLOCK_SIZE = 256)
// h state: register-resident in MFMA accumulator layout, N_K × [BK_SUB, BV]
// Chunks: serial iteration within each workgroup
//
// 5 GEMMs via MFMA bf16 16×16×16 (BT≥32), scalar fallback for BT<32 GEMM4/5
// Target: gfx942 (MI300X) / gfx950 (MI350)
#pragma once

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_mfma_utils.h"

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, Traits::OCC_HINT)
gdn_k2_kernel(gdn_k2_kargs kargs) {
    using namespace gdn_mfma;
    using T = Traits;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    constexpr int BT     = T::BT;
    constexpr int BK_SUB = T::BK_SUB;     // 64
    constexpr int BV     = T::BV;          // 64
    constexpr int N_K    = T::N_K;         // 2 for K=128
    constexpr int BS     = T::BLOCK_SIZE;  // 256
    constexpr int WS     = T::WARP_SIZE;   // 64
    constexpr int PAD    = T::SMEM_PAD;    // 4

    // MFMA 16×16×16 tile dimensions
    constexpr int W = 16;

    // h-accumulate GEMM3 tiling: C[BK_SUB, BV], 2D warp tiling
    constexpr int H_T_M = (T::NUM_WARPS < BK_SUB / W) ? T::NUM_WARPS : BK_SUB / W;
    constexpr int H_T_N = T::NUM_WARPS / H_T_M;
    constexpr int H_E_M = BK_SUB / (W * H_T_M);
    constexpr int H_E_N = BV / (W * H_T_N);
    constexpr int H_E_K = BT / W;
    static_assert(H_E_N > 0, "Too many warps for BV dimension (h state)");

    // Output GEMM1/2/5 tiling: C[BT, BV], 2D warp tiling
    constexpr bool BT_LARGE = (BT >= 32);
    constexpr int O_T_M = BT_LARGE
        ? ((T::NUM_WARPS < BT / W) ? T::NUM_WARPS : BT / W) : 1;
    constexpr int O_T_N = T::NUM_WARPS / O_T_M;
    constexpr int O_E_M = BT / (W * O_T_M);
    constexpr int O_E_N = BV / (W * O_T_N);
    constexpr int O_E_K = BK_SUB / W;
    static_assert(O_E_N > 0, "Too many warps for BV dimension (output)");

    // QK^T GEMM4 tiling: C[BT, BT], separate 2D tiling (BT may differ from BV)
    constexpr int QKT_T_M = BT_LARGE
        ? ((T::NUM_WARPS < BT / W) ? T::NUM_WARPS : BT / W) : 1;
    constexpr int QKT_T_N = BT_LARGE ? (T::NUM_WARPS / QKT_T_M) : T::NUM_WARPS;
    constexpr int QKT_E_M = BT / (W * QKT_T_M);
    constexpr int QKT_E_N = BT / (W * QKT_T_N);
    constexpr int QKT_E_K = BK_SUB / W;

    // LDS strides (bf16 element count, including padding)
    constexpr int STRIDE_BK = BK_SUB + PAD;     // 68: for h^T, w_bar, q, k
    constexpr int STRIDE_BT = BT + PAD;         // 68 (BT=64) or 20 (BT=16): for k_T, v_T, A

    // Thread identity
    const int i_v  = blockIdx.x;
    const int i_nh = blockIdx.y;
    const int i_n  = i_nh / kargs.H;
    const int i_h  = i_nh % kargs.H;
    const int tid  = threadIdx.x;
    const int warp_id = tid / WS;
    const int lane_id = tid % WS;

    const int v_off = i_v * BV;
    const int bos   = i_n * kargs.T;

    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;
    const int NT = kargs.NT;

    const int stride_k = H * K;
    const int stride_v = H * V;
    const int stride_g = H;

    // =====================================================================
    // MFMA warp base offsets (2D tiling: warp_m tiles M, warp_n tiles N)
    // =====================================================================
    const int h_m_base = (warp_id / H_T_N) * W;
    const int h_n_base = (warp_id % H_T_N) * (H_E_N * W);

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

    // =====================================================================
    // Register-resident h state — MFMA accumulator layout
    //
    // h[bk][en][p] maps to h_matrix[row, col] where:
    //   row = h_m_base + (lane_id/16)*4 + p     ∈ [0, BK_SUB)
    //   col = en*16 + lane_id%16                 ∈ [0, BV)
    // =====================================================================
    v4f32_t h1[H_E_N];
    v4f32_t h2[H_E_N];
    clear_v4f32<H_E_N>(h1);
    if constexpr (N_K >= 2) clear_v4f32<H_E_N>(h2);

    if (kargs.ptr_h0 != nullptr) {
        const D_ACC* h0 = reinterpret_cast<const D_ACC*>(kargs.ptr_h0)
                          + (i_n * H + i_h) * V * K;
        for (int en = 0; en < H_E_N; en++) {
            for (int p = 0; p < 4; p++) {
                int row = h_m_base + (lane_id >> 4) * 4 + p;
                int col = h_n_base + en * W + (lane_id & 15);
                if (v_off + col < V) {
                    h1[en][p] = h0[(v_off + col) * K + row];
                    if constexpr (N_K >= 2)
                        h2[en][p] = h0[(v_off + col) * K + BK_SUB + row];
                }
            }
        }
    }

    // =====================================================================
    // Shared memory layout
    //
    // s_g[BT] fp32:              persistent gate cumsum
    // s_v_T[BV, STRIDE_BT] bf16: v_new transposed (persistent b'→e)
    // pool[...]:                  aliased per phase
    // =====================================================================
    extern __shared__ char smem_buf[];

    D_ACC*   s_g   = reinterpret_cast<D_ACC*>(smem_buf);
    D_ATTN*  s_v_T = reinterpret_cast<D_ATTN*>(smem_buf + T::smem_g_bytes);
    // Persistent q (all N_K subtiles) — written once in phase b/c, reused in
    // phase e to avoid a second HBM read of q. Layout: subtile bk at
    // s_q + bk*BT*STRIDE_BK, each [BT, STRIDE_BK].
    D_ATTN*  s_q   = reinterpret_cast<D_ATTN*>(smem_buf + T::smem_g_bytes + T::smem_vT_bytes);
    D_ATTN*  s_pool = reinterpret_cast<D_ATTN*>(
                         smem_buf + T::smem_g_bytes + T::smem_vT_bytes + T::smem_q_bytes);

    // =====================================================================
    // HBM base pointers
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
    D_ATTN* vn_hbm = kargs.ptr_v_new
        ? reinterpret_cast<D_ATTN*>(kargs.ptr_v_new) + (bos * H + i_h) * V
        : nullptr;

    // =====================================================================
    // Software pipelining: persistent prefetch buffers
    //
    // pf_w: w_bar → phase b/c, then reused for q → phase e
    // pf_q: q → phase b/c, then reused for k → phase e
    // pf_k: k → phase d
    // =====================================================================
    constexpr int PF_VEC   = 4;
    constexpr int PF_NVEC  = BK_SUB / PF_VEC;               // 16
    constexpr int PF_ELEMS = BT * PF_NVEC;                  // 1024 (BT=64), 256 (BT=16)
    constexpr int PF_LOADS = (PF_ELEMS + BS - 1) / BS;      // 4 (BT=64), 1 (BT=16)

    v4bf16_t pf_w[PF_LOADS];
    v4bf16_t pf_q[PF_LOADS];

    // Prologue: prefetch chunk 0, bk=0 w_bar and q only
    // (pf_k deferred to just before phase d to reduce peak VGPR pressure)
    {
        #pragma unroll
        for (int li = 0; li < PF_LOADS; li++) {
            int i = tid + li * BS;
            pf_w[li] = {};
            pf_q[li] = {};
            if (i < PF_ELEMS) {
                int row = i / PF_NVEC;
                int col = (i % PF_NVEC) * PF_VEC;
                if (row < kargs.T) {
                    pf_w[li] = *reinterpret_cast<const v4bf16_t*>(
                        &w_hbm[row * stride_k + col]);
                    pf_q[li] = *reinterpret_cast<const v4bf16_t*>(
                        &q_hbm[row * stride_k + col]);
                }
            }
        }
    }

    // =====================================================================
    // Chunk-serial main loop
    // =====================================================================
    for (int i_t = 0; i_t < NT; i_t++) {
        const int t0 = i_t * BT;

        // P2: chunk base pointers — hoist t0*stride out of inner loops
        const D_ATTN* q_ch = q_hbm + (int64_t)t0 * stride_k;
        const D_ATTN* k_ch = k_hbm + (int64_t)t0 * stride_k;
        const D_ATTN* w_ch = w_hbm + (int64_t)t0 * stride_k;
        const D_ATTN* u_ch = u_hbm + (int64_t)t0 * stride_v;
        D_ATTN*       o_ch = o_hbm + (int64_t)t0 * stride_v;
        D_ATTN*       vn_ch = vn_hbm ? vn_hbm + (int64_t)t0 * stride_v : nullptr;

        // P4: uniform boundary flags → SALU branch, skip per-lane v_cmp
        const int T_rem = kargs.T - t0;
        const bool full_chunk = (T_rem >= BT);
        const bool v_full = (v_off + BV <= V);

        // --- Load g_cumsum[BT] for this chunk ---
        for (int i = tid; i < BT; i += BS) {
            s_g[i] = (i < T_rem) ? g_hbm[(t0 + i) * stride_g] : 0.0f;
        }
        __syncthreads();

        int last_valid = (t0 + BT <= kargs.T) ? (BT - 1)
                                               : (kargs.T - t0 - 1);
        D_ACC g_last = s_g[last_valid];

        // =============================================================
        // (a) Store h_snapshot to HBM (pre-update state)
        // =============================================================
        if (kargs.ptr_h_snap != nullptr) {
            D_ACC* snap = reinterpret_cast<D_ACC*>(kargs.ptr_h_snap)
                + ((int64_t)i_n * NT * H + (int64_t)i_t * H + i_h) * V * K;
            for (int en = 0; en < H_E_N; en++) {
                for (int p = 0; p < 4; p++) {
                    int row = h_m_base + (lane_id >> 4) * 4 + p;
                    int col = h_n_base + en * W + (lane_id & 15);
                    if (v_off + col < V) {
                        snap[(v_off + col) * K + row] = h1[en][p];
                        if constexpr (N_K >= 2)
                            snap[(v_off + col) * K + BK_SUB + row] = h2[en][p];
                    }
                }
            }
        }

        // =============================================================
        // (b/c) Retrieve + Cross-chunk via MFMA
        //
        // retrieve[BT, BV] = Σ_bk  w_bar[BT, BK_SUB] × h[BK_SUB, BV]
        // o_cross [BT, BV] = Σ_bk  q    [BT, BK_SUB] × h[BK_SUB, BV]
        //
        // BT≥64: ds_permute path — A from registers (2 syncs/bk = 4 total)
        // BT<64: original LDS path — A and B from LDS (4 syncs/bk = 8 total)
        // =============================================================
        constexpr int C_ELEMS = O_E_M * O_E_N;
        v4f32_t r_retrieve[C_ELEMS];
        v4f32_t r_o_cross[C_ELEMS];
        clear_v4f32<C_ELEMS>(r_retrieve);
        clear_v4f32<C_ELEMS>(r_o_cross);

        D_ATTN* s_h_T = s_pool;                     // [BV, STRIDE_BK]

        for (int bk = 0; bk < N_K; bk++) {
            v4f32_t* h_cur = (bk == 0) ? h1 : h2;

            // Spill h[bk] → s_h_T transposed
            for (int en = 0; en < H_E_N; en++) {
                int row_T = h_n_base + en * W + (lane_id & 15);
                int col_T = h_m_base + (lane_id >> 4) * 4;
                v4bf16_t val;
                for (int p = 0; p < 4; p++)
                    val[p] = static_cast<D_ATTN>(h_cur[en][p]);
                *reinterpret_cast<v4bf16_t*>(&s_h_T[row_T * STRIDE_BK + col_T]) = val;
            }

            // Install prefetched w → s_sub_w (transient) and q → s_q[bk]
            // (persistent for the whole chunk). Separate buffers so retrieve and
            // cross GEMMs need no barrier between them (4→2 barriers/bk), and the
            // persistent q is reused in phase e instead of re-reading from HBM.
            D_ATTN* s_sub_w = s_pool + BV * STRIDE_BK;
            D_ATTN* s_q_bk  = s_q + bk * BT * STRIDE_BK;

            #pragma unroll
            for (int li = 0; li < PF_LOADS; li++) {
                int i = tid + li * BS;
                if (i < PF_ELEMS) {
                    int row = i / PF_NVEC;
                    int col = (i % PF_NVEC) * PF_VEC;
                    *reinterpret_cast<v4bf16_t*>(&s_sub_w[row * STRIDE_BK + col]) = pf_w[li];
                    *reinterpret_cast<v4bf16_t*>(&s_q_bk[row * STRIDE_BK + col]) = pf_q[li];
                }
            }
            __syncthreads();

            // Prefetch next bk's w/q from HBM (VMEM overlaps both GEMMs below)
            if (bk + 1 < N_K) {
                int k_off_next = (bk + 1) * BK_SUB;
                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    pf_w[li] = {};
                    pf_q[li] = {};
                    if (i < PF_ELEMS) {
                        int row = i / PF_NVEC;
                        int col = (i % PF_NVEC) * PF_VEC;
                        if (full_chunk || row < T_rem) {
                            pf_w[li] = *reinterpret_cast<const v4bf16_t*>(
                                &w_ch[row * stride_k + k_off_next + col]);
                            pf_q[li] = *reinterpret_cast<const v4bf16_t*>(
                                &q_ch[row * stride_k + k_off_next + col]);
                        }
                    }
                }
            }

            tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
                r_retrieve, s_sub_w, o_m_base, STRIDE_BK,
                            s_h_T,   o_n_base, STRIDE_BK, lane_id);
            tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
                r_o_cross,  s_q_bk,  o_m_base, STRIDE_BK,
                            s_h_T,   o_n_base, STRIDE_BK, lane_id);
            __syncthreads();
        }

        // Gate-scale o_cross: o_cross[s,:] *= exp(g_cumsum[s])
        for (int i = 0; i < C_ELEMS; i++) {
            int s;
            if constexpr (BT_LARGE)
                s = o_m_base + (i / O_E_N) * W + (lane_id >> 4) * 4;
            else
                s = (lane_id >> 4) * 4;
            for (int p = 0; p < 4; p++) {
                int sp = s + p;
                int row;
                if constexpr (BT_LARGE)
                    row = sp;
                else
                    row = (i / O_E_N) * W + sp;
                r_o_cross[i][p] *= fast_exp(s_g[row]);
            }
        }

        // =============================================================
        // (b') v_new = u_bar - retrieve
        // Store transposed to s_v_T[BV, STRIDE_BT] as bf16.
        // Also write bf16 to HBM (vn_hbm).
        //
        // Opt-5: pack 4 consecutive s positions into v4bf16_t for
        // vectorized LDS write (s_base is always 4-aligned → 8B aligned)
        // =============================================================
        for (int i = 0; i < C_ELEMS; i++) {
            int en = i % O_E_N;
            int s_base, c;
            if constexpr (BT_LARGE) {
                s_base = o_m_base + (i / O_E_N) * W + (lane_id >> 4) * 4;
                c = o_n_base + en * W + (lane_id & 15);
            } else {
                s_base = (lane_id >> 4) * 4;
                c = o_n_base + en * W + (lane_id & 15);
            }
            v4bf16_t v_new_pack;
            for (int p = 0; p < 4; p++) {
                int s = s_base + p;
                D_ACC u_val = 0.0f;
                if ((full_chunk || s < T_rem) && (v_full || v_off + c < V))
                    u_val = static_cast<D_ACC>(
                        u_ch[s * stride_v + v_off + c]);
                D_ACC v_new_val = u_val - r_retrieve[i][p];
                D_ATTN v_new_bf16 = static_cast<D_ATTN>(v_new_val);
                v_new_pack[p] = v_new_bf16;

                if (vn_ch && (full_chunk || s < T_rem) && (v_full || v_off + c < V))
                    vn_ch[s * stride_v + v_off + c] = v_new_bf16;
            }
            *reinterpret_cast<v4bf16_t*>(&s_v_T[c * STRIDE_BT + s_base]) = v_new_pack;
        }

        // Prefetch k[bk=0] → pf_q for phase e (QK^T). q is already resident in
        // the persistent s_q buffer from phase b/c, so it is NOT re-read here.
        // (VMEM loads overlap with phase d execution, ~1000+ cycles)
        #pragma unroll
        for (int li = 0; li < PF_LOADS; li++) {
            int i = tid + li * BS;
            pf_q[li] = {};
            if (i < PF_ELEMS) {
                int row = i / PF_NVEC;
                int col = (i % PF_NVEC) * PF_VEC;
                if (full_chunk || row < T_rem)
                    pf_q[li] = *reinterpret_cast<const v4bf16_t*>(
                        &k_ch[row * stride_k + col]);
            }
        }

        // Deferred pf_k: load k[bk=0] for phase d here instead of cross-chunk
        // to reduce peak VGPR pressure during phases b/c (saves 8 VGPRs)
        v4bf16_t pf_k[PF_LOADS];
        #pragma unroll
        for (int li = 0; li < PF_LOADS; li++) {
            int i = tid + li * BS;
            pf_k[li] = {};
            if (i < PF_ELEMS) {
                int row = i / PF_NVEC;
                int col = (i % PF_NVEC) * PF_VEC;
                if (full_chunk || row < T_rem)
                    pf_k[li] = *reinterpret_cast<const v4bf16_t*>(
                        &k_ch[row * stride_k + col]);
            }
        }
        // barrier removed: s_v_T visibility guaranteed by barrier at pf_k install
        // (all threads must pass v_new writes before reaching that sync)

        // =============================================================
        // (d) Decay h + Accumulate h += k_gated^T × v_new  (GEMM3)
        //
        // k_gated[s,j] = k[s,j] * exp(g_last - g[s])
        // Stored transposed: s_k_T[BK_SUB, STRIDE_BT] in pool
        // v_new^T already in s_v_T[BV, STRIDE_BT]
        // =============================================================
        D_ACC decay = fast_exp(g_last);
        for (int en = 0; en < H_E_N; en++)
            for (int p = 0; p < 4; p++) {
                h1[en][p] *= decay;
                if constexpr (N_K >= 2) h2[en][p] *= decay;
            }

        D_ATTN* s_k_T = s_pool;  // pool reused (phase b/c data no longer needed)

        for (int bk = 0; bk < N_K; bk++) {
            v4f32_t* h_cur = (bk == 0) ? h1 : h2;

            // Install pf_k + gate → s_k_T transposed (register→LDS, no HBM wait)
            #pragma unroll
            for (int li = 0; li < PF_LOADS; li++) {
                int i = tid + li * BS;
                if (i < PF_ELEMS) {
                    int s = i / PF_NVEC;
                    int j = (i % PF_NVEC) * PF_VEC;
                    D_ACC gate = (full_chunk || s < T_rem)
                        ? fast_exp(g_last - s_g[s]) : 0.0f;
                    for (int vi = 0; vi < PF_VEC; vi++)
                        s_k_T[(j + vi) * STRIDE_BT + s] = static_cast<D_ATTN>(
                            static_cast<D_ACC>(pf_k[li][vi]) * gate);
                }
            }
            __syncthreads();

            // Async prefetch k[bk+1] → pf_k (VMEM overlaps with GEMM3)
            if (bk + 1 < N_K) {
                int k_off_next = (bk + 1) * BK_SUB;
                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    pf_k[li] = {};
                    if (i < PF_ELEMS) {
                        int s = i / PF_NVEC;
                        int j = (i % PF_NVEC) * PF_VEC;
                        if (full_chunk || s < T_rem)
                            pf_k[li] = *reinterpret_cast<const v4bf16_t*>(
                                &k_ch[s * stride_k + k_off_next + j]);
                    }
                }
            }

            // GEMM3: h[bk] += k_gated^T(A) × v_new^T(B)
            tiled_gemm_mfma<H_E_M, H_E_N, H_E_K>(
                h_cur,
                s_k_T, h_m_base, STRIDE_BT,
                s_v_T, h_n_base, STRIDE_BT,
                lane_id);
            __syncthreads();
        }

        // =============================================================
        // (e) Intra-chunk causal attention
        //
        // GEMM4 (QK^T): A_intra[BT,BT] = Σ_bk q[BT,BK_SUB] × k[BT,BK_SUB]^T
        // Gate + causal mask on A_intra
        // GEMM5 (AV):   o_intra[BT,BV] = A_intra[BT,BT] × v_new[BT,BV]
        // =============================================================

        {
            // ---- GEMM4: QK^T via MFMA ----
            // q is read from the persistent s_q (subtile bk at s_q+bk*BT*STRIDE_BK);
            // only k needs an LDS install buffer here.
            D_ATTN* s_k4 = s_pool;                    // [BT, STRIDE_BK]
            D_ATTN* s_A5 = s_pool;                    // [BT, STRIDE_BT] reuses s_k4

            if constexpr (BT >= 32) {
                // Multi-tile MFMA: warps tile the [BT, BT] output
                v4f32_t r_A[QKT_E_M * QKT_E_N];
                clear_v4f32<QKT_E_M * QKT_E_N>(r_A);

                for (int bk = 0; bk < N_K; bk++) {
                    // Install pf_q → s_k4 (register→LDS); q already in s_q.
                    #pragma unroll
                    for (int li = 0; li < PF_LOADS; li++) {
                        int i = tid + li * BS;
                        if (i < PF_ELEMS) {
                            int row = i / PF_NVEC;
                            int col = (i % PF_NVEC) * PF_VEC;
                            *reinterpret_cast<v4bf16_t*>(&s_k4[row * STRIDE_BK + col]) = pf_q[li];
                        }
                    }
                    __syncthreads();

                    // Async prefetch k[bk+1] → pf_q (VMEM overlaps with GEMM4)
                    if (bk + 1 < N_K) {
                        int k_off_next = (bk + 1) * BK_SUB;
                        #pragma unroll
                        for (int li = 0; li < PF_LOADS; li++) {
                            int i = tid + li * BS;
                            pf_q[li] = {};
                            if (i < PF_ELEMS) {
                                int row = i / PF_NVEC;
                                int col = (i % PF_NVEC) * PF_VEC;
                                if (full_chunk || row < T_rem)
                                    pf_q[li] = *reinterpret_cast<const v4bf16_t*>(
                                        &k_ch[row * stride_k + k_off_next + col]);
                            }
                        }
                    }

                    tiled_gemm_mfma<QKT_E_M, QKT_E_N, QKT_E_K>(
                        r_A,
                        s_q + bk * BT * STRIDE_BK, qkt_m_base, STRIDE_BK,
                        s_k4,                      qkt_n_base, STRIDE_BK,
                        lane_id);
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
                    for (int p = 0; p < 4; p++) {
                        int row = row_base + p;
                        s_A5[row * STRIDE_BT + col_base] = static_cast<D_ATTN>(r_A[i][p]);
                    }
                }

            } else {
                // BT<32: output [BT,BT] = 1 MFMA tile, warp 0 computes
                v4f32_t r_A[1];
                clear_v4f32<1>(r_A);

                for (int bk = 0; bk < N_K; bk++) {
                    // Install pf_q → s_k4 (register→LDS); q already in s_q.
                    #pragma unroll
                    for (int li = 0; li < PF_LOADS; li++) {
                        int i = tid + li * BS;
                        if (i < PF_ELEMS) {
                            int row = i / PF_NVEC;
                            int col = (i % PF_NVEC) * PF_VEC;
                            *reinterpret_cast<v4bf16_t*>(&s_k4[row * STRIDE_BK + col]) = pf_q[li];
                        }
                    }
                    __syncthreads();

                    // Async prefetch k[bk+1] → pf_q
                    if (bk + 1 < N_K) {
                        int k_off_next = (bk + 1) * BK_SUB;
                        #pragma unroll
                        for (int li = 0; li < PF_LOADS; li++) {
                            int i = tid + li * BS;
                            pf_q[li] = {};
                            if (i < PF_ELEMS) {
                                int row = i / PF_NVEC;
                                int col = (i % PF_NVEC) * PF_VEC;
                                if (full_chunk || row < T_rem)
                                    pf_q[li] = *reinterpret_cast<const v4bf16_t*>(
                                        &k_ch[row * stride_k + k_off_next + col]);
                            }
                        }
                    }

                    if (warp_id == 0) {
                        constexpr int QKT_EK_SM = BK_SUB / W;
                        tiled_gemm_mfma<1, 1, QKT_EK_SM>(
                            r_A, s_q + bk * BT * STRIDE_BK, 0, STRIDE_BK,
                            s_k4, 0, STRIDE_BK, lane_id);
                    }
                    __syncthreads();
                }

                // Gate + causal mask + store (warp 0 only)
                if (warp_id == 0) {
                    for (int p = 0; p < 4; p++) {
                        int s = (lane_id >> 4) * 4 + p;
                        int r = lane_id & 15;
                        if (s >= r && (full_chunk || (s < T_rem && r < T_rem)))
                            r_A[0][p] *= fast_exp(s_g[s] - s_g[r]);
                        else
                            r_A[0][p] = 0.0f;
                        s_A5[s * STRIDE_BT + r] = static_cast<D_ATTN>(r_A[0][p]);
                    }
                }
            }
            __syncthreads();

            // Cross-chunk prefetch: load next chunk's bk=0 w_bar and q
            // (pf_k deferred to between b' and d to reduce peak VGPR pressure)
            // (VMEM overlaps with GEMM5 + output store, ~556 cycles)
            if (i_t + 1 < NT) {
                int next_t0 = (i_t + 1) * BT;
                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    pf_w[li] = {};
                    pf_q[li] = {};
                    if (i < PF_ELEMS) {
                        int row = i / PF_NVEC;
                        int col = (i % PF_NVEC) * PF_VEC;
                        if (next_t0 + row < kargs.T) {
                            pf_w[li] = *reinterpret_cast<const v4bf16_t*>(
                                &w_hbm[(int64_t)(next_t0 + row) * stride_k + col]);
                            pf_q[li] = *reinterpret_cast<const v4bf16_t*>(
                                &q_hbm[(int64_t)(next_t0 + row) * stride_k + col]);
                        }
                    }
                }
            }

            // ---- GEMM5: AV via MFMA (unified for all BT) ----
            constexpr int AV_E_K = BT / W;
            v4f32_t r_o_intra[C_ELEMS];
            clear_v4f32<C_ELEMS>(r_o_intra);

            tiled_gemm_mfma<O_E_M, O_E_N, AV_E_K>(
                r_o_intra,
                s_A5, o_m_base, STRIDE_BT,
                s_v_T, o_n_base, STRIDE_BT,
                lane_id);

            // Combine: o = scale * (o_cross + o_intra), store to HBM
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
                        o_ch[s * stride_v + v_off + c]
                            = static_cast<D_ATTN>(o_val);
                }
            }
            // end-of-chunk barrier removed: next chunk's barrier 276 (g_cumsum sync)
            // guarantees all threads finish GEMM5 pool reads before s_pool reuse
        }

    }  // end chunk loop

    // =====================================================================
    // Epilogue: store final h → ptr_ht[N, H, V, K]
    // =====================================================================
    if (kargs.ptr_ht != nullptr) {
        D_ACC* ht = reinterpret_cast<D_ACC*>(kargs.ptr_ht)
                    + (i_n * H + i_h) * V * K;
        for (int en = 0; en < H_E_N; en++) {
            for (int p = 0; p < 4; p++) {
                int row = h_m_base + (lane_id >> 4) * 4 + p;
                int col = h_n_base + en * W + (lane_id & 15);
                if (v_off + col < V) {
                    ht[(v_off + col) * K + row] = h1[en][p];
                    if constexpr (N_K >= 2)
                        ht[(v_off + col) * K + BK_SUB + row] = h2[en][p];
                }
            }
        }
    }
}
