// GDN Prefill K2 Kernel — MFMA bf16 16×16×16 optimized
// Step 3: Hidden state update (retrieve, gate-scale, decay, accumulate)
// Step 4: Output (cross-chunk QH, intra-chunk causal attention)
//
// Grid: (cdiv(V, BV), N*H)   Block: (BLOCK_SIZE = 256)
// h state: register-resident in MFMA accumulator layout, N_K × [BK_SUB, BV]
// Chunks: serial iteration within each workgroup
//
// 5 GEMMs via MFMA bf16 16×16×16 (BT≥32), scalar fallback for BT<32 GEMM4/5
// Target: gfx942 (MI300X)
#pragma once

#include "opus_gdn/gdn_defs.h"

namespace gdn_k2_mfma {

// MFMA register vector types (gfx942 bf16 16×16×16)
using v4bf16_t = __bf16 __attribute__((ext_vector_type(4)));
using v4f32_t  = float  __attribute__((ext_vector_type(4)));

// exp(x) via single-cycle v_exp_f32: exp(x) = 2^(x * log2(e))
__device__ inline float fast_exp(float x) {
    return __builtin_amdgcn_exp2f(x * 1.442695041f);
}

__device__ inline v4f32_t mfma_f32_16x16x16_bf16(v4bf16_t a, v4bf16_t b, v4f32_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

// Load one 16×16 MFMA tile from LDS
// LDS layout: [rows, stride] bf16, row-major
// A tile:  lane holds A[row_base + lane%16, col_base + (lane/16)*4 .. +3]
// B tile:  lane holds B[row_base + lane%16, col_base + (lane/16)*4 .. +3]
// (Same register packing for A and B; hardware interprets differently)
__device__ inline v4bf16_t load_mfma_tile(
        const __bf16* __restrict__ lds, int row_base, int col_base,
        int stride, int lane_id) {
    int addr = (row_base + (lane_id & 15)) * stride + col_base + ((lane_id >> 4) << 2);
    return *reinterpret_cast<const v4bf16_t*>(&lds[addr]);
}

// Tiled MFMA GEMM
// Computes C[E_M*16, E_N*16] += A[E_M*16, E_K*16] × B^T[E_N*16, E_K*16]
//   A stored as [M, stride_a] bf16 (M×K row-major)
//   B stored as [N, stride_b] bf16 (N×K row-major, opus convention)
//   Hardware: D = A_reg × B_reg^T, giving C[M,N]
template<int E_M, int E_N, int E_K>
__device__ void tiled_gemm_mfma(
        v4f32_t* __restrict__ c,
        const __bf16* __restrict__ lds_a, int m_base, int stride_a,
        const __bf16* __restrict__ lds_b, int n_base, int stride_b,
        int lane_id) {
    for (int ek = 0; ek < E_K; ek++) {
        v4bf16_t a_tiles[E_M];
        for (int em = 0; em < E_M; em++)
            a_tiles[em] = load_mfma_tile(lds_a, m_base + em * 16, ek * 16, stride_a, lane_id);
        v4bf16_t b_tiles[E_N];
        for (int en = 0; en < E_N; en++)
            b_tiles[en] = load_mfma_tile(lds_b, n_base + en * 16, ek * 16, stride_b, lane_id);
        for (int em = 0; em < E_M; em++)
            for (int en = 0; en < E_N; en++)
                c[em * E_N + en] = mfma_f32_16x16x16_bf16(
                    a_tiles[em], b_tiles[en], c[em * E_N + en]);
    }
}

// Zero a v4f32_t array
template<int N>
__device__ inline void clear_v4f32(v4f32_t* c) {
    for (int i = 0; i < N; i++) c[i] = v4f32_t{0.f, 0.f, 0.f, 0.f};
}

} // namespace gdn_k2_mfma

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, Traits::OCC_HINT)
gdn_k2_kernel(gdn_k2_kargs kargs) {
    using namespace gdn_k2_mfma;
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
    constexpr bool BT_LARGE = (BT >= 64);
    constexpr int O_T_M = BT_LARGE
        ? ((T::NUM_WARPS < BT / W) ? T::NUM_WARPS : BT / W) : 1;
    constexpr int O_T_N = T::NUM_WARPS / O_T_M;
    constexpr int O_E_M = BT / (W * O_T_M);
    constexpr int O_E_N = BV / (W * O_T_N);
    constexpr int O_E_K = BK_SUB / W;
    static_assert(O_E_N > 0, "Too many warps for BV dimension (output)");

    // QK^T GEMM4 tiling: C[BT, BT], 2D warp tiling
    constexpr int QKT_E_M = BT / (W * O_T_M);
    constexpr int QKT_E_N = BT / (W * O_T_N);
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
        o_m_base = (warp_id / O_T_N) * W;
        o_n_base = (warp_id % O_T_N) * (O_E_N * W);
    } else {
        o_m_base = 0;
        o_n_base = warp_id * W;
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
    D_ATTN*  s_pool = reinterpret_cast<D_ATTN*>(smem_buf + T::smem_g_bytes + T::smem_vT_bytes);

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
    v4bf16_t pf_k[PF_LOADS];

    // Prologue: prefetch chunk 0, bk=0 w_bar, q, and k
    {
        #pragma unroll
        for (int li = 0; li < PF_LOADS; li++) {
            int i = tid + li * BS;
            pf_w[li] = {};
            pf_q[li] = {};
            pf_k[li] = {};
            if (i < PF_ELEMS) {
                int row = i / PF_NVEC;
                int col = (i % PF_NVEC) * PF_VEC;
                if (row < kargs.T) {
                    pf_w[li] = *reinterpret_cast<const v4bf16_t*>(
                        &w_hbm[row * stride_k + col]);
                    pf_q[li] = *reinterpret_cast<const v4bf16_t*>(
                        &q_hbm[row * stride_k + col]);
                    pf_k[li] = *reinterpret_cast<const v4bf16_t*>(
                        &k_hbm[row * stride_k + col]);
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

            {
                // ---- Install prefetched registers → LDS, then standard GEMM ----
                D_ATTN* s_sub = s_pool + BV * STRIDE_BK;

                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    if (i < PF_ELEMS) {
                        int row = i / PF_NVEC;
                        int col = (i % PF_NVEC) * PF_VEC;
                        *reinterpret_cast<v4bf16_t*>(&s_sub[row * STRIDE_BK + col]) = pf_w[li];
                    }
                }
                __syncthreads();

                if (bk + 1 < N_K) {
                    int k_off_next = (bk + 1) * BK_SUB;
                    #pragma unroll
                    for (int li = 0; li < PF_LOADS; li++) {
                        int i = tid + li * BS;
                        pf_w[li] = {};
                        if (i < PF_ELEMS) {
                            int row = i / PF_NVEC;
                            int col = (i % PF_NVEC) * PF_VEC;
                            if (full_chunk || row < T_rem)
                                pf_w[li] = *reinterpret_cast<const v4bf16_t*>(
                                    &w_ch[row * stride_k + k_off_next + col]);
                        }
                    }
                }

                tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
                    r_retrieve,
                    s_sub, o_m_base, STRIDE_BK,
                    s_h_T, o_n_base, STRIDE_BK,
                    lane_id);
                __syncthreads();

                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    if (i < PF_ELEMS) {
                        int row = i / PF_NVEC;
                        int col = (i % PF_NVEC) * PF_VEC;
                        *reinterpret_cast<v4bf16_t*>(&s_sub[row * STRIDE_BK + col]) = pf_q[li];
                    }
                }

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
                                    &q_ch[row * stride_k + k_off_next + col]);
                        }
                    }
                }
                __syncthreads();

                tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
                    r_o_cross,
                    s_sub, o_m_base, STRIDE_BK,
                    s_h_T, o_n_base, STRIDE_BK,
                    lane_id);
                __syncthreads();
            }
        }

        // Gate-scale o_cross: o_cross[s,:] *= exp(g_cumsum[s])
        for (int i = 0; i < C_ELEMS; i++) {
            int s;
            if constexpr (BT_LARGE)
                s = o_m_base + (lane_id >> 4) * 4;
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
        // =============================================================
        for (int i = 0; i < C_ELEMS; i++) {
            int en = i % O_E_N;
            for (int p = 0; p < 4; p++) {
                int s, c;
                if constexpr (BT_LARGE) {
                    s = o_m_base + (lane_id >> 4) * 4 + p;
                    c = o_n_base + en * W + (lane_id & 15);
                } else {
                    s = (lane_id >> 4) * 4 + p;
                    c = o_n_base + en * W + (lane_id & 15);
                }
                D_ACC u_val = 0.0f;
                if ((full_chunk || s < T_rem) && (v_full || v_off + c < V))
                    u_val = static_cast<D_ACC>(
                        u_ch[s * stride_v + v_off + c]);
                D_ACC v_new_val = u_val - r_retrieve[i][p];
                D_ATTN v_new_bf16 = static_cast<D_ATTN>(v_new_val);

                // Store transposed: s_v_T[c, s]
                s_v_T[c * STRIDE_BT + s] = v_new_bf16;

                // Store to HBM (skipped when ptr_v_new is null)
                if (vn_ch && (full_chunk || s < T_rem) && (v_full || v_off + c < V))
                    vn_ch[s * stride_v + v_off + c] = v_new_bf16;
            }
        }

        // Prefetch q[bk=0] → pf_w, k[bk=0] → pf_q for phase e
        // (VMEM loads overlap with phase d execution, ~1000+ cycles)
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
                        &q_ch[row * stride_k + col]);
                    pf_q[li] = *reinterpret_cast<const v4bf16_t*>(
                        &k_ch[row * stride_k + col]);
                }
            }
        }
        __syncthreads();

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
            D_ATTN* s_q4 = s_pool;                    // [BT, STRIDE_BK]
            D_ATTN* s_k4 = s_pool + BT * STRIDE_BK;   // [BT, STRIDE_BK]
            D_ATTN* s_A5 = s_pool;                    // [BT, STRIDE_BT] reuses s_q4

            if constexpr (BT >= 32) {
                // Multi-tile MFMA: warps tile the [BT, BT] output
                v4f32_t r_A[QKT_E_M * QKT_E_N];
                clear_v4f32<QKT_E_M * QKT_E_N>(r_A);

                for (int bk = 0; bk < N_K; bk++) {
                    // Install pf_w → s_q4, pf_q → s_k4 (register→LDS, no HBM wait)
                    #pragma unroll
                    for (int li = 0; li < PF_LOADS; li++) {
                        int i = tid + li * BS;
                        if (i < PF_ELEMS) {
                            int row = i / PF_NVEC;
                            int col = (i % PF_NVEC) * PF_VEC;
                            *reinterpret_cast<v4bf16_t*>(&s_q4[row * STRIDE_BK + col]) = pf_w[li];
                            *reinterpret_cast<v4bf16_t*>(&s_k4[row * STRIDE_BK + col]) = pf_q[li];
                        }
                    }
                    __syncthreads();

                    // Async prefetch q[bk+1]/k[bk+1] (VMEM overlaps with GEMM4)
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
                                        &q_ch[row * stride_k + k_off_next + col]);
                                    pf_q[li] = *reinterpret_cast<const v4bf16_t*>(
                                        &k_ch[row * stride_k + k_off_next + col]);
                                }
                            }
                        }
                    }

                    tiled_gemm_mfma<QKT_E_M, QKT_E_N, QKT_E_K>(
                        r_A,
                        s_q4, o_m_base, STRIDE_BK,
                        s_k4, o_n_base, STRIDE_BK,
                        lane_id);
                    __syncthreads();
                }

                // Gate + causal mask
                for (int i = 0; i < QKT_E_M * QKT_E_N; i++) {
                    int en = i % QKT_E_N;
                    for (int p = 0; p < 4; p++) {
                        int s, r;
                        if constexpr (BT_LARGE) {
                            s = o_m_base + (lane_id >> 4) * 4 + p;
                            r = o_n_base + en * W + (lane_id & 15);
                        } else {
                            s = (lane_id >> 4) * 4 + p;
                            r = o_n_base + en * W + (lane_id & 15);
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
                        row_base = o_m_base + (lane_id >> 4) * 4;
                        col_base = o_n_base + en * W + (lane_id & 15);
                    } else {
                        row_base = (lane_id >> 4) * 4;
                        col_base = o_n_base + en * W + (lane_id & 15);
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
                    // Install pf_w → s_q4, pf_q → s_k4 (register→LDS)
                    #pragma unroll
                    for (int li = 0; li < PF_LOADS; li++) {
                        int i = tid + li * BS;
                        if (i < PF_ELEMS) {
                            int row = i / PF_NVEC;
                            int col = (i % PF_NVEC) * PF_VEC;
                            *reinterpret_cast<v4bf16_t*>(&s_q4[row * STRIDE_BK + col]) = pf_w[li];
                            *reinterpret_cast<v4bf16_t*>(&s_k4[row * STRIDE_BK + col]) = pf_q[li];
                        }
                    }
                    __syncthreads();

                    // Async prefetch q[bk+1]/k[bk+1]
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
                                        &q_ch[row * stride_k + k_off_next + col]);
                                    pf_q[li] = *reinterpret_cast<const v4bf16_t*>(
                                        &k_ch[row * stride_k + k_off_next + col]);
                                }
                            }
                        }
                    }

                    if (warp_id == 0) {
                        constexpr int QKT_EK_SM = BK_SUB / W;
                        tiled_gemm_mfma<1, 1, QKT_EK_SM>(
                            r_A, s_q4, 0, STRIDE_BK,
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

            // Cross-chunk prefetch: load next chunk's bk=0 data
            // (VMEM overlaps with GEMM5 + output store, ~556 cycles)
            if (i_t + 1 < NT) {
                int next_t0 = (i_t + 1) * BT;
                // Cooperative prefetch: pf_w, pf_q for next chunk
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
                // Cooperative: pf_k for phase d (always)
                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    pf_k[li] = {};
                    if (i < PF_ELEMS) {
                        int row = i / PF_NVEC;
                        int col = (i % PF_NVEC) * PF_VEC;
                        if (next_t0 + row < kargs.T)
                            pf_k[li] = *reinterpret_cast<const v4bf16_t*>(
                                &k_hbm[(int64_t)(next_t0 + row) * stride_k + col]);
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
                        s = o_m_base + (lane_id >> 4) * 4 + p;
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
            __syncthreads();
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
