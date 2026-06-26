// GDN Wavefront H-State Scan Kernel (with optional fused output)
// Scan-only or scan+output with inter-workgroup wavefront pipelining.
// Grid: (ceil(V/BV) * B * H,  N_super)
//   x-dim: independent (V-slice × batch × head)
//   y-dim: dependency chain (super-chunk index)
// Each workgroup scans S chunks, passing h_end to successor via atomic flags.
// Single kernel launch, single data pass — same HBM traffic as serial.
//
// Fused output mode (ptr_q, ptr_o non-null):
//   Computes o = scale * (Q × h × exp(g) + lower_tri(Q × K^T × exp(g diff)) × v_new)
//   Eliminates separate output kernel + h/v_new intermediate stores.
//
// Target: gfx942 (MI300X) / gfx950 (MI350)
#pragma once

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_mfma_utils.h"

static __device__ __forceinline__
gdn_mfma::v4f32_t buf_load_b128(__amdgpu_buffer_rsrc_t rsrc, int voff, int soff) {
    return __builtin_amdgcn_raw_buffer_load_b128(rsrc, voff, soff, 0);
}

static __device__ __forceinline__
void buf_store_b128(__amdgpu_buffer_rsrc_t rsrc, gdn_mfma::v4f32_t val, int voff, int soff) {
    __builtin_amdgcn_raw_buffer_store_b128(val, rsrc, voff, soff, 0);
}

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, 2)
gdn_wf_h_kernel(gdn_wf_h_kargs kargs) {
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

    constexpr int W = 16;

    constexpr int H_T_M = 1;
    constexpr int H_T_N = 4;
    constexpr int H_E_M = BK_SUB / (W * H_T_M);
    constexpr int H_E_N = BV / (W * H_T_N);
    constexpr int H_E_K = BT / W;

    constexpr int O_T_M = 1;
    constexpr int O_T_N = 4;
    constexpr int O_E_M = BT / (W * O_T_M);
    constexpr int O_E_N = BV / (W * O_T_N);
    constexpr int O_E_K = BK_SUB / W;

    constexpr int C_ELEMS = H_E_M * H_E_N;

    constexpr int STRIDE_BK = BK_SUB + PAD;
    constexpr int STRIDE_BT = BT + PAD;

    // QK^T tiling: T_M=1, T_N=4 → A[BT, BT], each warp covers A[:, warp*16..+15]
    constexpr int A_E_M = BT / W;
    constexpr int A_E_N = 1;  // per warp

    const int i_flat  = blockIdx.x;
    const int i_super = blockIdx.y;
    const int N_BH    = kargs.B * kargs.H;
    const int i_v     = i_flat / N_BH;
    const int i_nh    = i_flat % N_BH;
    const int i_n     = i_nh / kargs.H;
    const int i_h     = i_nh % kargs.H;
    const int tid     = threadIdx.x;
    const int warp_id = tid / WS;
    const int lane_id = tid % WS;

    const int v_off = i_v * BV;
    const int bos   = i_n * kargs.T;

    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;
    const int S  = kargs.S;

    const int stride_k = H * K;
    const int stride_v = H * V;
    const int stride_g = H;

    const int lane_col      = lane_id & 15;
    const int lane_row_base = (lane_id >> 4) << 2;
    const int warp_n_base = warp_id * W;

    #define ACC_ROW(n) (lane_row_base + (n))

    // Buffer resource descriptors
    const int hbm_k_off = (bos * H + i_h) * K;
    const int hbm_v_off = (bos * H + i_h) * V;

    auto w_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)((const D_ATTN*)kargs.ptr_w_bar + hbm_k_off),
        0, 0x7FFFFFFF, 0x00027FAC);
    auto k_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)((const D_ATTN*)kargs.ptr_k + hbm_k_off),
        0, 0x7FFFFFFF, 0x00027FAC);
    auto u_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)((const D_ATTN*)kargs.ptr_u_bar + hbm_v_off),
        0, 0x7FFFFFFF, 0x00027FAC);

    const bool do_output = (kargs.ptr_q != nullptr && kargs.ptr_o != nullptr);

    __amdgpu_buffer_rsrc_t q_rsrc;
    if (do_output) {
        q_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)((const D_ATTN*)kargs.ptr_q + hbm_k_off),
            0, 0x7FFFFFFF, 0x00027FAC);
    }

    D_ATTN* o_hbm = do_output ?
        reinterpret_cast<D_ATTN*>(kargs.ptr_o) + hbm_v_off : nullptr;

    // Register-resident h state
    v4f32_t h1[C_ELEMS];
    v4f32_t h2[C_ELEMS];
    clear_v4f32<C_ELEMS>(h1);
    if constexpr (N_K >= 2) clear_v4f32<C_ELEMS>(h2);

    // Wavefront sync: load h_init
    if (i_super == 0 && kargs.ptr_h0 != nullptr) {
        const D_ACC* h0 = reinterpret_cast<const D_ACC*>(kargs.ptr_h0)
                          + (i_n * H + i_h) * V * K;
        int col = warp_n_base + lane_col;
        for (int em = 0; em < C_ELEMS; em++) {
            for (int n = 0; n < 4; n++) {
                int row = em * W + ACC_ROW(n);
                h1[em][n] = h0[(v_off + col) * K + row];
            }
        }
        if constexpr (N_K >= 2) {
            for (int em = 0; em < C_ELEMS; em++) {
                for (int n = 0; n < 4; n++) {
                    int row = em * W + ACC_ROW(n);
                    h2[em][n] = h0[(v_off + col) * K + BK_SUB + row];
                }
            }
        }
    } else if (i_super > 0) {
        const uint32_t flag_idx = i_flat * kargs.N_super + (i_super - 1);
        if (tid == 0) {
            while (__atomic_load_n(kargs.ptr_flags + flag_idx, __ATOMIC_RELAXED) == 0) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

        const int hp_stride = kargs.N_super * N_K * BV * BK_SUB;
        const D_ACC* hp_base = reinterpret_cast<const D_ACC*>(kargs.ptr_h_pass)
                               + i_flat * hp_stride + (i_super - 1) * N_K * BV * BK_SUB;
        int col = warp_n_base + lane_col;
        for (int em = 0; em < C_ELEMS; em++) {
            for (int n = 0; n < 4; n++) {
                int row = em * W + ACC_ROW(n);
                h1[em][n] = hp_base[col * BK_SUB + row];
            }
        }
        if constexpr (N_K >= 2) {
            for (int em = 0; em < C_ELEMS; em++) {
                for (int n = 0; n < 4; n++) {
                    int row = em * W + ACC_ROW(n);
                    h2[em][n] = hp_base[BV * BK_SUB + col * BK_SUB + row];
                }
            }
        }
    }

    // Shared memory layout
    constexpr int smem_g_bytes  = BT * (int)sizeof(D_ACC);
    constexpr int smem_vT_bytes = BV * STRIDE_BT * (int)sizeof(D_ATTN);

    extern __shared__ char smem_buf[];
    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);
    D_ATTN* s_v_T  = reinterpret_cast<D_ATTN*>(smem_buf + smem_g_bytes);
    D_ATTN* s_pool = reinterpret_cast<D_ATTN*>(smem_buf + smem_g_bytes + smem_vT_bytes);

    const D_ACC* g_hbm = reinterpret_cast<const D_ACC*>(kargs.ptr_g_cumsum)
                          + bos * H + i_h;

    // Software pipelining constants
    constexpr int PF_VEC   = 8;
    constexpr int PF_NVEC  = BK_SUB / PF_VEC;
    constexpr int PF_ELEMS = BT * PF_NVEC;
    constexpr int PF_LOADS = (PF_ELEMS + BS - 1) / BS;

    v4f32_t pf_w[PF_LOADS];

    constexpr int UPF_VEC   = 8;
    constexpr int UPF_NVEC  = BV / UPF_VEC;
    constexpr int UPF_ELEMS = BT * UPF_NVEC;
    constexpr int UPF_LOADS = (UPF_ELEMS + BS - 1) / BS;

    // Prologue: prefetch first chunk's w_bar[bk=0]
    {
        const int t0 = i_super * S * BT;
        #pragma unroll
        for (int li = 0; li < PF_LOADS; li++) {
            int i = tid + li * BS;
            if (i < PF_ELEMS) {
                int row = i / PF_NVEC;
                int col = (i % PF_NVEC) * PF_VEC;
                int voff = ((t0 + row) * stride_k + col) * (int)sizeof(D_ATTN);
                pf_w[li] = buf_load_b128(w_rsrc, voff, 0);
            } else {
                pf_w[li] = {};
            }
        }
    }

    // =====================================================================
    // Main loop: S chunks
    // =====================================================================
    for (int i_s = 0; i_s < S; i_s++) {
        const int i_t = i_super * S + i_s;
        const int t0  = i_t * BT;

        // Load g_cumsum
        for (int i = tid; i < BT; i += BS) {
            s_g[i] = g_hbm[(t0 + i) * stride_g];
        }

        // =============================================================
        // (b) Retrieve: w_bar[BT, BK_SUB] × h[BK_SUB, BV] = [BT, BV]
        // =============================================================
        v4f32_t r_retrieve[O_E_M * O_E_N];
        clear_v4f32<O_E_M * O_E_N>(r_retrieve);

        D_ACC g_last = 0;

        for (int bk = 0; bk < N_K; bk++) {
            v4f32_t* h_cur = (bk == 0) ? h1 : h2;

            {
                D_ATTN* s_sub = s_pool;

                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    if (i < PF_ELEMS) {
                        int row = i / PF_NVEC;
                        int col = (i % PF_NVEC) * PF_VEC;
                        *reinterpret_cast<v4bf16_t*>(&s_sub[row * STRIDE_BK + col]) =
                            *reinterpret_cast<const v4bf16_t*>(&pf_w[li]);
                        *reinterpret_cast<v4bf16_t*>(&s_sub[row * STRIDE_BK + col + 4]) =
                            *(reinterpret_cast<const v4bf16_t*>(&pf_w[li]) + 1);
                    }
                }
                __syncthreads();

                if (bk == 0) g_last = s_g[BT - 1];

                if (bk + 1 < N_K) {
                    int k_off_next = (bk + 1) * BK_SUB;
                    const int t0_k_byte = t0 * stride_k * (int)sizeof(D_ATTN);
                    #pragma unroll
                    for (int li = 0; li < PF_LOADS; li++) {
                        int i = tid + li * BS;
                        if (i < PF_ELEMS) {
                            int row = i / PF_NVEC;
                            int col = (i % PF_NVEC) * PF_VEC;
                            int voff = (row * stride_k + k_off_next + col) * (int)sizeof(D_ATTN);
                            pf_w[li] = buf_load_b128(w_rsrc, voff, t0_k_byte);
                        } else {
                            pf_w[li] = {};
                        }
                    }
                }

                for (int ek = 0; ek < O_E_K; ek++) {
                    v4bf16_t b_tile = accum_to_src(h_cur[ek]);
                    #pragma unroll
                    for (int em = 0; em < O_E_M; em++) {
                        v4bf16_t a_tile = load_mfma_tile(
                            s_sub, em * W, ek * W, STRIDE_BK, lane_id);
                        r_retrieve[em] = mfma_f32_16x16x16_bf16(
                            a_tile, b_tile, r_retrieve[em]);
                    }
                }
                __syncthreads();
            }
        }

        // =============================================================
        // (b') v_new = u_bar - retrieve
        // =============================================================
        {
            D_ATTN* s_u = s_pool;
            v4f32_t pf_u[UPF_LOADS];

            const int t0_v_byte = t0 * stride_v * (int)sizeof(D_ATTN);
            #pragma unroll
            for (int li = 0; li < UPF_LOADS; li++) {
                int i = tid + li * BS;
                if (i < UPF_ELEMS) {
                    int row = i / UPF_NVEC;
                    int col = (i % UPF_NVEC) * UPF_VEC;
                    int voff = (row * stride_v + v_off + col) * (int)sizeof(D_ATTN);
                    pf_u[li] = buf_load_b128(u_rsrc, voff, t0_v_byte);
                } else {
                    pf_u[li] = {};
                }
            }

            // Store h snapshot (skip if fused output — h consumed in-kernel)
            if (!do_output && kargs.ptr_h) {
                D_ATTN* h_staging = s_pool;
                D_ATTN* snap = reinterpret_cast<D_ATTN*>(kargs.ptr_h)
                    + ((int64_t)i_n * kargs.NT * H + (int64_t)i_t * H + i_h) * K * V;

                for (int bk = 0; bk < N_K; bk++) {
                    v4f32_t* h_cur = (bk == 0) ? h1 : h2;
                    int hbm_k_base = bk * BK_SUB;

                    int lds_col = warp_n_base + lane_col;
                    for (int em = 0; em < C_ELEMS; em++) {
                        for (int n = 0; n < 4; n++) {
                            int row = em * W + ACC_ROW(n);
                            h_staging[row * BV + lds_col] =
                                static_cast<D_ATTN>(h_cur[em][n]);
                        }
                    }
                    __syncthreads();

                    constexpr int COOP_TPR = BV / 8;
                    int t_row_grp = tid / COOP_TPR;
                    int t_col_base = (tid % COOP_TPR) * 8;
                    constexpr int ROWS_PER_BATCH = BS / COOP_TPR;
                    for (int kr = t_row_grp; kr < BK_SUB; kr += ROWS_PER_BATCH) {
                        *reinterpret_cast<v4bf16_t*>(
                            &snap[(hbm_k_base + kr) * V + v_off + t_col_base]) =
                            *reinterpret_cast<v4bf16_t*>(&h_staging[kr * BV + t_col_base]);
                        *reinterpret_cast<v4bf16_t*>(
                            &snap[(hbm_k_base + kr) * V + v_off + t_col_base + 4]) =
                            *reinterpret_cast<v4bf16_t*>(&h_staging[kr * BV + t_col_base + 4]);
                    }
                    __syncthreads();
                }
            }

            // Install u_bar to LDS
            #pragma unroll
            for (int li = 0; li < UPF_LOADS; li++) {
                int i = tid + li * BS;
                if (i < UPF_ELEMS) {
                    int row = i / UPF_NVEC;
                    int col = (i % UPF_NVEC) * UPF_VEC;
                    *reinterpret_cast<v4bf16_t*>(&s_u[row * BV + col]) =
                        *reinterpret_cast<const v4bf16_t*>(&pf_u[li]);
                    *reinterpret_cast<v4bf16_t*>(&s_u[row * BV + col + 4]) =
                        *(reinterpret_cast<const v4bf16_t*>(&pf_u[li]) + 1);
                }
            }
            __syncthreads();

            // Compute ungated v_new in r_retrieve
            for (int em = 0; em < O_E_M; em++) {
                for (int n = 0; n < 4; n++) {
                    int s_idx = em * W + ACC_ROW(n);
                    int c = warp_n_base + lane_col;
                    D_ACC u_val = static_cast<D_ACC>(s_u[s_idx * BV + c]);
                    r_retrieve[em][n] = u_val - r_retrieve[em][n];
                }
            }

            // v_new stores (skip if fused output — v_new consumed in-kernel)
            if (!do_output && kargs.ptr_v_new) {
                D_ATTN* vn_out = reinterpret_cast<D_ATTN*>(kargs.ptr_v_new) + hbm_v_off;
                for (int em = 0; em < O_E_M; em++) {
                    for (int n = 0; n < 4; n++) {
                        int s_idx = em * W + ACC_ROW(n);
                        int c = warp_n_base + lane_col;
                        vn_out[(t0 + s_idx) * stride_v + v_off + c] =
                            static_cast<D_ATTN>(r_retrieve[em][n]);
                    }
                }
            }
        }

        // =============================================================
        // Fused output: o = scale * (Q×h×exp(g) + tril(QK^T×exp(g_diff))×v_new)
        // =============================================================
        if (do_output) {
            v4f32_t r_o[O_E_M * O_E_N];
            clear_v4f32<O_E_M * O_E_N>(r_o);

            v4f32_t r_A[A_E_M * A_E_N];
            clear_v4f32<A_E_M * A_E_N>(r_A);

            // Combined Q×h (cross-chunk) + QK^T (intra-chunk attention)
            for (int bk = 0; bk < N_K; bk++) {
                v4f32_t* h_cur = (bk == 0) ? h1 : h2;
                v4f32_t pf_q_buf[PF_LOADS];
                v4f32_t pf_k_buf[PF_LOADS];

                const int t0_k_byte = t0 * stride_k * (int)sizeof(D_ATTN);
                const int bk_off = bk * BK_SUB;

                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    if (i < PF_ELEMS) {
                        int row = i / PF_NVEC;
                        int col = (i % PF_NVEC) * PF_VEC;
                        int voff = (row * stride_k + bk_off + col) * (int)sizeof(D_ATTN);
                        pf_q_buf[li] = buf_load_b128(q_rsrc, voff, t0_k_byte);
                        pf_k_buf[li] = buf_load_b128(k_rsrc, voff, t0_k_byte);
                    } else {
                        pf_q_buf[li] = {};
                        pf_k_buf[li] = {};
                    }
                }

                D_ATTN* s_q = s_pool;
                D_ATTN* s_k = s_v_T;

                __syncthreads();

                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    if (i < PF_ELEMS) {
                        int row = i / PF_NVEC;
                        int col = (i % PF_NVEC) * PF_VEC;
                        *reinterpret_cast<v4bf16_t*>(&s_q[row * STRIDE_BK + col]) =
                            *reinterpret_cast<const v4bf16_t*>(&pf_q_buf[li]);
                        *reinterpret_cast<v4bf16_t*>(&s_q[row * STRIDE_BK + col + 4]) =
                            *(reinterpret_cast<const v4bf16_t*>(&pf_q_buf[li]) + 1);
                        *reinterpret_cast<v4bf16_t*>(&s_k[row * STRIDE_BK + col]) =
                            *reinterpret_cast<const v4bf16_t*>(&pf_k_buf[li]);
                        *reinterpret_cast<v4bf16_t*>(&s_k[row * STRIDE_BK + col + 4]) =
                            *(reinterpret_cast<const v4bf16_t*>(&pf_k_buf[li]) + 1);
                    }
                }
                __syncthreads();

                // Cross-chunk: o_cross += Q × h (A=Q from LDS, B=h from registers)
                for (int ek = 0; ek < O_E_K; ek++) {
                    v4bf16_t b_tile = accum_to_src(h_cur[ek]);
                    #pragma unroll
                    for (int em = 0; em < O_E_M; em++) {
                        v4bf16_t a_tile = load_mfma_tile(
                            s_q, em * W, ek * W, STRIDE_BK, lane_id);
                        r_o[em] = mfma_f32_16x16x16_bf16(
                            a_tile, b_tile, r_o[em]);
                    }
                }

                // QK^T: A += Q × K^T (A=Q from s_pool, B=K from s_v_T)
                tiled_gemm_mfma<A_E_M, A_E_N, O_E_K>(
                    r_A,
                    s_q, 0, STRIDE_BK,
                    s_k, warp_n_base, STRIDE_BK,
                    lane_id);

                __syncthreads();
            }

            // Apply exp(g) to cross-chunk output
            for (int em = 0; em < O_E_M; em++) {
                for (int n = 0; n < 4; n++) {
                    int s_idx = em * W + ACC_ROW(n);
                    r_o[em][n] *= fast_exp(s_g[s_idx]);
                }
            }

            // Gate and mask QK^T: A[row,col] *= exp(g[row]-g[col]) if row>=col, else 0
            for (int em = 0; em < A_E_M; em++) {
                for (int n = 0; n < 4; n++) {
                    int row = em * W + ACC_ROW(n);
                    int col = warp_n_base + lane_col;
                    r_A[em][n] = (row >= col) ?
                        r_A[em][n] * fast_exp(s_g[row] - s_g[col]) : 0.0f;
                }
            }

            // Scatter gated A_QK → s_pool [BT, STRIDE_BT] for AV GEMM
            // Scatter ungated v_new → s_v_T [BV, STRIDE_BT] (transposed) for AV GEMM
            {
                D_ATTN* s_A = s_pool;
                for (int em = 0; em < A_E_M; em++) {
                    for (int n = 0; n < 4; n++) {
                        int row = em * W + ACC_ROW(n);
                        int col = warp_n_base + lane_col;
                        s_A[row * STRIDE_BT + col] = static_cast<D_ATTN>(r_A[em][n]);
                    }
                }
                for (int em = 0; em < O_E_M; em++) {
                    for (int n = 0; n < 4; n++) {
                        int s_idx = em * W + ACC_ROW(n);
                        int c = warp_n_base + lane_col;
                        s_v_T[c * STRIDE_BT + s_idx] =
                            static_cast<D_ATTN>(r_retrieve[em][n]);
                    }
                }
            }
            __syncthreads();

            // AV GEMM: o_intra = A_QK × v_new^T
            // A from s_pool[BT, STRIDE_BT], B from s_v_T[BV, STRIDE_BT]
            v4f32_t r_o_intra[O_E_M * O_E_N];
            clear_v4f32<O_E_M * O_E_N>(r_o_intra);

            tiled_gemm_mfma<O_E_M, O_E_N, BT / W>(
                r_o_intra,
                s_pool, 0, STRIDE_BT,
                s_v_T, warp_n_base, STRIDE_BT,
                lane_id);
            __syncthreads();

            // Combine and store: o = scale * (o_cross_gated + o_intra)
            D_ACC sc = kargs.scale;
            for (int em = 0; em < O_E_M; em++) {
                for (int n = 0; n < 4; n++) {
                    int s_idx = em * W + ACC_ROW(n);
                    int c = warp_n_base + lane_col;
                    D_ACC o_val = sc * (r_o[em][n] + r_o_intra[em][n]);
                    o_hbm[(t0 + s_idx) * stride_v + v_off + c] =
                        static_cast<D_ATTN>(o_val);
                }
            }

            // Now gate v_new for accumulate GEMM (overwrite s_v_T with gated values)
            for (int em = 0; em < O_E_M; em++) {
                for (int n = 0; n < 4; n++) {
                    int s_idx = em * W + ACC_ROW(n);
                    int c = warp_n_base + lane_col;
                    D_ACC gate = fast_exp(g_last - s_g[s_idx]);
                    s_v_T[c * STRIDE_BT + s_idx] =
                        static_cast<D_ATTN>(r_retrieve[em][n] * gate);
                }
            }
        } else {
            // Non-fused path: gate v_new → s_v_T for accumulate GEMM
            for (int em = 0; em < O_E_M; em++) {
                for (int n = 0; n < 4; n++) {
                    int s_idx = em * W + ACC_ROW(n);
                    int c = warp_n_base + lane_col;
                    D_ACC gate = fast_exp(g_last - s_g[s_idx]);
                    s_v_T[c * STRIDE_BT + s_idx] =
                        static_cast<D_ATTN>(r_retrieve[em][n] * gate);
                }
            }
        }

        // =============================================================
        // (d) Decay h + Accumulate: h += k^T × v_gated
        // =============================================================
        v4f32_t pf_k[PF_LOADS];
        {
            const int t0_k_byte = t0 * stride_k * (int)sizeof(D_ATTN);
            #pragma unroll
            for (int li = 0; li < PF_LOADS; li++) {
                int i = tid + li * BS;
                if (i < PF_ELEMS) {
                    int row = i / PF_NVEC;
                    int col = (i % PF_NVEC) * PF_VEC;
                    int voff = (row * stride_k + col) * (int)sizeof(D_ATTN);
                    pf_k[li] = buf_load_b128(k_rsrc, voff, t0_k_byte);
                } else {
                    pf_k[li] = {};
                }
            }
        }

        D_ACC decay = fast_exp(g_last);
        for (int em = 0; em < C_ELEMS; em++) {
            for (int n = 0; n < 4; n++) {
                h1[em][n] *= decay;
                if constexpr (N_K >= 2) h2[em][n] *= decay;
            }
        }

        D_ATTN* s_k_T = s_pool;

        for (int bk = 0; bk < N_K; bk++) {
            v4f32_t* h_cur = (bk == 0) ? h1 : h2;

            #pragma unroll
            for (int li = 0; li < PF_LOADS; li++) {
                int i = tid + li * BS;
                if (i < PF_ELEMS) {
                    int s = i / PF_NVEC;
                    int j = (i % PF_NVEC) * PF_VEC;
                    const v4bf16_t* bf16_ptr = reinterpret_cast<const v4bf16_t*>(&pf_k[li]);
                    for (int vi = 0; vi < 4; vi++)
                        s_k_T[(j + vi) * STRIDE_BT + s] = bf16_ptr[0][vi];
                    for (int vi = 0; vi < 4; vi++)
                        s_k_T[(j + 4 + vi) * STRIDE_BT + s] = bf16_ptr[1][vi];
                }
            }
            __syncthreads();

            if (bk + 1 < N_K) {
                int k_off_next = (bk + 1) * BK_SUB;
                const int t0_k_byte = t0 * stride_k * (int)sizeof(D_ATTN);
                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    if (i < PF_ELEMS) {
                        int s = i / PF_NVEC;
                        int j = (i % PF_NVEC) * PF_VEC;
                        int voff = (s * stride_k + k_off_next + j) * (int)sizeof(D_ATTN);
                        pf_k[li] = buf_load_b128(k_rsrc, voff, t0_k_byte);
                    } else {
                        pf_k[li] = {};
                    }
                }
            }

            tiled_gemm_mfma<H_E_M, H_E_N, H_E_K>(
                h_cur,
                s_k_T, 0, STRIDE_BT,
                s_v_T, warp_n_base, STRIDE_BT,
                lane_id);
            __syncthreads();
        }

        // Prefetch next chunk's w_bar[bk=0]
        if (i_s + 1 < S) {
            int next_t0 = (i_t + 1) * BT;
            #pragma unroll
            for (int li = 0; li < PF_LOADS; li++) {
                int i = tid + li * BS;
                if (i < PF_ELEMS) {
                    int row = i / PF_NVEC;
                    int col = (i % PF_NVEC) * PF_VEC;
                    int voff = ((next_t0 + row) * stride_k + col) * (int)sizeof(D_ATTN);
                    pf_w[li] = buf_load_b128(w_rsrc, voff, 0);
                } else {
                    pf_w[li] = {};
                }
            }
        }

    }  // end S-chunk loop

    #undef ACC_ROW

    // Epilogue: h_pass + signal
    {
        const int hp_stride = kargs.N_super * N_K * BV * BK_SUB;
        D_ACC* hp = reinterpret_cast<D_ACC*>(kargs.ptr_h_pass)
                    + i_flat * hp_stride + i_super * N_K * BV * BK_SUB;
        int col = warp_n_base + lane_col;
        for (int em = 0; em < C_ELEMS; em++) {
            for (int n = 0; n < 4; n++) {
                int row = em * W + lane_row_base + n;
                hp[col * BK_SUB + row] = h1[em][n];
            }
        }
        if constexpr (N_K >= 2) {
            for (int em = 0; em < C_ELEMS; em++) {
                for (int n = 0; n < 4; n++) {
                    int row = em * W + lane_row_base + n;
                    hp[BV * BK_SUB + col * BK_SUB + row] = h2[em][n];
                }
            }
        }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (tid == 0) {
        atomicAdd(kargs.ptr_flags + i_flat * kargs.N_super + i_super, 1u);
    }

    // Final state
    if (kargs.ptr_ht != nullptr && i_super == kargs.N_super - 1) {
        D_ACC* ht = reinterpret_cast<D_ACC*>(kargs.ptr_ht)
                    + (i_n * H + i_h) * V * K;
        int col = warp_n_base + lane_col;
        for (int em = 0; em < C_ELEMS; em++) {
            for (int n = 0; n < 4; n++) {
                int row = em * W + lane_row_base + n;
                ht[(v_off + col) * K + row] = h1[em][n];
            }
        }
        if constexpr (N_K >= 2) {
            for (int em = 0; em < C_ELEMS; em++) {
                for (int n = 0; n < 4; n++) {
                    int row = em * W + lane_row_base + n;
                    ht[(v_off + col) * K + BK_SUB + row] = h2[em][n];
                }
            }
        }
    }
}
