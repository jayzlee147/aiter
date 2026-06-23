// GDN Prefill K1 Kernel — BT=64, MFMA Neumann variant
// Phase 2a uses MFMA Neumann series instead of scalar forward substitution.
//
// Grid: (NT, B*H)   Block: (BLOCK_SIZE = 256)
// Target: gfx942 (MI300X) / gfx950 (MI350), MFMA bf16 16×16×16
#pragma once

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_mfma_utils.h"

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, 2)
gdn_k1_neumann_kernel(gdn_k1_kargs kargs) {
    using namespace gdn_mfma;
    using T = Traits;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    static_assert(T::BT == 64, "This template is for BT=64 only");

    const int i_t  = blockIdx.x;
    const int i_bh = blockIdx.y;
    const int i_b  = i_bh / kargs.H;
    const int i_h  = i_bh % kargs.H;

    const int tid  = threadIdx.x;
    const int warp_id = tid / T::WARP_SIZE;
    const int lane_id = tid % T::WARP_SIZE;

    constexpr int BT = T::BT;
    constexpr int BS = T::BLOCK_SIZE;
    constexpr int PAD = T::SMEM_PAD;
    constexpr int K_STRIDE = T::K_STRIDE;
    constexpr int BK_SUB = T::BK_SUB;
    constexpr int A_STRIDE = T::A_STRIDE;
    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;

    const int chunk_start = i_t * BT;
    const int bos = i_b * kargs.T;

    // =====================================================================
    // Shared memory allocation
    //
    // Phase 1:  s_g[BT] + s_beta[BT] + s_k[BT×K_STRIDE]     = 17408 bytes
    // Phase 2a: s_g + s_beta + s_A[BT×A_STRIDE] (aliases s_k) = padded bytes
    // Phase 2c: s_g + s_beta + s_vT[BK_SUB×VT_STRIDE]        =  9216 bytes
    //           (C_inv cached in registers, no s_C_bf16)
    // Peak: 17408 bytes → fits 3 blocks/CU (17408×3 = 52224 < 65536)
    // =====================================================================
    extern __shared__ char smem_buf[];

    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);
    D_ACC*  s_beta = s_g + BT;
    D_ATTN* s_k    = reinterpret_cast<D_ATTN*>(s_beta + BT);

    D_ACC*  s_A    = reinterpret_cast<D_ACC*>(s_k);

    // =====================================================================
    // Phase 1a: Load g and beta, compute prefix sum
    // =====================================================================
    const D_ACC* g_base    = reinterpret_cast<const D_ACC*>(kargs.ptr_g)
                             + (bos + chunk_start) * H + i_h;
    const D_ACC* beta_base = reinterpret_cast<const D_ACC*>(kargs.ptr_beta)
                             + (bos + chunk_start) * H + i_h;

    for (int i = tid; i < BT; i += BS) {
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

    // Prefix sum (Hillis-Steele)
    for (int stride = 1; stride < BT; stride <<= 1) {
        for (int i = tid; i < BT; i += BS) {
            if (i >= stride)
                s_g[i] += s_g[i - stride];
        }
        __syncthreads();
    }

    // Write g_cumsum to HBM
    D_ACC* g_cumsum_base = reinterpret_cast<D_ACC*>(kargs.ptr_g_cumsum)
                           + (bos + chunk_start) * H + i_h;
    for (int i = tid; i < BT; i += BS) {
        int global_t = chunk_start + i;
        if (global_t < kargs.T)
            g_cumsum_base[i * H] = s_g[i];
    }
    __syncthreads();

    // =====================================================================
    // Phase 1b: Load k[BT, K] into LDS with padding (stride = K_STRIDE)
    // =====================================================================
    const D_ATTN* k_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                           + ((bos + chunk_start) * H + i_h) * K;

    {
        constexpr int VEC = 4;
        constexpr int K_VEC = T::K / VEC;
        for (int i = tid; i < BT * K_VEC; i += BS) {
            int row = i / K_VEC;
            int col = (i % K_VEC) * VEC;
            int global_t = chunk_start + row;
            v4bf16_t val = {};
            if (global_t < kargs.T)
                val = *reinterpret_cast<const v4bf16_t*>(
                    &k_base[row * H * K + col]);
            *reinterpret_cast<v4bf16_t*>(&s_k[row * K_STRIDE + col]) = val;
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 1c+1d: KKT GEMM via MFMA — k × k^T self-matmul
    // =====================================================================

    constexpr int KKT_E_M = 1;
    constexpr int KKT_E_N = BT / 16;       // 4
    constexpr int KKT_E_K = T::K / 16;     // 8

    v4f32_t kkt_c[KKT_E_M * KKT_E_N];
    clear_v4f32<KKT_E_M * KKT_E_N>(kkt_c);

    tiled_gemm_mfma<KKT_E_M, KKT_E_N, KKT_E_K>(
        kkt_c, s_k, warp_id * 16, K_STRIDE,
               s_k, 0,            K_STRIDE, lane_id);

    __syncthreads();

    // Post-MFMA: gate-scale lower triangle, zero upper+diagonal, write fp32 to s_A
    for (int en = 0; en < KKT_E_N; en++) {
        for (int p = 0; p < 4; p++) {
            int s = warp_id * 16 + (lane_id >> 4) * 4 + p;
            int r = en * 16 + (lane_id & 15);
            float val = 0.0f;
            if (s > r)
                val = kkt_c[en][p] * s_beta[s] * __expf(s_g[s] - s_g[r]);
            s_A[s * A_STRIDE + r] = val;
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 2a: Triangular inverse — MFMA Neumann series
    //
    // (I+A)^{-1} = sum_{n=0}^{15} (-A)^n (exact: A is 16×16 strictly lower
    // triangular, hence nilpotent with (-A)^16 = 0).
    //
    // Each warp handles one 16×16 diagonal block independently.
    // 15 MFMA iterations per block via register chaining (accum_to_src).
    // =====================================================================

    constexpr v4f32_t z4 = {0.f, 0.f, 0.f, 0.f};

    {
        int br = warp_id * 16;

        // Load (-A) as MFMA A operand
        v4bf16_t neg_A_tile;
        {
            int base = (br + (lane_id & 15)) * A_STRIDE + br + ((lane_id >> 4) << 2);
            neg_A_tile = v4bf16_t{
                static_cast<__bf16>(-s_A[base]),
                static_cast<__bf16>(-s_A[base + 1]),
                static_cast<__bf16>(-s_A[base + 2]),
                static_cast<__bf16>(-s_A[base + 3])};
        }

        // Horner's method: (I+A)^{-1} = I + (-A)(I + (-A)(I + ...))
        v4f32_t I_accum;
        {
            int n = lane_id & 15;
            int m_base = (lane_id >> 4) * 4;
            I_accum = v4f32_t{
                (m_base == n) ? 1.0f : 0.0f,
                ((m_base + 1) == n) ? 1.0f : 0.0f,
                ((m_base + 2) == n) ? 1.0f : 0.0f,
                ((m_base + 3) == n) ? 1.0f : 0.0f};
        }

        v4f32_t C_accum = I_accum;
        for (int iter = 0; iter < 15; iter++) {
            C_accum = mfma_f32_16x16x16_bf16(
                neg_A_tile, accum_to_src(C_accum), I_accum);
        }

        // Store result and zero upper triangle
        store_fp32_tile(s_A, br, br, A_STRIDE, C_accum, lane_id);
        for (int idx = lane_id; idx < 16 * 16; idx += T::WARP_SIZE) {
            int r = idx / 16, c = idx % 16;
            if (r < c)
                s_A[(br + r) * A_STRIDE + br + c] = 0.0f;
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 2b: Schur complement merge — MFMA bf16 16×16×16
    //
    // Dependency DAG enables warp-level parallelism (3 levels, 3 barriers):
    //   Level 1: C_21, C_32, C_43  (3 warps, independent)
    //   Level 2: C_31, C_42        (2 warps, independent)
    //   Level 3: C_41              (1 warp)
    // =====================================================================

    // Pre-save L blocks overwritten in Level 1
    v4bf16_t sav_L32, sav_L43, sav_L42;
    if (warp_id == 0) {
        sav_L32 = load_fp32_tile(s_A, 32, 16, A_STRIDE, lane_id);
        sav_L43 = load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id);
    } else if (warp_id == 1) {
        sav_L43 = load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id);
    }

    v4f32_t kept_c21 = z4, kept_c32 = z4, kept_c31 = z4;

    // --- Level 1: C_21, C_32, C_43 ---
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 16, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id), z4);
        kept_c21 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 16, 16, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) kept_c21[p] = -kept_c21[p];
        store_fp32_tile(s_A, 16, 0, A_STRIDE, kept_c21, lane_id);
    } else if (warp_id == 1) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 16, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 16, 16, A_STRIDE, lane_id), z4);
        kept_c32 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 32, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) kept_c32[p] = -kept_c32[p];
        store_fp32_tile(s_A, 32, 16, A_STRIDE, kept_c32, lane_id);
    } else if (warp_id == 2) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 32, 32, A_STRIDE, lane_id), z4);
        v4f32_t c43 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) c43[p] = -c43[p];
        store_fp32_tile(s_A, 48, 32, A_STRIDE, c43, lane_id);
    }
    __syncthreads();

    // Save L_42 (overwritten by C_42 in Level 2)
    if (warp_id == 0)
        sav_L42 = load_fp32_tile(s_A, 48, 16, A_STRIDE, lane_id);

    // --- Level 2: C_31, C_42 ---
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id), z4);
        t = mfma_f32_16x16x16_bf16(sav_L32, accum_to_src(kept_c21), t);
        kept_c31 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 32, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) kept_c31[p] = -kept_c31[p];
        store_fp32_tile(s_A, 32, 0, A_STRIDE, kept_c31, lane_id);
    } else if (warp_id == 1) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 16, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 16, 16, A_STRIDE, lane_id), z4);
        t = mfma_f32_16x16x16_bf16(sav_L43, accum_to_src(kept_c32), t);
        v4f32_t c42 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) c42[p] = -c42[p];
        store_fp32_tile(s_A, 48, 16, A_STRIDE, c42, lane_id);
    }
    __syncthreads();

    // --- Level 3: C_41 ---
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id), z4);
        t = mfma_f32_16x16x16_bf16(sav_L42, accum_to_src(kept_c21), t);
        t = mfma_f32_16x16x16_bf16(sav_L43, accum_to_src(kept_c31), t);
        v4f32_t c41 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) c41[p] = -c41[p];
        store_fp32_tile(s_A, 48, 0, A_STRIDE, c41, lane_id);
    }
    __syncthreads();

    // s_A now contains C = (I + A)^{-1}

    // =====================================================================
    // Phase 2c: WY factor GEMMs via MFMA
    //
    // u_bar = C @ (v * beta)
    // w_bar = C @ (k * beta * exp(g_cumsum))
    //
    // Step 1: Convert C (fp32, s_A) → C_bf16 (placed after s_A in LDS)
    // Step 2: For each subtile, load pre-scaled v/k transposed → MFMA
    // =====================================================================

    // Convert C_inv to bf16 with padding, placed AFTER s_A
    constexpr int C_STRIDE = BT + PAD;  // 68
    D_ATTN* s_C_bf16 = reinterpret_cast<D_ATTN*>(
        smem_buf + BT * 2 * sizeof(D_ACC) + BT * A_STRIDE * sizeof(D_ACC));

    for (int i = tid; i < BT * BT; i += BS) {
        int s = i / BT;
        int j = i % BT;
        s_C_bf16[s * C_STRIDE + j] = static_cast<D_ATTN>(s_A[s * A_STRIDE + j]);
    }
    __syncthreads();

    // s_A region freed — reuse for v_scaled_T / k_scaled_T
    constexpr int VT_STRIDE = BT + PAD;  // 68
    D_ATTN* s_vT = s_k;

    constexpr int WY_EM = 1;
    constexpr int WY_EN = BK_SUB / 16;   // 4
    constexpr int WY_EK = BT / 16;       // 4

    D_ATTN* w_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_w_bar)
                         + ((bos + chunk_start) * H + i_h) * K;
    D_ATTN* u_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_u_bar)
                         + ((bos + chunk_start) * H + i_h) * V;

    // --- u_bar = C @ (v * beta) ---
    const D_ATTN* v_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_v)
                           + ((bos + chunk_start) * H + i_h) * V;

    for (int iv = 0; iv < T::N_V_ITERS; iv++) {
        int v_offset = iv * BK_SUB;

        {
            constexpr int VEC = 4;
            constexpr int NVEC = BK_SUB / VEC;
            for (int i = tid; i < BT * NVEC; i += BS) {
                int j  = i / NVEC;
                int vi = (i % NVEC) * VEC;
                v4bf16_t vals = {};
                if (chunk_start + j < kargs.T)
                    vals = *reinterpret_cast<const v4bf16_t*>(
                        &v_base[j * H * V + v_offset + vi]);
                D_ACC beta_j = s_beta[j];
                for (int vv = 0; vv < VEC; vv++)
                    s_vT[(vi + vv) * VT_STRIDE + j] = static_cast<D_ATTN>(
                        static_cast<D_ACC>(vals[vv]) * beta_j);
            }
        }
        __syncthreads();

        v4f32_t wy_c[WY_EM * WY_EN];
        clear_v4f32<WY_EM * WY_EN>(wy_c);
        tiled_gemm_mfma<WY_EM, WY_EN, WY_EK>(
            wy_c, s_C_bf16, warp_id * 16, C_STRIDE,
                  s_vT,     0,             VT_STRIDE, lane_id);

        for (int en = 0; en < WY_EN; en++) {
            for (int p = 0; p < 4; p++) {
                int s  = warp_id * 16 + (lane_id >> 4) * 4 + p;
                int vi = en * 16 + (lane_id & 15);
                if (chunk_start + s < kargs.T)
                    u_bar_base[s * H * V + v_offset + vi] =
                        static_cast<D_ATTN>(wy_c[en][p]);
            }
        }
        __syncthreads();
    }

    // --- w_bar = C @ (k * beta * exp(g_cumsum)) ---
    for (int ik = 0; ik < T::N_K_ITERS; ik++) {
        int k_offset = ik * BK_SUB;

        {
            constexpr int VEC = 4;
            constexpr int NVEC = BK_SUB / VEC;
            for (int i = tid; i < BT * NVEC; i += BS) {
                int j  = i / NVEC;
                int ki = (i % NVEC) * VEC;
                v4bf16_t vals = {};
                if (chunk_start + j < kargs.T)
                    vals = *reinterpret_cast<const v4bf16_t*>(
                        &k_base[j * H * K + k_offset + ki]);
                D_ACC scale_j = s_beta[j] * __expf(s_g[j]);
                for (int vv = 0; vv < VEC; vv++)
                    s_vT[(ki + vv) * VT_STRIDE + j] = static_cast<D_ATTN>(
                        static_cast<D_ACC>(vals[vv]) * scale_j);
            }
        }
        __syncthreads();

        v4f32_t wy_c[WY_EM * WY_EN];
        clear_v4f32<WY_EM * WY_EN>(wy_c);
        tiled_gemm_mfma<WY_EM, WY_EN, WY_EK>(
            wy_c, s_C_bf16, warp_id * 16, C_STRIDE,
                  s_vT,     0,             VT_STRIDE, lane_id);

        for (int en = 0; en < WY_EN; en++) {
            for (int p = 0; p < 4; p++) {
                int s  = warp_id * 16 + (lane_id >> 4) * 4 + p;
                int ki = en * 16 + (lane_id & 15);
                if (chunk_start + s < kargs.T)
                    w_bar_base[s * H * K + k_offset + ki] =
                        static_cast<D_ATTN>(wy_c[en][p]);
            }
        }
        __syncthreads();
    }
}
