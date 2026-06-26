// Gated DeltaNet (GDN) Prefill Kernel — shared types, kargs, traits
// Target: gfx942 (MI300X) / gfx950 (MI350), MFMA bf16 16×16×16
#pragma once

#ifdef __HIP_DEVICE_COMPILE__
using bf16_t = __bf16;
#else
using bf16_t = unsigned short;
#endif

// --------------------------------------------------------------------------
// K1 kernel arguments: Step 1 (cumsum + KKT) + Step 2 (trisol + WY factors)
// --------------------------------------------------------------------------
struct gdn_k1_kargs {
    const void* __restrict__ ptr_k;        // [B, T, H, K]   bf16
    const void* __restrict__ ptr_v;        // [B, T, H, V]   bf16
    const void* __restrict__ ptr_beta;     // [B, T, H]      fp32
    const void* __restrict__ ptr_g;        // [B, T, H]      fp32
    void* __restrict__ ptr_w_bar;          // [B, T, H, K]   bf16  output
    void* __restrict__ ptr_u_bar;          // [B, T, H, V]   bf16  output
    void* __restrict__ ptr_g_cumsum;       // [B, T, H]      fp32  output
    int B;
    int T;
    int H;
    int K;
    int V;
};

// --------------------------------------------------------------------------
// K2 kernel arguments: Step 3 (h update) + Step 4 (output)
// --------------------------------------------------------------------------
struct gdn_k2_kargs {
    const void* __restrict__ ptr_q;        // [B, T, H, K]    bf16
    const void* __restrict__ ptr_k;        // [B, T, H, K]    bf16
    const void* __restrict__ ptr_w_bar;    // [B, T, H, K]    bf16
    const void* __restrict__ ptr_u_bar;    // [B, T, H, V]    bf16
    const void* __restrict__ ptr_g_cumsum; // [B, T, H]       fp32
    const void* __restrict__ ptr_h0;       // [B, H, V, K]    fp32  (nullable)
    void* __restrict__ ptr_o;              // [B, T, H, V]    bf16  output
    void* __restrict__ ptr_ht;             // [B, H, V, K]    fp32  final state (nullable)
    void* __restrict__ ptr_h_snap;         // [B, NT, H, V, K] fp32 h snapshots
    void* __restrict__ ptr_v_new;          // [B, T, H, V]    bf16  corrected values
    int B;
    int T;
    int H;
    int K;
    int V;
    int NT;
    float scale;
};

// --------------------------------------------------------------------------
// K1 traits
//
// Layout: [B, T, H, K] with stride_t = H*K, stride_h = K, stride_k = 1
// MFMA: bf16 16×16×16 uniformly on gfx942/gfx950
// --------------------------------------------------------------------------
template<int BT_,
         int K_  = 128,
         int V_  = 128,
         int NUM_WARPS_ = 4>
struct gdn_k1_traits {
    static constexpr int BT = BT_;
    static constexpr int K  = K_;
    static constexpr int V  = V_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_ATTN = bf16_t;
    using D_ACC  = float;

    // MFMA tile: 16×16×16 bf16 → fp32
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 16;

    // Wave tiling: all warps along M, 1 wave along N and K
    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    // KKT GEMM: [BT, K] × [K, BT] = [BT, BT]
    static constexpr int KKT_E_M = BT / W_M;          // 4 (BT=64), 1 (BT=16)
    static constexpr int KKT_E_N = BT / W_N;          // 4 (BT=64), 1 (BT=16)
    static constexpr int KKT_E_K = K / W_K;           // 8

    // w_bar GEMM: C[BT, BT] × k_scaled[BT, BK_sub] = [BT, BK_sub]
    // u_bar GEMM: C[BT, BT] × v_scaled[BT, BV_sub] = [BT, BV_sub]
    // K and V are processed in 64-wide subtiles (BK_sub=BV_sub=64)
    static constexpr int BK_SUB = 64;
    static constexpr int BV_SUB = 64;
    static constexpr int N_K_ITERS = K / BK_SUB;      // 2
    static constexpr int N_V_ITERS = V / BV_SUB;      // 2

    static constexpr int WY_E_M = BT / W_M;           // 4 (BT=64), 1 (BT=16)
    static constexpr int WY_E_N = BK_SUB / W_N;       // 4
    static constexpr int WY_E_K = BT / W_K;           // 4 (BT=64), 1 (BT=16)

    // Vector widths for load/store (bf16x8 = 16 bytes)
    static constexpr int VEC_KV = 8;

    // MFMA LDS padding: +4 bf16 elements per row to avoid bank conflicts
    // (effective on both 32-bank gfx942 and 64-bank gfx950)
    static constexpr int SMEM_PAD = 4;
    static constexpr int K_STRIDE = K + SMEM_PAD;

    // fp32 A matrix: stride padded to BT+1 to avoid bank conflicts
    // (BT DWORDs would align every row to the same bank on 32/64-bank LDS)
    static constexpr int A_STRIDE = BT + 1;

    // LDS layout (sizes in bytes)
    static constexpr int smem_k_padded_bytes = BT * K_STRIDE * (int)sizeof(D_ATTN);
    static constexpr int smem_A_bytes     = BT * A_STRIDE * (int)sizeof(D_ACC);
    static constexpr int smem_scalar_bytes = BT * 2 * (int)sizeof(D_ACC);
    static constexpr int smem_subtile_bytes = BT * BK_SUB * (int)sizeof(D_ATTN);

    static constexpr size_t smem_size_bytes() {
        int phase1 = smem_k_padded_bytes + smem_scalar_bytes;
        if constexpr (BT >= 64) {
            int phase2ab = smem_A_bytes + 16*16*(int)sizeof(D_ACC) + smem_scalar_bytes;
            int c_bf16_bytes = BT * (BT + SMEM_PAD) * (int)sizeof(D_ATTN);
            int phase2c = smem_A_bytes + c_bf16_bytes + smem_scalar_bytes;
            int peak = phase1;
            if (phase2ab > peak) peak = phase2ab;
            if (phase2c  > peak) peak = phase2c;
            return peak;
        } else {
            int phase2 = smem_A_bytes + smem_subtile_bytes + smem_scalar_bytes;
            return (phase1 > phase2) ? phase1 : phase2;
        }
    }

    // Number of 16×16 diagonal blocks for forward substitution (BT=64 → 4 blocks)
    static constexpr int N_DIAG_BLOCKS = BT / 16;
};

// --------------------------------------------------------------------------
// K2 traits
//
// Grid: (cdiv(V, BV), N*H) — V dimension parallel, chunks serial
// h state: N_K registers × [BK_SUB, BV] fp32 in MFMA accumulator layout
// GEMMs: 5× MFMA bf16 16×16×16 (BT≥32 multi-tile, BT<32 single-tile QK^T)
// --------------------------------------------------------------------------
template<int BT_,
         int K_  = 128,
         int V_  = 128,
         int BV_ = 64,
         int NUM_WARPS_ = 4>
struct gdn_k2_traits {
    static constexpr int BT = BT_;
    static constexpr int K  = K_;
    static constexpr int V  = V_;
    static constexpr int BV = BV_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int OCC_HINT = (BT_ >= 128) ? 1 : ((NUM_WARPS_ <= 4) ? 2 : 1);

    using D_ATTN = bf16_t;
    using D_ACC  = float;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 16;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    // K subtiling: h state split into 64-wide K subtiles (matches Triton)
    static constexpr int BK_SUB = 64;
    static constexpr int N_K = K / BK_SUB;             // 2 for K=128

    // Step 3 Retrieve GEMM: w_bar_sub[BT, BK_SUB] × h[BK_SUB, BV] = tmp[BT, BV]
    static constexpr int RET_E_M = BT / W_M;
    static constexpr int RET_E_N = BV / W_N;
    static constexpr int RET_E_K = BK_SUB / W_K;

    // Step 3 Accumulate GEMM: k_sub^T[BK_SUB, BT] × v_hat[BT, BV] = [BK_SUB, BV]
    static constexpr int ACC_E_M = BK_SUB / W_M;
    static constexpr int ACC_E_N = BV / W_N;
    static constexpr int ACC_E_K = BT / W_K;

    // Step 4 Intra QK^T: q_sub[BT, BK_SUB] × k_sub^T[BK_SUB, BT] = A_intra[BT, BT]
    static constexpr int QKT_E_M = BT / W_M;
    static constexpr int QKT_E_N = BT / W_N;
    static constexpr int QKT_E_K = BK_SUB / W_K;

    // Step 4 Intra AV: A_intra[BT, BT] × v_new[BT, BV] = [BT, BV]
    static constexpr int AV_E_M = BT / W_M;
    static constexpr int AV_E_N = BV / W_N;
    static constexpr int AV_E_K = BT / W_K;

    static constexpr int VEC_KV = 8;

    // MFMA LDS padding: +4 bf16 elements per row to avoid bank conflicts
    // (effective on both 32-bank gfx942 and 64-bank gfx950)
    static constexpr int SMEM_PAD = 4;

    // LDS layout (MFMA optimized, all bf16 buffers padded)
    //
    // Persistent regions:
    //   s_g[BT] fp32                          — gate cumsum
    //   s_v_T[BV, BT+PAD] bf16               — v_new transposed (phases b'→e)
    //
    // Pool (aliased per phase):
    //   Phase b/c: s_h_T[BV, BK_SUB+PAD] + s_sub[BT, BK_SUB+PAD]
    //   Phase d:   s_k_T[BK_SUB, BT+PAD] bf16
    //   Phase e:   s_q[BT, BK_SUB+PAD] + s_k[BT, BK_SUB+PAD] bf16  (QK^T)
    //              s_A[BT, BT+PAD] bf16                              (AV)
    static constexpr int smem_g_bytes = BT * (int)sizeof(D_ACC);
    static constexpr int smem_vT_bytes = BV * (BT + SMEM_PAD) * (int)sizeof(D_ATTN);
    // Persistent q for the whole chunk: all N_K subtiles, [BT, BK_SUB+PAD] each.
    // Loaded once in phase b/c and reused in phase e (QK^T) — avoids re-reading
    // q from HBM. Free LDS: k2 occupancy is VGPR-bound, not LDS-bound.
    static constexpr int N_K_ = K / BK_SUB;
    static constexpr int smem_q_bytes = N_K_ * BT * (BK_SUB + SMEM_PAD) * (int)sizeof(D_ATTN);

    static constexpr size_t smem_size_bytes() {
        constexpr int P = SMEM_PAD;
        int pool_bc  = BV * (BK_SUB + P) * (int)sizeof(D_ATTN)     // s_h_T
                     + BT * (BK_SUB + P) * (int)sizeof(D_ATTN);   // s_sub_w
        int pool_d   = BK_SUB * (BT + P) * (int)sizeof(D_ATTN);    // s_k_T
        int pool_eqk = BT * (BK_SUB + P) * (int)sizeof(D_ATTN);    // s_k4 (q now persistent)
        int pool_eav = BT * (BT + P) * (int)sizeof(D_ATTN);        // s_A
        int pool = pool_bc;
        if (pool_d   > pool) pool = pool_d;
        if (pool_eqk > pool) pool = pool_eqk;
        if (pool_eav > pool) pool = pool_eav;
        return smem_g_bytes + smem_vT_bytes + smem_q_bytes + pool;
    }
};

// --------------------------------------------------------------------------
// Wavefront scan-only kernel arguments (h-state update with inter-WG sync)
// --------------------------------------------------------------------------
struct gdn_wf_h_kargs {
    const void* __restrict__ ptr_k;         // [B, T, H, K]   bf16
    const void* __restrict__ ptr_w_bar;     // [B, T, H, K]   bf16
    const void* __restrict__ ptr_u_bar;     // [B, T, H, V]   bf16
    const void* __restrict__ ptr_g_cumsum;  // [B, T, H]      fp32
    const void* __restrict__ ptr_h0;        // [B, H, V, K]   fp32  (nullable)
    void* __restrict__ ptr_h;               // [B, NT, H, K, V] bf16  h snapshots (nullable)
    void* __restrict__ ptr_v_new;           // [B, T, H, V]   bf16  (nullable)
    void* __restrict__ ptr_ht;              // [B, H, V, K]   fp32  final state (nullable)
    void* __restrict__ ptr_h_pass;          // [N_flat, N_super, K, BV] fp32
    uint32_t* __restrict__ ptr_flags;       // [N_flat * N_super] uint32
    const void* __restrict__ ptr_q;         // [B, T, H, K]   bf16  (nullable, for fused output)
    void* __restrict__ ptr_o;               // [B, T, H, V]   bf16  (nullable, fused output)
    int B, T, H, K, V, NT;
    int S;
    int N_super;
    float scale;
};

// --------------------------------------------------------------------------
// Utilities
// --------------------------------------------------------------------------
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
