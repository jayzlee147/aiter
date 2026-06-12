// Gated DeltaNet (GDN) Prefill Kernel — shared types, kargs, traits
// Target: MI300X (gfx942), MFMA bf16 16×16×16
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
    const void* __restrict__ ptr_h0;       // [B, H, K, V]    fp32  (nullable)
    void* __restrict__ ptr_o;              // [B, T, H, V]    bf16  output
    void* __restrict__ ptr_ht;             // [B, H, K, V]    fp32  final state (nullable)
    void* __restrict__ ptr_h_snap;         // [B, NT, H, K, V] fp32 h snapshots
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
// MFMA: bf16 16×16×16 uniformly on gfx942
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

    // LDS layout (sizes in bytes)
    // Phase 1: k[BT×K×2] + g/beta[BT×4×2]
    // Phase 2: A[BT²×4] — fp32 lower-triangular + identity
    // Phase 3: C[BT²×4] + k/v subtile[BT×64×2]
    static constexpr int smem_k_bytes     = BT * K * sizeof(D_ATTN);        // 16KB (BT=64)
    static constexpr int smem_v_bytes     = BT * V * sizeof(D_ATTN);        // 16KB
    static constexpr int smem_A_bytes     = BT * BT * sizeof(D_ACC);        // 16KB (BT=64)
    static constexpr int smem_scalar_bytes = BT * 2 * sizeof(D_ACC);        // 512B (g+beta)
    static constexpr int smem_subtile_bytes = BT * BK_SUB * sizeof(D_ATTN); // 8KB

    static constexpr size_t smem_size_bytes() {
        // Peak: max(k+scalar, A+subtile) with lifetime union
        int phase1 = smem_k_bytes + smem_scalar_bytes;
        int phase2 = smem_A_bytes + smem_subtile_bytes + smem_scalar_bytes;
        return (phase1 > phase2) ? phase1 : phase2;
    }

    // Number of 16×16 diagonal blocks for forward substitution (BT=64 → 4 blocks)
    static constexpr int N_DIAG_BLOCKS = BT / 16;
};

// --------------------------------------------------------------------------
// K2 traits
//
// Grid: (cdiv(V, BV), N*H) — V dimension parallel, chunks serial
// h state: N_K registers × [BK_SUB, BV] fp32
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

    // h state per WG: N_K × [BK_SUB, BV] fp32 in registers
    // For K=128, BV=64: b_h1[64,64] + b_h2[64,64]
    // VGPRs per h subtile: BK_SUB * BV / WARP_SIZE = 64 (fp32)
    // Total h VGPRs: N_K * 64 = 128

    // Step 3 Retrieve GEMM: w_bar_sub[BT, BK_SUB] × h[BK_SUB, BV] = tmp[BT, BV]
    static constexpr int RET_E_M = BT / W_M;          // 4 (BT=64), 1 (BT=16)
    static constexpr int RET_E_N = BV / W_N;          // 4
    static constexpr int RET_E_K = BK_SUB / W_K;      // 4

    // Step 3 Accumulate GEMM: k_sub^T[BK_SUB, BT] × v_hat[BT, BV] = [BK_SUB, BV]
    static constexpr int ACC_E_M = BK_SUB / W_M;      // 4
    static constexpr int ACC_E_N = BV / W_N;          // 4
    static constexpr int ACC_E_K = BT / W_K;          // 4 (BT=64), 1 (BT=16)

    // Step 4 Cross-chunk: q_sub[BT, BK_SUB] × h_snap[BK_SUB, BV] = [BT, BV]
    // Same tile config as Retrieve

    // Step 4 Intra QK^T: q_sub[BT, BK_SUB] × k_sub^T[BK_SUB, BT] = A_intra[BT, BT]
    static constexpr int QKT_E_M = BT / W_M;
    static constexpr int QKT_E_N = BT / W_N;
    static constexpr int QKT_E_K = BK_SUB / W_K;

    // Step 4 Intra AV: A_intra[BT, BT] × v_new[BT, BV] = [BT, BV]
    static constexpr int AV_E_M = BT / W_M;
    static constexpr int AV_E_N = BV / W_N;
    static constexpr int AV_E_K = BT / W_K;

    static constexpr int VEC_KV = 8;

    // LDS layout (with lifetime union across phases)
    // Phase (b): w_bar_sub[BT×64×2] + u_bar_sub[BT×BV×2]
    // Phase (c): q_sub[BT×64×2]
    // Phase (d): k_sub[BT×64×2] + v_hat[BT×BV×2]
    // Phase (e): A_intra[BT²×4] + v_new[BT×BV×2]
    // Scalars: g_cumsum[BT×4]
    static constexpr int smem_subtile_bytes  = BT * BK_SUB * sizeof(D_ATTN);  // 8KB (BT=64)
    static constexpr int smem_vbuf_bytes     = BT * BV * sizeof(D_ATTN);      // 8KB
    static constexpr int smem_A_intra_bytes  = BT * BT * sizeof(D_ACC);       // 16KB (BT=64)
    static constexpr int smem_g_bytes        = BT * sizeof(D_ACC);            // 256B

    static constexpr size_t smem_size_bytes() {
        // Worst case: Phase (e) — A_intra + v_new + g_cumsum
        int phase_e = smem_A_intra_bytes + smem_vbuf_bytes + smem_g_bytes;
        // Phase (b/d): 2 subtiles + g_cumsum
        int phase_bd = 2 * smem_subtile_bytes + smem_g_bytes;
        return (phase_e > phase_bd) ? phase_e : phase_bd;
    }
};

// --------------------------------------------------------------------------
// Utilities
// --------------------------------------------------------------------------
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
