// GDN Prefill K1 Kernel — BT=64, Forward Substitution (Schur Complement)
// Step 1: g_cumsum + KKT Gram matrix
// Step 2: Triangular inverse (I+A)^{-1} via 4×16x16 forward sub + Schur merge
//         + WY factor assembly (w_bar, u_bar)
//
// Grid: (NT, B*H)   Block: (BLOCK_SIZE = 256)
// Target: gfx942 (MI300X), MFMA bf16 16×16×16
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

    static_assert(T::BT == 64, "This template is for BT=64 only");

    const int i_t  = blockIdx.x;   // chunk index
    const int i_bh = blockIdx.y;   // batch*H index
    const int i_b  = i_bh / kargs.H;
    const int i_h  = i_bh % kargs.H;

    const int tid  = threadIdx.x;
    const int warp_id = tid / T::WARP_SIZE;
    const int lane_id = tid % T::WARP_SIZE;

    const int BT = T::BT;
    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;
    const int stride_t = H;  // g/beta stride along T

    // Base offset into the chunk
    const int chunk_start = i_t * BT;
    const int bos = i_b * kargs.T;  // batch offset in T dimension

    // =====================================================================
    // Shared memory allocation
    // =====================================================================
    extern __shared__ char smem_buf[];

    // Phase 1 layout: g[BT] + beta[BT] + k[BT×K]
    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);                    // BT fp32
    D_ACC*  s_beta = s_g + BT;                                              // BT fp32
    D_ATTN* s_k    = reinterpret_cast<D_ATTN*>(s_beta + BT);               // BT×K bf16

    // Phase 2 layout (reuses s_k region):
    // A[BT×BT] fp32 = 16KB, stored where s_k was (16KB for BT=64,K=128)
    D_ACC*  s_A    = reinterpret_cast<D_ACC*>(s_k);                         // BT×BT fp32

    // =====================================================================
    // Phase 1a: Load g and beta, compute prefix sum
    // =====================================================================
    const D_ACC* g_base    = reinterpret_cast<const D_ACC*>(kargs.ptr_g)
                             + (bos + chunk_start) * H + i_h;
    const D_ACC* beta_base = reinterpret_cast<const D_ACC*>(kargs.ptr_beta)
                             + (bos + chunk_start) * H + i_h;

    // Cooperative load: each thread loads multiple elements
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

    // Prefix sum (Hillis-Steele style, in-place on s_g)
    // BT=64, so we need 6 steps (log2(64) = 6)
    for (int stride = 1; stride < BT; stride <<= 1) {
        // Each thread handles one element
        for (int i = tid; i < BT; i += T::BLOCK_SIZE) {
            if (i >= stride) {
                s_g[i] += s_g[i - stride];
            }
        }
        __syncthreads();
    }

    // Write g_cumsum to HBM
    D_ACC* g_cumsum_base = reinterpret_cast<D_ACC*>(kargs.ptr_g_cumsum)
                           + (bos + chunk_start) * H + i_h;
    for (int i = tid; i < BT; i += T::BLOCK_SIZE) {
        int global_t = chunk_start + i;
        if (global_t < kargs.T) {
            g_cumsum_base[i * H] = s_g[i];
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 1b: Load k[BT, K] into LDS
    // =====================================================================
    const D_ATTN* k_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                           + ((bos + chunk_start) * H + i_h) * K;

    // Cooperative load: BT*K = 64*128 = 8192 bf16 elements
    // With 256 threads loading 8 elements each = 2048 per pass, need 4 passes
    for (int i = tid; i < BT * K; i += T::BLOCK_SIZE) {
        int row = i / K;
        int col = i % K;
        int global_t = chunk_start + row;
        if (global_t < kargs.T) {
            s_k[i] = k_base[row * H * K + col];
        } else {
            s_k[i] = static_cast<D_ATTN>(0);
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 1c: KKT GEMM — A[BT, BT] = k[BT, K] × k^T[K, BT]
    // Using MFMA bf16 16×16×16
    // =====================================================================
    auto mma_kkt = make_tiled_mma<bf16_t, bf16_t, fp32_t>(
        seq<T::KKT_E_M, T::KKT_E_N, T::KKT_E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    // Build layouts for reading k from LDS as A-operand and B-operand (transposed)
    auto p_coord = make_tuple(number<lane_id>{}, number<warp_id>{});

    auto u_ka = mma_kkt.layout_a(make_tuple(number<K>{}, number<1>{}), p_coord);
    auto u_kb = mma_kkt.layout_b(make_tuple(number<K>{}, number<1>{}), p_coord);

    auto s_k_handle = make_smem(reinterpret_cast<D_ATTN*>(s_k));

    // Load k fragments for A operand (row-major: k[BT, K])
    auto v_ka = s_k_handle.load<T::VEC_KV>(u_ka);

    // For B operand we need k^T[K, BT] — but k is stored as [BT, K] in LDS.
    // The mfma_adaptor_swap_ab handles the transposition in the MFMA layout.
    // We load k again using the B-operand layout which reads in transposed order.
    auto v_kb = s_k_handle.load<T::VEC_KV>(u_kb);

    // Initialize accumulator and compute KKT
    auto v_A = mma_kkt.template init_c<D_ACC>();
    v_A = mma_kkt(v_ka, v_kb, v_A);

    __syncthreads();

    // =====================================================================
    // Phase 1d: Apply gate scaling and lower-triangular mask to A
    // Store result to s_A[BT×BT] in LDS (fp32)
    // =====================================================================
    // A[s,r] = beta[s] * exp(g_cumsum[s] - g_cumsum[r]) * kkt[s,r]  if s > r
    //        = 0                                                       if s <= r

    // The MFMA result v_A is distributed across threads according to the C layout.
    // We need to extract (row, col) for each element this thread owns.
    auto u_c = mma_kkt.layout_c(make_tuple(number<BT>{}, number<1>{}), p_coord);

    // Store scaled A to LDS
    // Each thread iterates over its y-dim elements of the C tile
    auto y_shape_c = mma_kkt.y_shape_c();
    constexpr int c_elems = decltype(y_shape_c)::size();

    static_for<c_elems>([&](auto i) {
        // Compute the global row/col for this C element
        // The layout maps (y_idx) → linear offset in the [BT, BT] matrix
        int offset = u_c(number<i>{});
        int row = offset / BT;
        int col = offset % BT;

        D_ACC val = v_A[i];

        if (row > col && row < BT && col < BT) {
            D_ACC g_diff = s_g[row] - s_g[col];
            D_ACC beta_s = s_beta[row];
            val = beta_s * __expf(g_diff) * val;
        } else {
            val = 0.0f;
        }
        s_A[row * BT + col] = val;
    });
    __syncthreads();

    // =====================================================================
    // Phase 2a: Triangular inverse — 4 × 16×16 forward substitution
    //
    // The matrix (I + A) is unit lower triangular. We invert each 16×16
    // diagonal block independently, then merge via Schur complement.
    //
    // Following solve_tril.py: merge_16x16_to_64x64_inverse_kernel
    // =====================================================================

    // Each warp handles one 16×16 diagonal block (4 warps, 4 blocks)
    // Block b covers rows [b*16, (b+1)*16) and cols [b*16, (b+1)*16)
    // Result: C_ii = (I + A_ii)^{-1}

    // Forward substitution within 16×16 block (serial, per-warp)
    // C starts as -A (strict lower tri), then accumulates row by row
    // C += I at the end

    // Each warp owns 16×16/64 = 4 elements of the block
    // But for simplicity, use scalar code: each lane processes part of each row

    int blk = warp_id;  // which 16×16 block this warp handles
    int blk_row_start = blk * 16;

    // Local 16×16 block in registers (each lane holds part)
    // We'll use s_A in-place: the diagonal blocks are at s_A[(blk*16+r)*BT + blk*16+c]

    // Forward substitution: for row i = 2..15
    //   a[i,:] = -A[i,:]
    //   a[i,:] += sum_{j<i} a[j,:] * a[i,j]  (dot product)
    // Then C += I
    for (int i = 2; i < 16; i++) {
        // Each thread handles a subset of the 16 columns
        for (int c = lane_id; c < 16; c += T::WARP_SIZE) {
            if (c >= i) continue;

            int global_row = blk_row_start + i;
            int global_col = blk_row_start + c;
            if (global_row >= BT) continue;

            D_ACC neg_a = -s_A[global_row * BT + global_col];

            // Accumulate: sum over j in [c+1, i-1] of A[i,j] * C[j,c]
            // where C[j,c] has already been computed (j < i)
            D_ACC acc = neg_a;
            for (int j = c + 1; j < i; j++) {
                int j_global = blk_row_start + j;
                acc += (-s_A[global_row * BT + j_global]) *
                       s_A[j_global * BT + global_col];
            }
            s_A[global_row * BT + global_col] = acc;
        }
        // Warp-level sync (all lanes in the warp must see updated s_A)
        __syncthreads();
    }

    // Add identity: C[i,i] = 1
    for (int i = lane_id; i < 16; i += T::WARP_SIZE) {
        int r = blk_row_start + i;
        if (r < BT) {
            s_A[r * BT + r] += 1.0f;
        }
    }
    // Zero upper triangle within the block
    for (int idx = lane_id; idx < 16 * 16; idx += T::WARP_SIZE) {
        int r = idx / 16;
        int c = idx % 16;
        int gr = blk_row_start + r;
        int gc = blk_row_start + c;
        if (gr < BT && gc < BT && r < c) {
            s_A[gr * BT + gc] = 0.0f;
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 2b: Schur complement merge — build full 64×64 inverse
    //
    // Following merge_16x16_to_64x64_inverse_kernel:
    //   C_21 = -C_22 @ A_21 @ C_11
    //   C_32 = -C_33 @ A_32 @ C_22
    //   C_43 = -C_44 @ A_43 @ C_33
    //   C_31 = -C_33 @ (A_31 @ C_11 + A_32 @ C_21)
    //   C_42 = -C_44 @ (A_42 @ C_22 + A_43 @ C_32)
    //   C_41 = -C_44 @ (A_41 @ C_11 + A_42 @ C_21 + A_43 @ C_31)
    //
    // Each off-diagonal block is [16,16]. We need to compute 6 such blocks.
    // The A_ij sub-blocks are in s_A (the original A, now partially overwritten
    // by the diagonal C_ii blocks). We need the ORIGINAL A_ij values.
    //
    // Strategy: Before the forward substitution, we should have saved the
    // off-diagonal blocks. However, since we only modified the diagonal blocks
    // in-place, the off-diagonal A_ij blocks are still intact in s_A.
    //
    // Actually, the forward substitution only touched s_A at positions within
    // the diagonal blocks [blk*16..(blk+1)*16, blk*16..(blk+1)*16].
    // The off-diagonal blocks are untouched. So A_21, A_31, etc. are still valid.
    //
    // For the matmuls, we use scalar code (16×16 is too small for efficient MFMA
    // tiling with 4 warps). Each block of threads handles one matmul.
    // =====================================================================

    // Helper lambda: 16×16 matmul C = X @ Y, result in s_A[r_start*BT+c_start]
    // X is at s_A[xr*BT + xc], Y at s_A[yr*BT + yc]
    // This is done cooperatively by all 256 threads

    // Compute C_21 = -C_22 @ A_21 @ C_11
    // First: tmp = A_21 @ C_11, then C_21 = -C_22 @ tmp
    // A_21 at rows [16,32), cols [0,16)
    // C_11 at rows [0,16), cols [0,16)
    // C_22 at rows [16,32), cols [16,32)

    // We need temporary storage for 16×16 matmul results.
    // Use a portion of LDS beyond s_A for temporaries.
    // s_A occupies BT*BT*4 = 16KB. We have ~48KB remaining.
    D_ACC* s_tmp = s_A + BT * BT;  // temporary 16×16 = 1KB

    // Cooperative 16×16 matmul: each thread computes one or more output elements
    // Total elements: 256 = 16*16, so each of 256 threads handles 1 element
    auto matmul_16x16 = [&](int xr, int xc, int yr, int yc, int dr, int dc,
                            D_ACC sign) {
        // Output block at (dr, dc), each 16×16
        for (int idx = tid; idx < 16 * 16; idx += T::BLOCK_SIZE) {
            int r = idx / 16;
            int c = idx % 16;
            D_ACC acc = 0.0f;
            for (int j = 0; j < 16; j++) {
                acc += s_A[(xr + r) * BT + (xc + j)] *
                       s_A[(yr + j) * BT + (yc + c)];
            }
            s_tmp[idx] = sign * acc;
        }
        __syncthreads();
        // Copy result to s_A
        for (int idx = tid; idx < 16 * 16; idx += T::BLOCK_SIZE) {
            int r = idx / 16;
            int c = idx % 16;
            s_A[(dr + r) * BT + (dc + c)] = s_tmp[idx];
        }
        __syncthreads();
    };

    auto matmul_16x16_to_tmp = [&](int xr, int xc, int yr, int yc, D_ACC sign) {
        for (int idx = tid; idx < 16 * 16; idx += T::BLOCK_SIZE) {
            int r = idx / 16;
            int c = idx % 16;
            D_ACC acc = 0.0f;
            for (int j = 0; j < 16; j++) {
                acc += s_A[(xr + r) * BT + (xc + j)] *
                       s_A[(yr + j) * BT + (yc + c)];
            }
            s_tmp[idx] = sign * acc;
        }
        __syncthreads();
    };

    auto matmul_tmp_by = [&](int yr, int yc, int dr, int dc, D_ACC sign) {
        // result = sign * s_tmp @ Y, store at (dr, dc) in s_A
        // Need another temp buffer
        D_ACC* s_tmp2 = s_tmp + 16 * 16;
        for (int idx = tid; idx < 16 * 16; idx += T::BLOCK_SIZE) {
            int r = idx / 16;
            int c = idx % 16;
            D_ACC acc = 0.0f;
            for (int j = 0; j < 16; j++) {
                acc += s_tmp[r * 16 + j] *
                       s_A[(yr + j) * BT + (yc + c)];
            }
            s_tmp2[idx] = sign * acc;
        }
        __syncthreads();
        for (int idx = tid; idx < 16 * 16; idx += T::BLOCK_SIZE) {
            int r = idx / 16;
            int c = idx % 16;
            s_A[(dr + r) * BT + (dc + c)] = s_tmp2[idx];
        }
        __syncthreads();
    };

    // C_21 = -C_22 @ A_21 @ C_11
    // Step 1: tmp = A_21 @ C_11
    matmul_16x16_to_tmp(16, 0, 0, 0, 1.0f);    // tmp = A_21 @ C_11
    matmul_tmp_by(16, 16, 16, 0, -1.0f);         // C_21 = -C_22 @ tmp

    // C_32 = -C_33 @ A_32 @ C_22
    matmul_16x16_to_tmp(32, 16, 16, 16, 1.0f);  // tmp = A_32 @ C_22
    matmul_tmp_by(32, 32, 32, 16, -1.0f);         // C_32 = -C_33 @ tmp

    // C_43 = -C_44 @ A_43 @ C_33
    matmul_16x16_to_tmp(48, 32, 32, 32, 1.0f);  // tmp = A_43 @ C_33
    matmul_tmp_by(48, 48, 48, 32, -1.0f);         // C_43 = -C_44 @ tmp

    // C_31 = -C_33 @ (A_31 @ C_11 + A_32 @ C_21)
    // tmp = A_31 @ C_11
    matmul_16x16_to_tmp(32, 0, 0, 0, 1.0f);
    // tmp2_buf = A_32 @ C_21
    D_ACC* s_tmp2 = s_tmp + 16 * 16;
    for (int idx = tid; idx < 16 * 16; idx += T::BLOCK_SIZE) {
        int r = idx / 16;
        int c = idx % 16;
        D_ACC acc = 0.0f;
        for (int j = 0; j < 16; j++) {
            acc += s_A[(32 + r) * BT + (16 + j)] *
                   s_A[(16 + j) * BT + c];
        }
        s_tmp[idx] += acc;  // tmp += A_32 @ C_21
    }
    __syncthreads();
    matmul_tmp_by(32, 32, 32, 0, -1.0f);  // C_31 = -C_33 @ tmp

    // C_42 = -C_44 @ (A_42 @ C_22 + A_43 @ C_32)
    matmul_16x16_to_tmp(48, 16, 16, 16, 1.0f);  // tmp = A_42 @ C_22
    for (int idx = tid; idx < 16 * 16; idx += T::BLOCK_SIZE) {
        int r = idx / 16;
        int c = idx % 16;
        D_ACC acc = 0.0f;
        for (int j = 0; j < 16; j++) {
            acc += s_A[(48 + r) * BT + (32 + j)] *
                   s_A[(32 + j) * BT + (16 + c)];
        }
        s_tmp[idx] += acc;  // tmp += A_43 @ C_32
    }
    __syncthreads();
    matmul_tmp_by(48, 48, 48, 16, -1.0f);  // C_42 = -C_44 @ tmp

    // C_41 = -C_44 @ (A_41 @ C_11 + A_42 @ C_21 + A_43 @ C_31)
    matmul_16x16_to_tmp(48, 0, 0, 0, 1.0f);  // tmp = A_41 @ C_11
    for (int idx = tid; idx < 16 * 16; idx += T::BLOCK_SIZE) {
        int r = idx / 16;
        int c = idx % 16;
        D_ACC acc = 0.0f;
        for (int j = 0; j < 16; j++) {
            acc += s_A[(48 + r) * BT + (16 + j)] *
                   s_A[(16 + j) * BT + c];
            acc += s_A[(48 + r) * BT + (32 + j)] *
                   s_A[(32 + j) * BT + c];
        }
        s_tmp[idx] += acc;  // tmp += A_42 @ C_21 + A_43 @ C_31
    }
    __syncthreads();
    matmul_tmp_by(48, 48, 48, 0, -1.0f);  // C_41 = -C_44 @ tmp

    // Now s_A contains C = (I + A)^{-1}, the full 64×64 inverse.

    // =====================================================================
    // Phase 2c: Compute w_bar = C @ (k * beta * exp(g_cumsum))
    // and       u_bar = C @ (v * beta)
    //
    // Following recompute_w_u_fwd_kernel:
    //   w = A @ (k * beta * exp(g_cumsum))   where A here = C = (I+A)^{-1}
    //   u = A @ (v * beta)
    //
    // C is [BT, BT] in s_A. We load k/v subtiles from HBM, scale, and GEMM.
    // Process K in 64-wide subtiles, V in 64-wide subtiles.
    // =====================================================================

    // Output pointers
    D_ATTN* w_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_w_bar)
                         + ((bos + chunk_start) * H + i_h) * K;
    D_ATTN* u_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_u_bar)
                         + ((bos + chunk_start) * H + i_h) * V;

    // Reuse LDS region beyond s_A for k/v subtiles
    // s_A = BT*BT fp32 = 16KB
    // k_subtile right after: BT*64 bf16 = 8KB
    D_ATTN* s_kv_sub = reinterpret_cast<D_ATTN*>(s_A + BT * BT);

    // --- Compute u_bar = C @ (v * beta) ---
    const D_ATTN* v_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_v)
                           + ((bos + chunk_start) * H + i_h) * V;

    for (int iv = 0; iv < T::N_V_ITERS; iv++) {
        int v_offset = iv * T::BV_SUB;

        // Load v subtile [BT, BV_SUB=64] into LDS, scaled by beta
        for (int i = tid; i < BT * T::BV_SUB; i += T::BLOCK_SIZE) {
            int row = i / T::BV_SUB;
            int col = i % T::BV_SUB;
            int global_t = chunk_start + row;
            if (global_t < kargs.T) {
                D_ACC v_val = static_cast<D_ACC>(v_base[row * H * V + v_offset + col]);
                D_ACC scaled = v_val * s_beta[row];
                s_kv_sub[i] = static_cast<D_ATTN>(scaled);
            } else {
                s_kv_sub[i] = static_cast<D_ATTN>(0);
            }
        }
        __syncthreads();

        // GEMM: u_bar_subtile[BT, 64] = C[BT, BT] @ v_scaled[BT, 64]
        // Scalar implementation (C in fp32 LDS, v_scaled in bf16 LDS)
        for (int idx = tid; idx < BT * T::BV_SUB; idx += T::BLOCK_SIZE) {
            int row = idx / T::BV_SUB;
            int col = idx % T::BV_SUB;
            D_ACC acc = 0.0f;
            for (int j = 0; j < BT; j++) {
                acc += s_A[row * BT + j] * static_cast<D_ACC>(s_kv_sub[j * T::BV_SUB + col]);
            }
            int global_t = chunk_start + row;
            if (global_t < kargs.T) {
                u_bar_base[row * H * V + v_offset + col] = static_cast<D_ATTN>(acc);
            }
        }
        __syncthreads();
    }

    // --- Compute w_bar = C @ (k * beta * exp(g_cumsum)) ---
    for (int ik = 0; ik < T::N_K_ITERS; ik++) {
        int k_offset = ik * T::BK_SUB;

        // Load k subtile [BT, BK_SUB=64] into LDS, scaled by beta * exp(g_cumsum)
        for (int i = tid; i < BT * T::BK_SUB; i += T::BLOCK_SIZE) {
            int row = i / T::BK_SUB;
            int col = i % T::BK_SUB;
            int global_t = chunk_start + row;
            if (global_t < kargs.T) {
                D_ACC k_val = static_cast<D_ACC>(k_base[row * H * K + k_offset + col]);
                D_ACC scaled = k_val * s_beta[row] * __expf(s_g[row]);
                s_kv_sub[i] = static_cast<D_ATTN>(scaled);
            } else {
                s_kv_sub[i] = static_cast<D_ATTN>(0);
            }
        }
        __syncthreads();

        // GEMM: w_bar_subtile[BT, 64] = C[BT, BT] @ k_scaled[BT, 64]
        for (int idx = tid; idx < BT * T::BK_SUB; idx += T::BLOCK_SIZE) {
            int row = idx / T::BK_SUB;
            int col = idx % T::BK_SUB;
            D_ACC acc = 0.0f;
            for (int j = 0; j < BT; j++) {
                acc += s_A[row * BT + j] * static_cast<D_ACC>(s_kv_sub[j * T::BK_SUB + col]);
            }
            int global_t = chunk_start + row;
            if (global_t < kargs.T) {
                w_bar_base[row * H * K + k_offset + col] = static_cast<D_ATTN>(acc);
            }
        }
        __syncthreads();
    }
}
