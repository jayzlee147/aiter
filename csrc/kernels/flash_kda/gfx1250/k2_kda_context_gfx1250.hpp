// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 wave32 context-parallel FlashKDA recurrence.
//
// This is the native AOT time-parallel counterpart of the serial V-split
// walk. A context group contains 4, 8, or 16 consecutive C16 chunks.
// Two independent passes form the affine map
//
//   h_out = A_group h_in + b_group,          h: [K,V]
//
// for every group.  A short per-sequence scan overwrites b_group with the
// true incoming state, then all groups replay in parallel and write output.
// The public state remains V-first [N,H,V,K].
//
// Workspace reuse is identical to the established native gfx950 route:
//
//   ws.cs_sin = affine A [context,H,K,K] BF16
//   route-private affine_b = affine b / incoming h [context,H,K,V] FP32
//   ws.cs_beta = activated beta [tile,C] FP32
//
// affine_b aliases ws.cs_u when the legacy tile arena is large enough.  Short
// or highly packed shapes use a checked, route-private tail arena so the C16
// rollback retains its established workspace footprint and offsets.
//
// The ordinary C16 kd/qd/kr/decay/inv/mqk workspace remains live through all
// three phases.  No Triton or Gluon code participates in this path.
#pragma once

constexpr int K2_CONTEXT_GFX1250_GROUP4 = 4;
constexpr int K2_CONTEXT_GFX1250_GROUP8 = 8;
constexpr int K2_CONTEXT_GFX1250_GROUP16 = 16;
typedef bf16 ContextBf16x8 __attribute__((ext_vector_type(8)));
typedef bf16 ContextBf16x2 __attribute__((ext_vector_type(2)));
static_assert(sizeof(ContextBf16x2) == 4,
              "packed replay value load must remain one dword");

// Full-chunk replay owns adjacent BF16 value pairs, but the public tensor
// contract guarantees only element alignment.  A fixed-size builtin memcpy
// retains one dword load on gfx1250 without constructing a potentially
// misaligned or aliasing ContextBf16x2 lvalue over the BF16 allocation.
__device__ __forceinline__ ContextBf16x2
k2_context_gfx1250_load_bf16x2_unaligned(const bf16* source)
{
    ContextBf16x2 packed;
    __builtin_memcpy(&packed, source, sizeof(packed));
    return packed;
}

// The async KR publisher writes one b128 vector at a time.  Padding every
// [C16,C16] tile by one C16 row rotates consecutive tiles by 32 bytes in LDS,
// so both the global->LDS b128 stores and the later ds_load_tr16_b128 reads
// cover all bank groups instead of aliasing the same eight groups.
constexpr int K2_CONTEXT_GFX1250_KR_ASYNC_TILE_STRIDE = C * C + C;
static_assert((D / C) * K2_CONTEXT_GFX1250_KR_ASYNC_TILE_STRIDE ==
                  C * KD_STRIDE,
              "padded async KR must retain the established LDS allocation");

static_assert(WAVE == 32 && C == 16 && D == 128 && BV == 16,
              "gfx1250 context kernels require wave32 C16 K128 V16");
static_assert(K2_CONTEXT_GFX1250_GROUP4 * C == 64,
              "one G4 affine-A map must align one 64-token cs_sin segment");
static_assert(WorkspaceSizes::kCsplitSin == D * D * int(sizeof(bf16)),
              "one context A map must fit one cs_sin slot");
static_assert(4 * K2_CONTEXT_GFX1250_GROUP4 * WorkspaceSizes::kCsplitU ==
                  D * D * int(sizeof(float)),
              "G4 affine b requires the route-private full-map tail");
static_assert(2 * K2_CONTEXT_GFX1250_GROUP8 * WorkspaceSizes::kCsplitU ==
                  D * D * int(sizeof(float)),
              "G8 affine b requires the route-private full-map tail");
static_assert(K2_CONTEXT_GFX1250_GROUP16 * WorkspaceSizes::kCsplitU ==
                  D * D * int(sizeof(float)),
              "sixteen cs_u tiles must fit one context b map");

// The four-wave producer trades half as many independently schedulable
// workgroups for lower V-independent input traffic.  The MI455 crossover is
// not a function of context_head_groups alone: group size, chunks per head,
// and head count all change the balance.  Keep the first production policy to
// the exact dense single-sequence shapes measured with both capture orders.
template <int GroupChunks>
inline bool k2_context_gfx1250_should_use_2vtile_4wave(
    bool varlen,
    int nseq,
    int t_seq,
    int nt_dense,
    int h_count)
{
    if(varlen || nseq != 1 || h_count == 1 || t_seq <= 0 ||
       (t_seq % C) != 0 || nt_dense != t_seq / C)
        return false;

    if constexpr(GroupChunks == K2_CONTEXT_GFX1250_GROUP4)
    {
        switch(nt_dense)
        {
        case 64:
            return h_count == 12;
        case 128:
            return h_count == 2 || h_count == 4 || h_count == 12;
        case 256:
            return h_count == 2 || h_count == 12;
        case 512:
            return h_count == 12;
        default:
            return false;
        }
    }
    else if constexpr(GroupChunks == K2_CONTEXT_GFX1250_GROUP8)
    {
        switch(nt_dense)
        {
        case 128:
            return h_count == 12;
        case 256:
            return h_count == 4 || h_count == 12;
        case 512:
            return h_count == 2 || h_count == 4 || h_count == 12;
        default:
            return false;
        }
    }
    else
    {
        return false;
    }
}

// A[M16,K128] @ state[K128,V16].  State is retained in the native WMMA
// accumulator layout used by the serial walk.
template <int Lda>
__device__ __forceinline__ Acc k2_context_gfx1250_contract_with_state(
    const bf16* a,
    const Acc (&state)[D / C],
    int lane)
{
    const int row = lane & 15;
    Acc result{};
    #pragma unroll
    for(int panel = 0; panel < D / 32; ++panel)
    {
        FragB state_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            state_fragment[i] = f32_to_bf16(state[2 * panel][i]);
            state_fragment[8 + i] =
                f32_to_bf16(state[2 * panel + 1][i]);
        }
        const FragA a_fragment =
            load_row_fragment<bf16, Lda>(a, row, panel * 32, lane);
        result = wmma(a_fragment, state_fragment, result);
    }
    return result;
}

// A[M16,K64] @ state[K64,V16].  Four K16 accumulator tiles concatenate
// lane-locally into the native pair of K32 WMMA right operands.
template <int Lda>
__device__ __forceinline__ Acc k2_context_gfx1250_contract_k64_with_state(
    const bf16* a,
    const Acc (&state)[4],
    int lane)
{
    const int row = lane & 15;
    Acc result{};
    #pragma unroll
    for(int panel = 0; panel < 2; ++panel)
    {
        FragB state_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            state_fragment[i] = f32_to_bf16(state[2 * panel][i]);
            state_fragment[8 + i] =
                f32_to_bf16(state[2 * panel + 1][i]);
        }
        const FragA a_fragment =
            load_row_fragment<bf16, Lda>(a, row, panel * 32, lane);
        result = wmma(a_fragment, state_fragment, result);
    }
    return result;
}

// Contract two independent destination K16 tiles against the same resident
// K64 state.  Keeping both panel-0 A fragments live lets gfx1250 overlap their
// four LDS reads.  Each accumulator still executes panel 0 before panel 1, so
// the WMMA inputs, BF16 conversion points, and FP32 accumulation graph remain
// identical to two calls to k2_context_gfx1250_contract_k64_with_state.
template <int Lda>
__device__ __forceinline__ void
k2_context_gfx1250_contract_k64_pair_fragment_pipeline(
    const bf16* a0,
    const bf16* a1,
    const Acc (&state)[4],
    int lane,
    Acc& result0,
    Acc& result1)
{
    const int row = lane & 15;
    result0 = Acc{};
    result1 = Acc{};
#if defined(__gfx1250__)
    __builtin_amdgcn_sched_barrier(0);
#endif

    FragB state_fragment{};
    #pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        state_fragment[i] = f32_to_bf16(state[0][i]);
        state_fragment[8 + i] = f32_to_bf16(state[1][i]);
    }

    const FragA a0_panel0 =
        load_row_fragment<bf16, Lda>(a0, row, 0, lane);
    const FragA a1_panel0 =
        load_row_fragment<bf16, Lda>(a1, row, 0, lane);
    result0 = wmma(a0_panel0, state_fragment, result0);

    // Issue the first panel-1 fragment while the second panel-0 WMMA is
    // ready.  This is the source-level ordering that the compiler already
    // discovers for the first output pair, made explicit for both pairs.
    const FragA a0_panel1 =
        load_row_fragment<bf16, Lda>(a0, row, 32, lane);
    result1 = wmma(a1_panel0, state_fragment, result1);

    #pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        state_fragment[i] = f32_to_bf16(state[2][i]);
        state_fragment[8 + i] = f32_to_bf16(state[3][i]);
    }
    const FragA a1_panel1 =
        load_row_fragment<bf16, Lda>(a1, row, 32, lane);
    result0 = wmma(a0_panel1, state_fragment, result0);
    result1 = wmma(a1_panel1, state_fragment, result1);
#if defined(__gfx1250__)
    // Preserve the intended DS/MMA issue groups for both output fragments:
    // DS4, MMA1, DS2, MMA2, DS2, MMA1.  The surrounding barriers keep this
    // local schedule from being merged with the caller's independent work.
    __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
    __builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
    __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
    __builtin_amdgcn_sched_barrier(0);
#endif
}

// KR is published as eight independent [C16,K16] row-major tiles on the
// production tiled path.  gfx1250's ds_load_tr16_b128 consumes
// one b128 segment per lane and transposes each 8x8 quadrant across the wave:
//
//   source row = (lane / 16) * 8 + lane % 8
//   source col = (lane / 8 % 2) * 8
//   result[i]  = KR[(lane / 16) * 8 + i][lane % 16]
//
// Thus one transpose load forms exactly the live C16 half of the native K32
// WMMA A fragment.  The upper half remains zero, preserving the established
// WMMA count, ordering, and BF16 rounding points of state_update_k16.
__device__ __forceinline__ Acc k2_context_gfx1250_state_update_tiled(
    const bf16* kr, FragB u, int k_tile, int lane)
{
    constexpr int KrTileElements = C * C;
    const int source_row = (lane >> 4) * 8 + (lane & 7);
    const int source_col = ((lane >> 3) & 1) * 8;
    const int source =
        k_tile * KrTileElements + source_row * C + source_col;

    opus::smem<bf16> tiled_kr(const_cast<bf16*>(kr));
    const auto kr_fragment = tiled_kr.template tr_load<8>(source);
    FragA af{};
    #pragma unroll
    for(int i = 0; i < 8; ++i)
        af[i] = kr_fragment[i];

    Acc acc{};
    return wmma(af, u, acc);
}

// Consumer for the padded KR layout used by the direct async path.  Keep the
// scalar-published tiled rollback above unchanged so an explicit disable does
// not alter its layout contract.
__device__ __forceinline__ Acc
k2_context_gfx1250_state_update_tiled_async_b128(
    const bf16* kr, FragB u, int k_tile, int lane)
{
    const int source_row = (lane >> 4) * 8 + (lane & 7);
    const int source_col = ((lane >> 3) & 1) * 8;
    const int source =
        k_tile * K2_CONTEXT_GFX1250_KR_ASYNC_TILE_STRIDE +
        source_row * C + source_col;

    opus::smem<bf16> tiled_kr(const_cast<bf16*>(kr));
    const auto kr_fragment = tiled_kr.template tr_load<8>(source);
    FragA af{};
    #pragma unroll
    for(int i = 0; i < 8; ++i)
        af[i] = kr_fragment[i];

    Acc acc{};
    return wmma(af, u, acc);
}

// Direct gfx1250 global->LDS b128 publishers.  Source pointers are already
// rebased to one C16 chunk, which keeps the instruction voffset within 32 bits
// even for large packed batches.  All destinations and row pitches are b128
// aligned.  Threads is 128 for the fused two-vtile paired producer, 64 for
// paired A/B, and 32 for replay.
template <int Threads>
__device__ __forceinline__ void k2_context_gfx1250_async_stage_kd_b128(
    const bf16* source, bf16* destination, int thread)
{
    constexpr int Vector = 8;
    constexpr int VectorsPerRow = D / Vector;
    opus::gmem<bf16> global(source);
    #pragma unroll
    for(int vector = thread;
        vector < C * VectorsPerRow;
        vector += Threads)
    {
        const int row = vector / VectorsPerRow;
        const int col8 = vector - row * VectorsPerRow;
        global.template async_load<Vector>(
            destination + row * KD_STRIDE + col8 * Vector,
            vector * Vector);
    }
}

// Publish one row-major K128xK128 affine-A map into the row-padded scan LDS
// arena. The four-wave K-split scan assigns sixteen aligned b128 vectors to
// every thread, while the eight-wave A-pipeline assigns eight. Consumers must
// wait and rendezvous before reading the matrix.
template <int Threads>
__device__ __forceinline__ void
k2_context_gfx1250_async_stage_affine_a_b128(
    const bf16* source, bf16* destination, int thread)
{
    constexpr int Vector = 8;
    constexpr int VectorsPerRow = D / Vector;
    static_assert(Threads == 4 * WAVE || Threads == 8 * WAVE,
                  "affine-A async publisher requires four or eight waves");
    opus::gmem<bf16> global(source);
    #pragma unroll
    for(int vector = thread;
        vector < D * VectorsPerRow;
        vector += Threads)
    {
        const int row = vector / VectorsPerRow;
        const int col8 = vector - row * VectorsPerRow;
        global.template async_load<Vector>(
            destination + row * KD_STRIDE + col8 * Vector,
            vector * Vector);
    }
}

// The eight-wave scan has an exact static assignment: 2048 b128 vectors are
// divided evenly over 256 threads, so every thread always publishes eight
// vectors.  Spell out the fixed trip count instead of relying on the generic
// upper-bound loop.  Current gfx1250 clang otherwise retains EXEC-masked exit
// checks for the final three unrolled trips even though the launch geometry
// guarantees threadIdx.x < 256.
__device__ __forceinline__ void
k2_context_gfx1250_async_stage_affine_a_8w_exact_b128(
    const bf16* source, bf16* destination, int thread)
{
    constexpr int Vector = 8;
    constexpr int VectorsPerRow = D / Vector;
    constexpr int Threads = 8 * WAVE;
    constexpr int Trips = D * VectorsPerRow / Threads;
    static_assert(D * VectorsPerRow == Trips * Threads && Trips == 8,
                  "eight-wave affine-A publisher must have eight exact trips");
    opus::gmem<bf16> global(source);
    #pragma unroll
    for(int trip = 0; trip < Trips; ++trip)
    {
        const int vector = thread + trip * Threads;
        const int row = vector / VectorsPerRow;
        const int col8 = vector - row * VectorsPerRow;
        global.template async_load<Vector>(
            destination + row * KD_STRIDE + col8 * Vector,
            vector * Vector);
    }
}

template <int Threads>
__device__ __forceinline__ void k2_context_gfx1250_async_stage_kr_b128(
    const bf16* source, bf16* destination, int thread)
{
    constexpr int Vector = 8;
    constexpr int VectorsPerRow = D / Vector;
    constexpr int VectorsPerTile = C / Vector;
    opus::gmem<bf16> global(source);
    #pragma unroll
    for(int vector = thread;
        vector < C * VectorsPerRow;
        vector += Threads)
    {
        const int row = vector / VectorsPerRow;
        const int col8 = vector - row * VectorsPerRow;
        const int k_tile = col8 / VectorsPerTile;
        const int k_inner8 =
            (col8 - k_tile * VectorsPerTile) * Vector;
        const int lds =
            k_tile * K2_CONTEXT_GFX1250_KR_ASYNC_TILE_STRIDE +
            row * C + k_inner8;
        global.template async_load<Vector>(
            destination + lds, vector * Vector);
    }
}

template <int Threads>
__device__ __forceinline__ void k2_context_gfx1250_async_stage_factor_b128(
    const bf16* source, bf16* destination, int thread)
{
    constexpr int Vector = 8;
    constexpr int VectorsPerRow = C / Vector;
    static_assert(Threads == WAVE || Threads == 2 * WAVE,
                  "factor async publisher supports one or two waves");
    opus::gmem<bf16> global(source);
    if(thread < C * VectorsPerRow)
    {
        // Each half-wave publishes the same 16 rows and a different b128
        // half.  FACTOR_STRIDE=24 BF16 advances by three bank groups per row,
        // a permutation of all 16 groups.  Flattened q=thread would instead
        // make each half-wave cover only eight rows and introduce conflicts.
        const int vector =
            (thread & (C - 1)) * VectorsPerRow + (thread >> 4);
        const int row = vector / VectorsPerRow;
        const int col8 = vector - row * VectorsPerRow;
        global.template async_load<Vector>(
            destination + row * FACTOR_STRIDE + col8 * Vector,
            vector * Vector);
    }
}

template <int Threads, int Elements>
__device__ __forceinline__ void k2_context_gfx1250_async_stage_f32_b128(
    const float* source, float* destination, int thread)
{
    constexpr int Vector = 4;
    static_assert(Elements % Vector == 0,
                  "async f32 staging requires whole b128 vectors");
    opus::gmem<float> global(source);
    #pragma unroll
    for(int vector = thread;
        vector < Elements / Vector;
        vector += Threads)
    {
        global.template async_load<Vector>(
            destination + vector * Vector, vector * Vector);
    }
}

// The four-wave/two-vtile producer gives every wave four matrix publications.
// Distribute the singleton operands across three different waves so its
// critical publisher retires five async operations instead of concentrating
// factor, decay, and beta on one seven-operation wave.
__device__ __forceinline__ void
k2_context_gfx1250_async_stage_shared_factors_4wave_b128(
    const bf16* factor_source,
    bf16* factor_destination,
    const float* decay_source,
    float* decay_destination,
    const float* beta_source,
    float* beta_destination,
    int thread)
{
    static_assert(C * (C / 8) == WAVE && D / 4 == WAVE && C / 4 == 4,
                  "four-wave factor publisher geometry changed");
    if(thread < WAVE)
    {
        k2_context_gfx1250_async_stage_factor_b128<WAVE>(
            factor_source, factor_destination, thread);
    }
    else if(thread < 2 * WAVE)
    {
        k2_context_gfx1250_async_stage_f32_b128<WAVE, D>(
            decay_source, decay_destination, thread - WAVE);
    }
    else if(thread < 2 * WAVE + C / 4)
    {
        k2_context_gfx1250_async_stage_f32_b128<WAVE, C>(
            beta_source, beta_destination, thread - 2 * WAVE);
    }
}

// Single-wave replay has an exact compile-time assignment: every lane
// publishes eight matrix vectors, one factor vector, one decay vector, and
// lanes zero through three publish one beta vector.  Spell those fixed trip
// counts out directly so the backend does not need an EXEC-masked loop exit.
// The per-lane vector order, source offsets, LDS layout, and async payload are
// identical to the generic publishers used by the two-wave paired producer.
__device__ __forceinline__ void
k2_context_gfx1250_replay_exact_stage_kd_b128(
    const bf16* source, bf16* destination, int lane)
{
    constexpr int Vector = 8;
    constexpr int VectorsPerRow = D / Vector;
    constexpr int Trips = C * VectorsPerRow / WAVE;
    static_assert(C * VectorsPerRow == Trips * WAVE,
                  "replay KD publisher must have exact wave trips");
    opus::gmem<bf16> global(source);
    #pragma unroll
    for(int trip = 0; trip < Trips; ++trip)
    {
        const int vector = lane + trip * WAVE;
        const int row = vector / VectorsPerRow;
        const int col8 = vector - row * VectorsPerRow;
        global.template async_load<Vector>(
            destination + row * KD_STRIDE + col8 * Vector,
            vector * Vector);
    }
}

__device__ __forceinline__ void
k2_context_gfx1250_replay_exact_stage_kr_b128(
    const bf16* source, bf16* destination, int lane)
{
    constexpr int Vector = 8;
    constexpr int VectorsPerRow = D / Vector;
    constexpr int VectorsPerTile = C / Vector;
    constexpr int Trips = C * VectorsPerRow / WAVE;
    static_assert(C * VectorsPerRow == Trips * WAVE,
                  "replay KR publisher must have exact wave trips");
    opus::gmem<bf16> global(source);
    #pragma unroll
    for(int trip = 0; trip < Trips; ++trip)
    {
        const int vector = lane + trip * WAVE;
        const int row = vector / VectorsPerRow;
        const int col8 = vector - row * VectorsPerRow;
        const int k_tile = col8 / VectorsPerTile;
        const int k_inner8 =
            (col8 - k_tile * VectorsPerTile) * Vector;
        const int lds =
            k_tile * K2_CONTEXT_GFX1250_KR_ASYNC_TILE_STRIDE +
            row * C + k_inner8;
        global.template async_load<Vector>(
            destination + lds, vector * Vector);
    }
}

__device__ __forceinline__ void
k2_context_gfx1250_replay_exact_stage_factor_b128(
    const bf16* source, bf16* destination, int lane)
{
    constexpr int Vector = 8;
    constexpr int VectorsPerRow = C / Vector;
    static_assert(C * VectorsPerRow == WAVE,
                  "replay factor publisher must assign one vector per lane");
    opus::gmem<bf16> global(source);
    const int vector =
        (lane & (C - 1)) * VectorsPerRow + (lane >> 4);
    const int row = vector / VectorsPerRow;
    const int col8 = vector - row * VectorsPerRow;
    global.template async_load<Vector>(
        destination + row * FACTOR_STRIDE + col8 * Vector,
        vector * Vector);
}

__device__ __forceinline__ void
k2_context_gfx1250_replay_exact_stage_decay_b128(
    const float* source, float* destination, int lane)
{
    constexpr int Vector = 4;
    static_assert(D / Vector == WAVE,
                  "replay decay publisher must assign one vector per lane");
    opus::gmem<float> global(source);
    global.template async_load<Vector>(
        destination + lane * Vector, lane * Vector);
}

__device__ __forceinline__ void
k2_context_gfx1250_replay_exact_stage_beta_b128(
    const float* source, float* destination, int lane)
{
    constexpr int Vector = 4;
    static_assert(C / Vector == 4,
                  "replay beta publisher expects four b128 vectors");
    opus::gmem<float> global(source);
    if(lane < C / Vector)
    {
        global.template async_load<Vector>(
            destination + lane * Vector, lane * Vector);
    }
}

// The HIP host pass does not define __gfx1250__, so opus exposes only its
// combined-counter fallback while parsing this header.  Route through small
// dual-pass wrappers: device gfx1250 receives the split counters, and the
// never-executed host body remains well-formed.
__device__ __forceinline__ void
k2_context_gfx1250_wait_async_b128()
{
#if defined(__gfx1250__)
    opus::s_wait_asynccnt(opus::number<0>{});
#else
    opus::s_waitcnt_vmcnt(opus::number<0>{});
#endif
}

template <int Outstanding>
__device__ __forceinline__ void
k2_context_gfx1250_wait_async_b128_count()
{
#if defined(__gfx1250__)
    opus::s_wait_asynccnt(opus::number<Outstanding>{});
#else
    opus::s_waitcnt_vmcnt(opus::number<Outstanding>{});
#endif
}

template <int Outstanding>
__device__ __forceinline__ void
k2_context_gfx1250_wait_global_loads()
{
#if defined(__gfx1250__)
    opus::s_wait_loadcnt(opus::number<Outstanding>{});
#else
    opus::s_waitcnt_vmcnt(opus::number<Outstanding>{});
#endif
}

__device__ __forceinline__ void
k2_context_gfx1250_wait_lds()
{
#if defined(__gfx1250__)
    opus::s_wait_dscnt(opus::number<0>{});
#else
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
#endif
}

// Synchronize an LDS producer/consumer phase without attaching a global
// memory fence to the barrier.  Both the post-WMMA exchange and the opt-in
// fused-publication experiment use this when their dependency contract is
// LDS-local: all LDS stores retire before signal, while unrelated global-
// memory traffic remains free to overlap the barrier and post-barrier work.
__device__ __forceinline__ void
k2_context_gfx1250_sync_lds_only()
{
#if defined(__gfx1250__)
    k2_context_gfx1250_wait_lds();
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
#else
    __syncthreads();
#endif
}

template <int Outstanding>
__device__ __forceinline__ void
k2_context_gfx1250_wait_tensor()
{
#if defined(__gfx1250__)
    opus::s_wait_tensorcnt(opus::number<Outstanding>{});
#else
    // The HIP host/device fallback pass has no split tensor counter.  This
    // body is never selected by a gfx1250 code object; it only keeps the
    // dual-pass header well-formed, matching the async/DS wrappers above.
    opus::s_waitcnt_vmcnt(opus::number<0>{});
#endif
}

template <bool Varlen>
__global__ __launch_bounds__(2 * WAVE)
void k2_kda_context_gfx1250_paired_ab_async_b128_kernel(
    const bf16* __restrict__ value,
    const bf16* __restrict__ ws_kd,
    const bf16* __restrict__ ws_kr,
    const float* __restrict__ ws_decay,
    const bf16* __restrict__ ws_inv,
    const float* __restrict__ beta_cache,
    bf16* __restrict__ affine_a,
    float* __restrict__ affine_b,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ tile_prefix,
    const int* __restrict__ context_prefix,
    int total_context_groups,
    int total_tiles,
    int t_seq,
    int h_count,
    int nseq,
    int nt_dense,
    int group_chunks_max)
{
    if(group_chunks_max != K2_CONTEXT_GFX1250_GROUP4 &&
       group_chunks_max != K2_CONTEXT_GFX1250_GROUP8 &&
       group_chunks_max != K2_CONTEXT_GFX1250_GROUP16)
        return;

    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread & (WAVE - 1);
    const int wave = thread / WAVE;
    const bool form_affine_a = wave == 1;
    const int context_head = static_cast<int>(blockIdx.x);
    const int v_tile = static_cast<int>(blockIdx.y);
    const int global_context = context_head / h_count;
    const int head = context_head - global_context * h_count;
    const int v0 = v_tile * BV;

    int seq = 0;
    int local_context = 0;
    int bos = 0;
    int eos = 0;
    int tile0 = 0;
    int chunks = 0;
    if constexpr(Varlen)
    {
        if(nseq <= 0 || global_context >= context_prefix[nseq])
            return;
        int lo = 0;
        int hi = nseq;
        while(hi - lo > 1)
        {
            const int mid = (lo + hi) >> 1;
            if(context_prefix[mid] <= global_context)
                lo = mid;
            else
                hi = mid;
        }
        seq = lo;
        local_context = global_context - context_prefix[seq];
        bos = cu_seqlens[seq];
        eos = cu_seqlens[seq + 1];
        tile0 = tile_prefix[seq];
        chunks = tile_prefix[seq + 1] - tile0;
    }
    else
    {
        const int contexts_per_sequence =
            (nt_dense + group_chunks_max - 1) / group_chunks_max;
        if(contexts_per_sequence <= 0 ||
           global_context >= nseq * contexts_per_sequence)
            return;
        seq = global_context / contexts_per_sequence;
        local_context = global_context - seq * contexts_per_sequence;
        bos = seq * t_seq;
        eos = bos + t_seq;
        chunks = nt_dense;
    }

    const int first_chunk = local_context * group_chunks_max;
    int group_chunks = chunks - first_chunk;
    group_chunks = group_chunks < group_chunks_max
        ? group_chunks
        : group_chunks_max;
    if(group_chunks <= 0)
        return;
    const int64_t context_slab =
        (static_cast<int64_t>(global_context) * h_count + head) * D * D;

    alignas(16) __shared__ bf16 kd[C * KD_STRIDE];
    alignas(16) __shared__ bf16
        kr[(D / C) * K2_CONTEXT_GFX1250_KR_ASYNC_TILE_STRIDE];
    alignas(16) __shared__ bf16 inv[C * FACTOR_STRIDE];
    alignas(16) __shared__ float decay[D];
    alignas(16) __shared__ float beta_act[C];

    Acc state[D / C];
    Acc identity_fragment{};
    const int identity_v_local = lane & 15;
    const int identity_lane_group = lane >> 4;
    const float identity_value =
        form_affine_a && (identity_v_local >> 3) == identity_lane_group
            ? 1.0f
            : 0.0f;
    #pragma unroll
    for(int i = 0; i < 8; ++i)
        identity_fragment[i] =
            (identity_v_local & 7) == i ? identity_value : 0.0f;
    #pragma unroll
    for(int k_tile = 0; k_tile < D / C; ++k_tile)
    {
        const Acc zero{};
        state[k_tile] = v_tile == k_tile ? identity_fragment : zero;
    }

    for(int group_chunk = 0; group_chunk < group_chunks; ++group_chunk)
    {
        const int chunk = first_chunk + group_chunk;
        const int ht = Varlen
            ? head * total_tiles + tile0 + chunk
            : (seq * h_count + head) * nt_dense + chunk;
        const int token0 = bos + chunk * C;
        int active = eos - token0;
        active = active < C ? active : C;
        const int64_t matrix_offset = static_cast<int64_t>(ht) * C * D;
        const int64_t factor_offset = static_cast<int64_t>(ht) * C * C;

        k2_context_gfx1250_async_stage_kd_b128<2 * WAVE>(
            ws_kd + matrix_offset, kd, thread);
        k2_context_gfx1250_async_stage_kr_b128<2 * WAVE>(
            ws_kr + matrix_offset, kr, thread);
        k2_context_gfx1250_async_stage_factor_b128<2 * WAVE>(
            ws_inv + factor_offset, inv, thread);
        k2_context_gfx1250_async_stage_f32_b128<2 * WAVE, D>(
            ws_decay + static_cast<int64_t>(ht) * D, decay, thread);
        k2_context_gfx1250_async_stage_f32_b128<2 * WAVE, C>(
            beta_cache + static_cast<int64_t>(ht) * C, beta_act, thread);
        k2_context_gfx1250_wait_async_b128();
        __syncthreads();

        const Acc prediction =
            k2_context_gfx1250_contract_with_state<KD_STRIDE>(
                kd, state, lane);

        FragB residual_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int token = (lane >> 4) * 8 + i;
            const int v_local = lane & 15;
            float v = 0.0f;
            if(!form_affine_a && token < active)
            {
                const int64_t offset =
                    (static_cast<int64_t>(token0 + token) * h_count + head) *
                        D + v0 + v_local;
                v = bf16_to_f32(value[offset]);
            }
            residual_fragment[i] =
                f32_to_bf16((v - prediction[i]) * beta_act[token]);
            residual_fragment[8 + i] = f32_to_bf16(0.0f);
        }

        const int factor_row = lane & 15;
        const int lane_group = lane >> 4;
        FragA inv_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int col = lane_group * 8 + i;
            inv_fragment[i] = inv[factor_row * FACTOR_STRIDE + col];
            inv_fragment[8 + i] = f32_to_bf16(0.0f);
        }
        Acc zero{};
        const Acc u_acc = wmma(inv_fragment, residual_fragment, zero);
        FragB u_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            u_fragment[i] = f32_to_bf16(u_acc[i]);
            u_fragment[8 + i] = f32_to_bf16(0.0f);
        }

        #pragma unroll
        for(int k_tile = 0; k_tile < D / C; ++k_tile)
        {
            const Acc update =
                k2_context_gfx1250_state_update_tiled_async_b128(
                    kr, u_fragment, k_tile, lane);
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int k = k_tile * C + lane_group * 8 + i;
                state[k_tile][i] =
                    state[k_tile][i] * decay[k] + update[i];
            }
        }

        // LDS reads are independently tracked on gfx1250.  Retire them before
        // either wave can refill the single arena, then rendezvous both waves.
        k2_context_gfx1250_wait_lds();
        __syncthreads();
    }

    const int64_t lane_store_offset =
        context_slab +
        static_cast<int64_t>((lane >> 4) * 8) * D +
        v0 + (lane & 15);
    if(form_affine_a)
    {
        #pragma unroll
        for(int k_tile = 0; k_tile < D / C; ++k_tile)
        {
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int row = k_tile * C + i;
                affine_a[lane_store_offset + static_cast<int64_t>(row) * D] =
                    f32_to_bf16(state[k_tile][i]);
            }
        }
    }
    else
    {
        #pragma unroll
        for(int k_tile = 0; k_tile < D / C; ++k_tile)
        {
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int row = k_tile * C + i;
                affine_b[lane_store_offset + static_cast<int64_t>(row) * D] =
                    state[k_tile][i];
            }
        }
    }
}

// Experimental four-wave producer for two adjacent V16 tiles.  The wave
// mapping is deliberately affine-mode adjacent:
//
//   wave = vhalf * 2 + affine_mode
//
// so waves 0/1 reproduce the established b/A arithmetic for vtile 2*y and
// waves 2/3 reproduce it for vtile 2*y+1.  All four consumers share one
// publication of the V-independent chunk operands.  No state fragment crosses
// a wave boundary, and every wave retains the exact WMMA/FP32 recurrence order
// of k2_kda_context_gfx1250_paired_ab_async_b128_kernel.
template <bool Varlen>
__global__ __launch_bounds__(4 * WAVE)
void k2_kda_context_gfx1250_paired_ab_async_b128_2vtile_4wave_kernel(
    const bf16* __restrict__ value,
    const bf16* __restrict__ ws_kd,
    const bf16* __restrict__ ws_kr,
    const float* __restrict__ ws_decay,
    const bf16* __restrict__ ws_inv,
    const float* __restrict__ beta_cache,
    bf16* __restrict__ affine_a,
    float* __restrict__ affine_b,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ tile_prefix,
    const int* __restrict__ context_prefix,
    int total_context_groups,
    int total_tiles,
    int t_seq,
    int h_count,
    int nseq,
    int nt_dense,
    int group_chunks_max)
{
    static_assert(V_TILES == 8 && V_TILES % 2 == 0,
                  "two-vtile producer requires eight even V16 tiles");
    if(group_chunks_max != K2_CONTEXT_GFX1250_GROUP4 &&
       group_chunks_max != K2_CONTEXT_GFX1250_GROUP8)
        return;

    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread & (WAVE - 1);
    const int wave = thread / WAVE;
    const int vhalf = wave >> 1;
    const int affine_mode = wave & 1;
    const bool form_affine_a = affine_mode != 0;
    const int context_head = static_cast<int>(blockIdx.x);
    const int v_block = static_cast<int>(blockIdx.y);
    const int global_context = context_head / h_count;
    const int head = context_head - global_context * h_count;
    const int v_tile = v_block * 2 + vhalf;
    const int v0 = v_tile * BV;

    int seq = 0;
    int local_context = 0;
    int bos = 0;
    int eos = 0;
    int tile0 = 0;
    int chunks = 0;
    if constexpr(Varlen)
    {
        if(nseq <= 0 || global_context >= context_prefix[nseq])
            return;
        int lo = 0;
        int hi = nseq;
        while(hi - lo > 1)
        {
            const int mid = (lo + hi) >> 1;
            if(context_prefix[mid] <= global_context)
                lo = mid;
            else
                hi = mid;
        }
        seq = lo;
        local_context = global_context - context_prefix[seq];
        bos = cu_seqlens[seq];
        eos = cu_seqlens[seq + 1];
        tile0 = tile_prefix[seq];
        chunks = tile_prefix[seq + 1] - tile0;
    }
    else
    {
        const int contexts_per_sequence =
            (nt_dense + group_chunks_max - 1) / group_chunks_max;
        if(contexts_per_sequence <= 0 ||
           global_context >= nseq * contexts_per_sequence)
            return;
        seq = global_context / contexts_per_sequence;
        local_context = global_context - seq * contexts_per_sequence;
        bos = seq * t_seq;
        eos = bos + t_seq;
        chunks = nt_dense;
    }

    const int first_chunk = local_context * group_chunks_max;
    int group_chunks = chunks - first_chunk;
    group_chunks = group_chunks < group_chunks_max
        ? group_chunks
        : group_chunks_max;
    if(group_chunks <= 0)
        return;
    const int64_t context_slab =
        (static_cast<int64_t>(global_context) * h_count + head) * D * D;

    alignas(16) __shared__ bf16 kd[C * KD_STRIDE];
    alignas(16) __shared__ bf16
        kr[(D / C) * K2_CONTEXT_GFX1250_KR_ASYNC_TILE_STRIDE];
    alignas(16) __shared__ bf16 inv[C * FACTOR_STRIDE];
    alignas(16) __shared__ float decay[D];
    alignas(16) __shared__ float beta_act[C];

    Acc state[D / C];
    Acc identity_fragment{};
    const int identity_v_local = lane & 15;
    const int identity_lane_group = lane >> 4;
    const float identity_value =
        form_affine_a && (identity_v_local >> 3) == identity_lane_group
            ? 1.0f
            : 0.0f;
    #pragma unroll
    for(int i = 0; i < 8; ++i)
        identity_fragment[i] =
            (identity_v_local & 7) == i ? identity_value : 0.0f;
    #pragma unroll
    for(int k_tile = 0; k_tile < D / C; ++k_tile)
    {
        const Acc zero{};
        state[k_tile] = v_tile == k_tile ? identity_fragment : zero;
    }

    for(int group_chunk = 0; group_chunk < group_chunks; ++group_chunk)
    {
        const int chunk = first_chunk + group_chunk;
        const int ht = Varlen
            ? head * total_tiles + tile0 + chunk
            : (seq * h_count + head) * nt_dense + chunk;
        const int token0 = bos + chunk * C;
        int active = eos - token0;
        active = active < C ? active : C;
        const int64_t matrix_offset = static_cast<int64_t>(ht) * C * D;
        const int64_t factor_offset = static_cast<int64_t>(ht) * C * C;

        k2_context_gfx1250_async_stage_kd_b128<4 * WAVE>(
            ws_kd + matrix_offset, kd, thread);
        k2_context_gfx1250_async_stage_kr_b128<4 * WAVE>(
            ws_kr + matrix_offset, kr, thread);
        k2_context_gfx1250_async_stage_shared_factors_4wave_b128(
            ws_inv + factor_offset,
            inv,
            ws_decay + static_cast<int64_t>(ht) * D,
            decay,
            beta_cache + static_cast<int64_t>(ht) * C,
            beta_act,
            thread);
        k2_context_gfx1250_wait_async_b128();
        __syncthreads();

        const Acc prediction =
            k2_context_gfx1250_contract_with_state<KD_STRIDE>(
                kd, state, lane);

        FragB residual_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int token = (lane >> 4) * 8 + i;
            const int v_local = lane & 15;
            float v = 0.0f;
            if(!form_affine_a && token < active)
            {
                const int64_t offset =
                    (static_cast<int64_t>(token0 + token) * h_count + head) *
                        D + v0 + v_local;
                v = bf16_to_f32(value[offset]);
            }
            residual_fragment[i] =
                f32_to_bf16((v - prediction[i]) * beta_act[token]);
            residual_fragment[8 + i] = f32_to_bf16(0.0f);
        }

        const int factor_row = lane & 15;
        const int lane_group = lane >> 4;
        FragA inv_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int col = lane_group * 8 + i;
            inv_fragment[i] = inv[factor_row * FACTOR_STRIDE + col];
            inv_fragment[8 + i] = f32_to_bf16(0.0f);
        }
        Acc zero{};
        const Acc u_acc = wmma(inv_fragment, residual_fragment, zero);
        FragB u_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            u_fragment[i] = f32_to_bf16(u_acc[i]);
            u_fragment[8 + i] = f32_to_bf16(0.0f);
        }

        #pragma unroll
        for(int k_tile = 0; k_tile < D / C; ++k_tile)
        {
            const Acc update =
                k2_context_gfx1250_state_update_tiled_async_b128(
                    kr, u_fragment, k_tile, lane);
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int k = k_tile * C + lane_group * 8 + i;
                state[k_tile][i] =
                    state[k_tile][i] * decay[k] + update[i];
            }
        }

        // Every wave retires its own LDS reads before any publisher can reuse
        // the single shared arena.  The barrier covers all four consumers.
        k2_context_gfx1250_wait_lds();
        __syncthreads();
    }

    const int64_t lane_store_offset =
        context_slab +
        static_cast<int64_t>((lane >> 4) * 8) * D +
        v0 + (lane & 15);
    if(form_affine_a)
    {
        #pragma unroll
        for(int k_tile = 0; k_tile < D / C; ++k_tile)
        {
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int row = k_tile * C + i;
                affine_a[lane_store_offset + static_cast<int64_t>(row) * D] =
                    f32_to_bf16(state[k_tile][i]);
            }
        }
    }
    else
    {
        #pragma unroll
        for(int k_tile = 0; k_tile < D / C; ++k_tile)
        {
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int row = k_tile * C + i;
                affine_b[lane_store_offset + static_cast<int64_t>(row) * D] =
                    state[k_tile][i];
            }
        }
    }
}

// Single-wave replay companion for the async paired producer. Final-state
// publication mode is runtime-uniform and dense/varlen remains a compile-time
// axis. A wave wait is sufficient for publication because this workgroup
// contains one wave; no workgroup barrier is needed on the replay fast path.
template <bool Varlen>
__global__ __launch_bounds__(WAVE)
void k2_kda_context_gfx1250_replay_async_b128_kernel(
    const bf16* __restrict__ value,
    bf16* __restrict__ output,
    const bf16* __restrict__ ws_kd,
    const bf16* __restrict__ ws_qd,
    const bf16* __restrict__ ws_kr,
    const float* __restrict__ ws_decay,
    const bf16* __restrict__ ws_inv,
    const bf16* __restrict__ ws_mqk,
    const float* __restrict__ beta_cache,
    const float* __restrict__ affine_b,
    void* __restrict__ final_state,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ tile_prefix,
    const int* __restrict__ context_prefix,
    int total_context_groups,
    int total_tiles,
    int t_seq,
    int h_count,
    int nseq,
    int nt_dense,
    int group_chunks_max,
    bool has_state_out,
    bool state_fp32)
{
    if(group_chunks_max != K2_CONTEXT_GFX1250_GROUP4 &&
       group_chunks_max != K2_CONTEXT_GFX1250_GROUP8 &&
       group_chunks_max != K2_CONTEXT_GFX1250_GROUP16)
        return;

    const int lane = static_cast<int>(threadIdx.x);
    const int context_head = static_cast<int>(blockIdx.x);
    const int v_tile = static_cast<int>(blockIdx.y);
    const int global_context = context_head / h_count;
    const int head = context_head - global_context * h_count;
    const int v0 = v_tile * BV;

    int seq = 0;
    int local_context = 0;
    int bos = 0;
    int eos = 0;
    int tile0 = 0;
    int chunks = 0;
    if constexpr(Varlen)
    {
        if(nseq <= 0 || global_context >= context_prefix[nseq])
            return;
        int lo = 0;
        int hi = nseq;
        while(hi - lo > 1)
        {
            const int mid = (lo + hi) >> 1;
            if(context_prefix[mid] <= global_context)
                lo = mid;
            else
                hi = mid;
        }
        seq = lo;
        local_context = global_context - context_prefix[seq];
        bos = cu_seqlens[seq];
        eos = cu_seqlens[seq + 1];
        tile0 = tile_prefix[seq];
        chunks = tile_prefix[seq + 1] - tile0;
    }
    else
    {
        const int contexts_per_sequence =
            (nt_dense + group_chunks_max - 1) / group_chunks_max;
        if(contexts_per_sequence <= 0 ||
           global_context >= nseq * contexts_per_sequence)
            return;
        seq = global_context / contexts_per_sequence;
        local_context = global_context - seq * contexts_per_sequence;
        bos = seq * t_seq;
        eos = bos + t_seq;
        chunks = nt_dense;
    }

    const int first_chunk = local_context * group_chunks_max;
    int group_chunks = chunks - first_chunk;
    group_chunks = group_chunks < group_chunks_max
        ? group_chunks
        : group_chunks_max;
    if(group_chunks <= 0)
        return;
    const bool is_last_group = first_chunk + group_chunks == chunks;
    const int64_t context_slab =
        (static_cast<int64_t>(global_context) * h_count + head) * D * D;

    alignas(16) __shared__ bf16 kd[C * KD_STRIDE];
    alignas(16) __shared__ bf16 qd[C * KD_STRIDE];
    alignas(16) __shared__ bf16
        kr[(D / C) * K2_CONTEXT_GFX1250_KR_ASYNC_TILE_STRIDE];
    alignas(16) __shared__ bf16 inv[C * FACTOR_STRIDE];
    alignas(16) __shared__ bf16 mqk[C * FACTOR_STRIDE];
    alignas(16) __shared__ float decay[D];
    alignas(16) __shared__ float beta_act[C];

    Acc state[D / C];
    #pragma unroll
    for(int k_tile = 0; k_tile < D / C; ++k_tile)
    {
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int v = v0 + (lane & 15);
            const int k = k_tile * C + (lane >> 4) * 8 + i;
            state[k_tile][i] =
                affine_b[context_slab + static_cast<int64_t>(k) * D + v];
        }
    }

    for(int group_chunk = 0; group_chunk < group_chunks; ++group_chunk)
    {
        const int chunk = first_chunk + group_chunk;
        const int ht = Varlen
            ? head * total_tiles + tile0 + chunk
            : (seq * h_count + head) * nt_dense + chunk;
        const int token0 = bos + chunk * C;
        int active = eos - token0;
        active = active < C ? active : C;
        const int64_t matrix_offset = static_cast<int64_t>(ht) * C * D;
        const int64_t factor_offset = static_cast<int64_t>(ht) * C * C;

        k2_context_gfx1250_replay_exact_stage_kd_b128(
            ws_kd + matrix_offset, kd, lane);
        k2_context_gfx1250_replay_exact_stage_kd_b128(
            ws_qd + matrix_offset, qd, lane);
        k2_context_gfx1250_replay_exact_stage_kr_b128(
            ws_kr + matrix_offset, kr, lane);
        k2_context_gfx1250_replay_exact_stage_factor_b128(
            ws_inv + factor_offset, inv, lane);
        k2_context_gfx1250_replay_exact_stage_factor_b128(
            ws_mqk + factor_offset, mqk, lane);
        k2_context_gfx1250_replay_exact_stage_decay_b128(
            ws_decay + static_cast<int64_t>(ht) * D, decay, lane);
        k2_context_gfx1250_replay_exact_stage_beta_b128(
            beta_cache + static_cast<int64_t>(ht) * C, beta_act, lane);
        // The exact publisher emits KD(8), QD(8), KR(8), inv(1), mqk(1),
        // decay(1), and beta(1), in that order.  gfx1250 retires async LDS
        // writes in issue order, so leaving twelve requests outstanding proves
        // that all sixteen KD/QD vectors are resident.  Their contractions and
        // the full-chunk value loads cover the younger publication latency.
        k2_context_gfx1250_wait_async_b128_count<12>();

        Acc prediction{};
        Acc from_state{};
        FragB residual_fragment{};
        if(active == C)
        {
            const int v_local = lane & 15;
            const int value_pair = v_local & ~1;
            ContextBf16x2 packed_value[8];
            #pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                const int token = (lane >> 4) * 8 + i;
                const int64_t offset =
                    (static_cast<int64_t>(token0 + token) * h_count + head) *
                        D + v0 + value_pair;
                packed_value[i] =
                    k2_context_gfx1250_load_bf16x2_unaligned(value + offset);
            }
            #pragma unroll
            for(int i = 4; i < 8; ++i)
            {
                const int token = (lane >> 4) * 8 + i;
                const int64_t offset =
                    (static_cast<int64_t>(token0 + token) * h_count + head) *
                        D + v0 + value_pair;
                packed_value[i] =
                    k2_context_gfx1250_load_bf16x2_unaligned(value + offset);
            }

            // Keep the independent KD/QD contraction below all eight value
            // issues.  This compiler scheduling fence emits no ISA and does
            // not wait for the global loads; both contractions still share
            // each rounded state panel and retain their established order.
            asm volatile("" ::: "memory");
            const AccPair products =
                contract_pair_with_state<KD_STRIDE, KD_STRIDE>(
                    kd, qd, state, lane);
            k2_context_gfx1250_wait_async_b128();
            prediction = products.first;
            from_state = products.second;

            // Retire the oldest four b32 loads, consume them, then drain the
            // younger four.  KD/QD WMMA and LDS work fills the load latency
            // window without moving any arithmetic rounding point.
            k2_context_gfx1250_wait_global_loads<4>();
            #pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                const int token = (lane >> 4) * 8 + i;
                const float v =
                    bf16_to_f32(packed_value[i][v_local & 1]);
                residual_fragment[i] =
                    f32_to_bf16((v - prediction[i]) * beta_act[token]);
                residual_fragment[8 + i] = f32_to_bf16(0.0f);
            }
            k2_context_gfx1250_wait_global_loads<0>();
            #pragma unroll
            for(int i = 4; i < 8; ++i)
            {
                const int token = (lane >> 4) * 8 + i;
                const float v =
                    bf16_to_f32(packed_value[i][v_local & 1]);
                residual_fragment[i] =
                    f32_to_bf16((v - prediction[i]) * beta_act[token]);
                residual_fragment[8 + i] = f32_to_bf16(0.0f);
            }
        }
        else
        {
            const AccPair products =
                contract_pair_with_state<KD_STRIDE, KD_STRIDE>(
                    kd, qd, state, lane);
            k2_context_gfx1250_wait_async_b128();
            prediction = products.first;
            from_state = products.second;
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int token = (lane >> 4) * 8 + i;
                const int v_local = lane & 15;
                float v = 0.0f;
                if(token < active)
                {
                    const int64_t offset =
                        (static_cast<int64_t>(token0 + token) * h_count +
                         head) * D + v0 + v_local;
                    v = bf16_to_f32(value[offset]);
                }
                residual_fragment[i] =
                    f32_to_bf16((v - prediction[i]) * beta_act[token]);
                residual_fragment[8 + i] = f32_to_bf16(0.0f);
            }
        }

        const int factor_row = lane & 15;
        const int lane_group = lane >> 4;
        FragA inv_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int col = lane_group * 8 + i;
            inv_fragment[i] = inv[factor_row * FACTOR_STRIDE + col];
            inv_fragment[8 + i] = f32_to_bf16(0.0f);
        }
        Acc zero{};
        const Acc u_acc = wmma(inv_fragment, residual_fragment, zero);
        FragB u_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            u_fragment[i] = f32_to_bf16(u_acc[i]);
            u_fragment[8 + i] = f32_to_bf16(0.0f);
        }

        FragA mqk_fragment{};
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int col = lane_group * 8 + i;
            mqk_fragment[i] = mqk[factor_row * FACTOR_STRIDE + col];
            mqk_fragment[8 + i] = f32_to_bf16(0.0f);
        }
        const Acc local_output = wmma(mqk_fragment, u_fragment, zero);
        if(active == C)
        {
            // A full C16 chunk is uniform across the wave.  Keep the active
            // test outside the unrolled stores so EXEC stays unchanged across
            // all eight issues while preserving round-add-round exactly.
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int token = (lane >> 4) * 8 + i;
                const int v_local = lane & 15;
                const bf16 a = f32_to_bf16(from_state[i]);
                const bf16 b = f32_to_bf16(local_output[i]);
                const int64_t offset =
                    (static_cast<int64_t>(token0 + token) * h_count + head) *
                        D + v0 + v_local;
                output[offset] =
                    f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
            }
        }
        else
        {
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int token = (lane >> 4) * 8 + i;
                const int v_local = lane & 15;
                if(token < active)
                {
                    const bf16 a = f32_to_bf16(from_state[i]);
                    const bf16 b = f32_to_bf16(local_output[i]);
                    const int64_t offset =
                        (static_cast<int64_t>(token0 + token) * h_count +
                         head) * D + v0 + v_local;
                    output[offset] =
                        f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
                }
            }
        }

        #pragma unroll
        for(int k_tile = 0; k_tile < D / C; ++k_tile)
        {
            const Acc update =
                k2_context_gfx1250_state_update_tiled_async_b128(
                    kr, u_fragment, k_tile, lane);
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int k = k_tile * C + lane_group * 8 + i;
                state[k_tile][i] =
                    state[k_tile][i] * decay[k] + update[i];
            }
        }

        // No barrier is necessary for one wave, but the next async publisher
        // must not overwrite addresses still referenced by outstanding DS ops.
        k2_context_gfx1250_wait_lds();
    }

    if(has_state_out && is_last_group)
    {
        #pragma unroll
        for(int k_tile = 0; k_tile < D / C; ++k_tile)
        {
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int v = v0 + (lane & 15);
                const int k = k_tile * C + (lane >> 4) * 8 + i;
                const int64_t state_offset =
                    ((static_cast<int64_t>(seq) * h_count + head) * D + v) *
                        D + k;
                if(state_fp32)
                    static_cast<float*>(final_state)[state_offset] =
                        state[k_tile][i];
                else
                    static_cast<bf16*>(final_state)[state_offset] =
                        f32_to_bf16(state[k_tile][i]);
            }
        }
    }
}

// Four-wave K-split scan over the context-group affine transforms.
template <
    int GroupChunks,
    bool HasStateIn,
    bool HasStateOut,
    bool StateFp32,
    bool Varlen>
__global__ __launch_bounds__(4 * WAVE)
void k2_kda_context_gfx1250_scan_ksplit_kernel(
    const bf16* __restrict__ affine_a,
    float* __restrict__ affine_b,
    const void* __restrict__ initial_state,
    void* __restrict__ final_state,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ context_prefix,
    int total_context_groups,
    int t_seq,
    int h_count,
    int nseq,
    int nt_dense,
    bool skip_final_affine_update)
{
    static_assert(GroupChunks == K2_CONTEXT_GFX1250_GROUP4 ||
                      GroupChunks == K2_CONTEXT_GFX1250_GROUP8 ||
                      GroupChunks == K2_CONTEXT_GFX1250_GROUP16,
                  "gfx1250 K-split scan supports G4, G8, or G16");

    constexpr int KHalf = D / 2;
    constexpr int LocalKTiles = KHalf / C;
    constexpr int Waves = 4;
    constexpr int Threads = Waves * WAVE;
    constexpr int AMatrixElements = D * KD_STRIDE;
    constexpr int AMatrixBytes = AMatrixElements * int(sizeof(bf16));
    // [vhalf][destination khalf][local K16 tile][lane][8]
    constexpr int ExchangeElements =
        2 * 2 * LocalKTiles * WAVE * 8;
    constexpr int ExchangeBytes = ExchangeElements * int(sizeof(float));
    constexpr int SharedBytes = AMatrixBytes + ExchangeBytes;
    static_assert(KHalf == 64 && LocalKTiles == 4 && Threads == 128 &&
                      KD_STRIDE == 136,
                  "gfx1250 K-split scan geometry changed");
    static_assert(AMatrixBytes == 34816 && ExchangeBytes == 16384 &&
                      SharedBytes == 51200,
                  "gfx1250 K-split scan LDS contract changed");
    static_assert(V_TILES == 8,
                  "gfx1250 K-split scan requires eight V16 tiles");

    (void)total_context_groups;
    (void)t_seq;
    (void)nseq;
    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid / WAVE;
    const int lane = tid - wave * WAVE;
    const int vhalf = wave & 1;
    const int khalf = wave >> 1;
    const int seq_head = static_cast<int>(blockIdx.x);
    const int seq = seq_head / h_count;
    const int head = seq_head - seq * h_count;
    const int v0 =
        (static_cast<int>(blockIdx.y) * 2 + vhalf) * BV;
    const int k0 = khalf * KHalf;

    int context_base = 0;
    int context_count = 0;
    if constexpr(Varlen)
    {
        context_base = context_prefix[seq];
        context_count = context_prefix[seq + 1] - context_base;
    }
    else
    {
        const int contexts_per_sequence =
            (nt_dense + GroupChunks - 1) / GroupChunks;
        context_base = seq * contexts_per_sequence;
        context_count = contexts_per_sequence;
    }

    const int64_t state_slab =
        (static_cast<int64_t>(seq) * h_count + head) * D * D;
    if(context_count == 0)
    {
        // Replay owns final state for non-empty sequences.  The scan owns
        // only this empty-sequence initial-state copy.
        if constexpr(HasStateOut)
        {
            #pragma unroll
            for(int local_k_tile = 0;
                local_k_tile < LocalKTiles;
                ++local_k_tile)
            {
                #pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    const int v = v0 + (lane & 15);
                    const int k =
                        k0 + local_k_tile * C + (lane >> 4) * 8 + i;
                    const int64_t state_offset =
                        state_slab + static_cast<int64_t>(v) * D + k;
                    float value = 0.0f;
                    if constexpr(HasStateIn)
                    {
                        if constexpr(StateFp32)
                            value = static_cast<const float*>(initial_state)
                                [state_offset];
                        else
                            value = bf16_to_f32(
                                static_cast<const bf16*>(initial_state)
                                    [state_offset]);
                    }
                    if constexpr(StateFp32)
                        static_cast<float*>(final_state)[state_offset] = value;
                    else
                        static_cast<bf16*>(final_state)[state_offset] =
                            f32_to_bf16(value);
                }
            }
        }
        return;
    }

    Acc state[LocalKTiles];
    #pragma unroll
    for(int local_k_tile = 0;
        local_k_tile < LocalKTiles;
        ++local_k_tile)
    {
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int v = v0 + (lane & 15);
            const int k =
                k0 + local_k_tile * C + (lane >> 4) * 8 + i;
            const int64_t state_offset =
                state_slab + static_cast<int64_t>(v) * D + k;
            float value = 0.0f;
            if constexpr(HasStateIn)
            {
                if constexpr(StateFp32)
                    value = static_cast<const float*>(initial_state)
                        [state_offset];
                else
                    value = bf16_to_f32(
                        static_cast<const bf16*>(initial_state)[state_offset]);
            }
            state[local_k_tile][i] = value;
        }
    }

    alignas(32) __shared__ unsigned char shared_storage[SharedBytes];
    bf16* const a_matrix = reinterpret_cast<bf16*>(shared_storage);
    float* const exchange = reinterpret_cast<float*>(
        shared_storage + AMatrixBytes);

    for(int local_context = 0;
        local_context < context_count;
        ++local_context)
    {
        const int global_context = context_base + local_context;
        const int64_t context_slab =
            (static_cast<int64_t>(global_context) * h_count + head) * D * D;

        // As in the one-wave scan, only the incoming state of the final
        // context is visible to replay.  Avoid staging A and executing the
        // split K128 contraction when dead-update elimination is selected.
        // All four waves take the branch uniformly.
        if(skip_final_affine_update && local_context + 1 == context_count)
        {
            #pragma unroll
            for(int local_k_tile = 0;
                local_k_tile < LocalKTiles;
                ++local_k_tile)
            {
                #pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    const int v = v0 + (lane & 15);
                    const int k =
                        k0 + local_k_tile * C + (lane >> 4) * 8 + i;
                    const int64_t offset =
                        context_slab + static_cast<int64_t>(k) * D + v;
                    affine_b[offset] = state[local_k_tile][i];
                }
            }
            break;
        }

        // The publication barrier also protects exchange reuse: every wave
        // reaches it only after consuming the preceding context's partial.
        k2_context_gfx1250_async_stage_affine_a_b128<Threads>(
            affine_a + context_slab, a_matrix, tid);
        k2_context_gfx1250_wait_async_b128();
        __syncthreads();

        // Finish both destination halves from the old state before any lane
        // overwrites its four resident K16 tiles.
        Acc owned[LocalKTiles];
        #pragma unroll
        for(int output_local = 0;
            output_local < LocalKTiles;
            ++output_local)
        {
            const int output_k_tile =
                khalf * LocalKTiles + output_local;
            owned[output_local] =
                k2_context_gfx1250_contract_k64_with_state<KD_STRIDE>(
                    a_matrix + output_k_tile * C * KD_STRIDE + k0,
                    state,
                    lane);
        }

        const int remote_khalf = khalf ^ 1;
        #pragma unroll
        for(int output_local = 0;
            output_local < LocalKTiles;
            ++output_local)
        {
            const int output_k_tile =
                remote_khalf * LocalKTiles + output_local;
            const Acc partial =
                k2_context_gfx1250_contract_k64_with_state<KD_STRIDE>(
                    a_matrix + output_k_tile * C * KD_STRIDE + k0,
                    state,
                    lane);
            const int exchange_base =
                ((((vhalf * 2 + remote_khalf) * LocalKTiles +
                    output_local) * WAVE + lane) * 8);
            #pragma unroll
            for(int i = 0; i < 8; ++i)
                exchange[exchange_base + i] = partial[i];
        }
        __syncthreads();

        #pragma unroll
        for(int local_k_tile = 0;
            local_k_tile < LocalKTiles;
            ++local_k_tile)
        {
            const int exchange_base =
                ((((vhalf * 2 + khalf) * LocalKTiles + local_k_tile) *
                   WAVE + lane) * 8);
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int v = v0 + (lane & 15);
                const int k =
                    k0 + local_k_tile * C + (lane >> 4) * 8 + i;
                const int64_t offset =
                    context_slab + static_cast<int64_t>(k) * D + v;
                const float b = affine_b[offset];
                affine_b[offset] = state[local_k_tile][i];
                const float remote = exchange[exchange_base + i];
                const float partial_low =
                    khalf == 0 ? owned[local_k_tile][i] : remote;
                const float partial_high =
                    khalf == 0 ? remote : owned[local_k_tile][i];
                state[local_k_tile][i] =
                    (partial_low + partial_high) + b;
            }
        }
    }
}

// Strict-opt-in Stage-8 eight-wave affine-A pipeline.  The established
// four-wave K-split scan assigns one wave to each (V16, source-K64) pair and
// makes that wave compute both destination K64 halves.  This topology adds
// destination-K64 as a wave dimension:
//
//   wave = source_khalf * 4 + output_khalf * 2 + vhalf.
//
// Two affine-A buffers let context i+1 publish while context i executes.  A
// separate next-state arena preserves the duplicated source state needed by
// both destination groups.  The K64-low + K64-high + b reduction graph is
// unchanged, so the kernel remains bitwise-compatible with the four-wave
// K-split family.  Calls without final-state output deliberately fail closed
// to the current four-wave kernel at launch time.
template <
    int GroupChunks,
    bool HasStateIn,
    bool HasStateOut,
    bool StateFp32,
    bool Varlen,
    bool ExactAPublisher = false,
    bool PostWmmaBPrefetch = false,
    bool FragmentPipeline = false,
    bool FusedPublish = false,
    bool DeferStateWait = false,
    bool LdsOnlyPublishBarrier = false,
    bool BPrefetchStateReuse = false,
    bool PostBExchangeLdsOnly = false>
__global__ __launch_bounds__(8 * WAVE)
void k2_kda_context_gfx1250_scan_ksplit8_apipeline_kernel(
    const bf16* __restrict__ affine_a,
    float* __restrict__ affine_b,
    const void* __restrict__ initial_state,
    void* __restrict__ final_state,
    const int32_t* __restrict__ cu_seqlens,
    const int* __restrict__ context_prefix,
    int total_context_groups,
    int t_seq,
    int h_count,
    int nseq,
    int nt_dense,
    bool skip_final_affine_update)
{
    static_assert(GroupChunks == K2_CONTEXT_GFX1250_GROUP4 ||
                      GroupChunks == K2_CONTEXT_GFX1250_GROUP8 ||
                      GroupChunks == K2_CONTEXT_GFX1250_GROUP16,
                  "gfx1250 K-split8 A-pipeline supports G4/G8/G16");

    constexpr int KHalf = D / 2;
    constexpr int LocalKTiles = KHalf / C;
    constexpr int Waves = 8;
    constexpr int Threads = Waves * WAVE;
    constexpr int AMatrixElements = D * KD_STRIDE;
    constexpr int AMatrixBytes = AMatrixElements * int(sizeof(bf16));
    constexpr int AMatrixBuffers = 2;
    // [vhalf][destination khalf][local K16 tile][lane][8]
    constexpr int ExchangeElements =
        2 * 2 * LocalKTiles * WAVE * 8;
    constexpr int ExchangeBytes = ExchangeElements * int(sizeof(float));
    constexpr int StateElements = 2 * D * BV;
    constexpr int StateBytes = StateElements * int(sizeof(float));
    constexpr int SharedBytes =
        AMatrixBuffers * AMatrixBytes + ExchangeBytes + StateBytes;
    static_assert(KHalf == 64 && LocalKTiles == 4 && Threads == 256 &&
                      KD_STRIDE == 136,
                  "gfx1250 K-split8 A-pipeline geometry changed");
    static_assert(AMatrixBytes == 34816 && ExchangeBytes == 16384 &&
                      StateBytes == 16384 && SharedBytes == 102400,
                  "gfx1250 K-split8 A-pipeline LDS contract changed");
    static_assert(V_TILES == 8,
                  "gfx1250 K-split8 A-pipeline requires eight V16 tiles");
    static_assert(!DeferStateWait || FusedPublish,
                  "deferred state wait requires fused publication");
    static_assert(!LdsOnlyPublishBarrier || FusedPublish,
                  "LDS-only publication barrier requires fused publication");
    static_assert(!BPrefetchStateReuse || PostWmmaBPrefetch,
                  "B-prefetch state reuse requires post-WMMA B prefetch");
    static_assert(!PostBExchangeLdsOnly || PostWmmaBPrefetch,
                  "LDS-only post-B exchange requires post-WMMA B prefetch");
    static_assert(
        !PostBExchangeLdsOnly ||
            (!ExactAPublisher && !FragmentPipeline && !FusedPublish &&
             !DeferStateWait && !LdsOnlyPublishBarrier &&
             !BPrefetchStateReuse),
        "LDS-only post-B exchange is instantiated only for the base "
        "post-WMMA B-prefetch schedule");
    static_assert(!BPrefetchStateReuse || !LdsOnlyPublishBarrier,
                  "B-prefetch state reuse is not validated with the "
                  "LDS-only publication barrier");
    static_assert(
        !BPrefetchStateReuse ||
            ((!ExactAPublisher && !FragmentPipeline && !FusedPublish &&
              !DeferStateWait) ||
             (ExactAPublisher && FragmentPipeline && FusedPublish &&
              DeferStateWait)),
        "B-prefetch state reuse is instantiated only for the base or fully "
        "composed schedule");
    static_assert(!DeferStateWait || HasStateIn || !FragmentPipeline ||
                      PostWmmaBPrefetch,
                  "deferred state wait without initial state requires "
                  "post-WMMA B prefetch when fragment pipelining is enabled");

    (void)cu_seqlens;
    (void)total_context_groups;
    (void)t_seq;
    (void)nseq;
    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid / WAVE;
    const int lane = tid - wave * WAVE;
    const int vhalf = wave & 1;
    const int output_khalf = (wave >> 1) & 1;
    const int source_khalf = wave >> 2;
    const bool owns_output = source_khalf == output_khalf;
    const int seq_head = static_cast<int>(blockIdx.x);
    const int seq = seq_head / h_count;
    const int head = seq_head - seq * h_count;
    const int v0 =
        (static_cast<int>(blockIdx.y) * 2 + vhalf) * BV;
    const int source_k0 = source_khalf * KHalf;
    const int output_k0 = output_khalf * KHalf;

    int context_base = 0;
    int context_count = 0;
    if constexpr(Varlen)
    {
        context_base = context_prefix[seq];
        context_count = context_prefix[seq + 1] - context_base;
    }
    else
    {
        const int contexts_per_sequence =
            (nt_dense + GroupChunks - 1) / GroupChunks;
        context_base = seq * contexts_per_sequence;
        context_count = contexts_per_sequence;
    }

    const int64_t state_slab =
        (static_cast<int64_t>(seq) * h_count + head) * D * D;
    if(context_count == 0)
    {
        // Replay owns final state for non-empty sequences.  For an empty
        // sequence, only the source/destination-matched waves copy the
        // initial state so every output element has one writer.
        if constexpr(HasStateOut)
        {
            if(owns_output)
            {
                #pragma unroll
                for(int local_k_tile = 0;
                    local_k_tile < LocalKTiles;
                    ++local_k_tile)
                {
                    #pragma unroll
                    for(int i = 0; i < 8; ++i)
                    {
                        const int v = v0 + (lane & 15);
                        const int k = output_k0 + local_k_tile * C +
                            (lane >> 4) * 8 + i;
                        const int64_t state_offset =
                            state_slab + static_cast<int64_t>(v) * D + k;
                        float value = 0.0f;
                        if constexpr(HasStateIn)
                        {
                            if constexpr(StateFp32)
                                value =
                                    static_cast<const float*>(initial_state)
                                        [state_offset];
                            else
                                value = bf16_to_f32(
                                    static_cast<const bf16*>(initial_state)
                                        [state_offset]);
                        }
                        if constexpr(StateFp32)
                            static_cast<float*>(final_state)[state_offset] =
                                value;
                        else
                            static_cast<bf16*>(final_state)[state_offset] =
                                f32_to_bf16(value);
                    }
                }
            }
        }
        return;
    }

    // The two destination groups deliberately duplicate the old source
    // state.  This costs one extra initial-state read per element, not per
    // recurrent context.
    Acc state[LocalKTiles];
    #pragma unroll
    for(int local_k_tile = 0;
        local_k_tile < LocalKTiles;
        ++local_k_tile)
    {
        #pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int v = v0 + (lane & 15);
            const int k = source_k0 + local_k_tile * C +
                (lane >> 4) * 8 + i;
            const int64_t state_offset =
                state_slab + static_cast<int64_t>(v) * D + k;
            float value = 0.0f;
            if constexpr(HasStateIn)
            {
                if constexpr(StateFp32)
                    value = static_cast<const float*>(initial_state)
                        [state_offset];
                else
                    value = bf16_to_f32(
                        static_cast<const bf16*>(initial_state)[state_offset]);
            }
            state[local_k_tile][i] = value;
        }
    }

    alignas(32) __shared__ unsigned char shared_storage[SharedBytes];
    bf16* const a_matrix = reinterpret_cast<bf16*>(shared_storage);
    float* const exchange = reinterpret_cast<float*>(
        shared_storage + AMatrixBuffers * AMatrixBytes);
    float* const next_state = reinterpret_cast<float*>(
        shared_storage + AMatrixBuffers * AMatrixBytes + ExchangeBytes);

    // Replay consumes the incoming state of the final context.  When the
    // dead final affine update is disabled, keep the historical Stage-8 path
    // by setting update_count == context_count.  Otherwise the last context
    // is publication-only and its affine-A map is never fetched.
    const int update_count =
        context_count - (skip_final_affine_update ? 1 : 0);
    if(update_count > 0)
    {
        const int64_t first_context_slab =
            (static_cast<int64_t>(context_base) * h_count + head) * D * D;
        if constexpr(ExactAPublisher)
            k2_context_gfx1250_async_stage_affine_a_8w_exact_b128(
                affine_a + first_context_slab, a_matrix, tid);
        else
            k2_context_gfx1250_async_stage_affine_a_b128<Threads>(
                affine_a + first_context_slab, a_matrix, tid);
        k2_context_gfx1250_wait_async_b128();
        __syncthreads();
    }

    for(int local_context = 0;
        local_context < update_count;
        ++local_context)
    {
        const int global_context = context_base + local_context;
        const int64_t context_slab =
            (static_cast<int64_t>(global_context) * h_count + head) * D * D;
        const int current_buffer = local_context & 1;
        bf16* const current_a =
            a_matrix + current_buffer * AMatrixElements;
        const bool has_next_update = local_context + 1 < update_count;

        if(has_next_update)
        {
            const int next_global_context = global_context + 1;
            const int next_buffer = current_buffer ^ 1;
            const int64_t next_context_slab =
                (static_cast<int64_t>(next_global_context) * h_count + head) *
                D * D;
            if constexpr(ExactAPublisher)
                k2_context_gfx1250_async_stage_affine_a_8w_exact_b128(
                    affine_a + next_context_slab,
                    a_matrix + next_buffer * AMatrixElements,
                    tid);
            else
                k2_context_gfx1250_async_stage_affine_a_b128<Threads>(
                    affine_a + next_context_slab,
                    a_matrix + next_buffer * AMatrixElements,
                    tid);
        }

        // The fused publication barrier has already made next_state visible.
        // On continuation iterations, defer its wave-local completion wait
        // until after the independent next-A address work and async copies.
        // The final iteration still drains immediately at the load site below
        // so the publication-only tail never consumes an outstanding state.
        if constexpr(DeferStateWait)
        {
            if(local_context != 0)
                k2_context_gfx1250_wait_lds();
        }

        Acc partial[LocalKTiles];
        if constexpr(FragmentPipeline)
        {
            static_assert(LocalKTiles == 4,
                          "fragment pipeline requires four local K16 tiles");
            #pragma unroll
            for(int output_pair = 0;
                output_pair < LocalKTiles / 2;
                ++output_pair)
            {
                const int output_local0 = output_pair * 2;
                const int output_local1 = output_local0 + 1;
                const int output_k_tile0 =
                    output_khalf * LocalKTiles + output_local0;
                const int output_k_tile1 = output_k_tile0 + 1;
                k2_context_gfx1250_contract_k64_pair_fragment_pipeline<
                    KD_STRIDE>(
                        current_a +
                            output_k_tile0 * C * KD_STRIDE + source_k0,
                        current_a +
                            output_k_tile1 * C * KD_STRIDE + source_k0,
                        state,
                        lane,
                        partial[output_local0],
                        partial[output_local1]);
            }
        }
        else
        {
            #pragma unroll
            for(int output_local = 0;
                output_local < LocalKTiles;
                ++output_local)
            {
                const int output_k_tile =
                    output_khalf * LocalKTiles + output_local;
                partial[output_local] =
                    k2_context_gfx1250_contract_k64_with_state<KD_STRIDE>(
                        current_a +
                            output_k_tile * C * KD_STRIDE + source_k0,
                        state,
                        lane);
            }
        }

        // Owner waves can issue the 32 affine-b loads as soon as their two
        // K32 WMMAs finish.  The established path keeps a separate B register
        // file until after the exchange barrier.  The strict-opt-in reuse
        // path instead publishes old state immediately after each same-address
        // B load and lets the now-dead state registers carry B across the
        // rendezvous.  This preserves load-before-store and the recurrence
        // graph while shortening the live range by 32 FP32 values per lane.
        Acc prefetched_b[LocalKTiles];
        if constexpr(PostWmmaBPrefetch)
        {
            if(owns_output)
            {
                #pragma unroll
                for(int local_k_tile = 0;
                    local_k_tile < LocalKTiles;
                    ++local_k_tile)
                {
                    #pragma unroll
                    for(int i = 0; i < 8; ++i)
                    {
                        const int v = v0 + (lane & 15);
                        const int k = output_k0 + local_k_tile * C +
                            (lane >> 4) * 8 + i;
                        const int64_t offset =
                            context_slab + static_cast<int64_t>(k) * D + v;
                        if constexpr(BPrefetchStateReuse)
                        {
                            const float b = affine_b[offset];
                            affine_b[offset] = state[local_k_tile][i];
                            state[local_k_tile][i] = b;
                        }
                        else
                        {
                            prefetched_b[local_k_tile][i] = affine_b[offset];
                        }
                    }
                }
            }
        }

        if(!owns_output)
        {
            #pragma unroll
            for(int output_local = 0;
                output_local < LocalKTiles;
                ++output_local)
            {
                const int exchange_base =
                    ((((vhalf * 2 + output_khalf) * LocalKTiles +
                       output_local) * WAVE + lane) * 8);
                #pragma unroll
                for(int i = 0; i < 8; ++i)
                    exchange[exchange_base + i] = partial[output_local][i];
            }
        }
        // Keep the B-load scheduling experiment independent from barrier
        // scope.  The default path retains a complete workgroup/global-memory
        // fence; the strict opt-in diagnostic reproduces the prior LDS-only
        // exchange while owner waves may still have B VMEM loads outstanding.
        if constexpr(PostBExchangeLdsOnly)
            k2_context_gfx1250_sync_lds_only();
        else
            __syncthreads();

        if(owns_output)
        {
            #pragma unroll
            for(int local_k_tile = 0;
                local_k_tile < LocalKTiles;
                ++local_k_tile)
            {
                const int exchange_base =
                    ((((vhalf * 2 + output_khalf) * LocalKTiles +
                       local_k_tile) * WAVE + lane) * 8);
                const int state_base = exchange_base;
                #pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    const int v = v0 + (lane & 15);
                    const int k = output_k0 + local_k_tile * C +
                        (lane >> 4) * 8 + i;
                    const int64_t offset =
                        context_slab + static_cast<int64_t>(k) * D + v;
                    const float b = BPrefetchStateReuse
                        ? state[local_k_tile][i]
                        : (PostWmmaBPrefetch
                               ? prefetched_b[local_k_tile][i]
                               : affine_b[offset]);
                    if constexpr(!BPrefetchStateReuse)
                        affine_b[offset] = state[local_k_tile][i];
                    const float remote = exchange[exchange_base + i];
                    const float partial_low = output_khalf == 0
                        ? partial[local_k_tile][i] : remote;
                    const float partial_high = output_khalf == 0
                        ? remote : partial[local_k_tile][i];
                    next_state[state_base + i] =
                        (partial_low + partial_high) + b;
                }
            }
        }
        // The next affine-A copy and the newly reduced state can share one
        // publication rendezvous.  The LDS-only experiment still drains the
        // async A publisher above and the next-state DS stores inside
        // sync_lds_only(), but does not attach a global-memory fence to this
        // rendezvous.  Keep it compile-time gated so the established control
        // retains its exact synchronization topology.  The following
        // iteration's exchange barrier protects this single next-state arena
        // from reuse after every wave has consumed it.
        if constexpr(FusedPublish)
        {
            if(has_next_update)
                k2_context_gfx1250_wait_async_b128();
        }
        if constexpr(LdsOnlyPublishBarrier)
            k2_context_gfx1250_sync_lds_only();
        else
            __syncthreads();

        #pragma unroll
        for(int local_k_tile = 0;
            local_k_tile < LocalKTiles;
            ++local_k_tile)
        {
            const int state_base =
                ((((vhalf * 2 + source_khalf) * LocalKTiles +
                   local_k_tile) * WAVE + lane) * 8);
            #pragma unroll
            for(int i = 0; i < 8; ++i)
                state[local_k_tile][i] = next_state[state_base + i];
        }
        if constexpr(!DeferStateWait)
            k2_context_gfx1250_wait_lds();
        else if(!has_next_update)
            k2_context_gfx1250_wait_lds();
        if constexpr(!FusedPublish)
        {
            if(has_next_update)
                k2_context_gfx1250_wait_async_b128();
            // Publish the next affine-A buffer while protecting
            // exchange/next-state reuse by the following update.
            __syncthreads();
        }
    }

    if(skip_final_affine_update)
    {
        const int final_global_context = context_base + update_count;
        const int64_t final_context_slab =
            (static_cast<int64_t>(final_global_context) * h_count + head) *
            D * D;
        if(owns_output)
        {
            #pragma unroll
            for(int local_k_tile = 0;
                local_k_tile < LocalKTiles;
                ++local_k_tile)
            {
                #pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    const int v = v0 + (lane & 15);
                    const int k = output_k0 + local_k_tile * C +
                        (lane >> 4) * 8 + i;
                    const int64_t offset = final_context_slab +
                        static_cast<int64_t>(k) * D + v;
                    affine_b[offset] = state[local_k_tile][i];
                }
            }
        }
    }
}

// Four-wave counterpart of the strict-opt-in two-update scan.  b1 is kept in
// registers while A0 executes.  Once every wave has finished reading A0, the
// A1 direct-to-LDS publication overlaps reduction/publication of h1.  A single
// exchange arena is retained; the wait+barrier before the second contraction
// is also its cross-wave retirement fence.
template <int GroupChunks>
inline void launch_k2_kda_context_gfx1250_impl(
    const bf16* value,
    bf16* output,
    const bf16* ws_kd,
    const bf16* ws_qd,
    const bf16* ws_kr,
    const float* ws_decay,
    const bf16* ws_inv,
    const bf16* ws_mqk,
    const float* beta_cache,
    bf16* affine_a,
    float* affine_b,
    const void* initial_state,
    void* final_state,
    const int32_t* cu_seqlens,
    const int* tile_prefix,
    const int* context_prefix,
    int total_context_groups,
    int total_tiles,
    int t_seq,
    int h_count,
    int nseq,
    int nt_dense,
    bool has_state_in,
    bool has_state_out,
    bool state_fp32,
    hipStream_t stream)
{
    static_assert(
        GroupChunks == K2_CONTEXT_GFX1250_GROUP4 ||
            GroupChunks == K2_CONTEXT_GFX1250_GROUP8 ||
            GroupChunks == K2_CONTEXT_GFX1250_GROUP16,
        "production gfx1250 context routes support only G4/G8/G16");

    const bool varlen = cu_seqlens != nullptr;
    const uint64_t context_head_groups =
        static_cast<uint64_t>(total_context_groups) *
        static_cast<uint64_t>(h_count);
    if(context_head_groups == 0 ||
       context_head_groups > std::numeric_limits<unsigned int>::max())
        throw std::runtime_error(
            "FlashKDA gfx1250: context grid.x exceeds uint32_t");

    const dim3 group_grid(
        static_cast<unsigned int>(context_head_groups), V_TILES);
    const dim3 paired_2vtile_grid(
        static_cast<unsigned int>(context_head_groups), V_TILES / 2);

    if constexpr(GroupChunks == K2_CONTEXT_GFX1250_GROUP4 ||
                 GroupChunks == K2_CONTEXT_GFX1250_GROUP8)
    {
        const bool use_2vtile_4wave =
            k2_context_gfx1250_should_use_2vtile_4wave<GroupChunks>(
                varlen, nseq, t_seq, nt_dense, h_count);
        if(varlen)
        {
            if(use_2vtile_4wave)
                k2_kda_context_gfx1250_paired_ab_async_b128_2vtile_4wave_kernel<
                    true><<<paired_2vtile_grid, 4 * WAVE, 0, stream>>>(
                    value, ws_kd, ws_kr, ws_decay, ws_inv, beta_cache,
                    affine_a, affine_b, cu_seqlens, tile_prefix,
                    context_prefix, total_context_groups, total_tiles, 0,
                    h_count, nseq, 0, GroupChunks);
            else
                k2_kda_context_gfx1250_paired_ab_async_b128_kernel<true>
                    <<<group_grid, 2 * WAVE, 0, stream>>>(
                    value, ws_kd, ws_kr, ws_decay, ws_inv, beta_cache,
                    affine_a, affine_b, cu_seqlens, tile_prefix,
                    context_prefix, total_context_groups, total_tiles, 0,
                    h_count, nseq, 0, GroupChunks);
        }
        else
        {
            if(use_2vtile_4wave)
                k2_kda_context_gfx1250_paired_ab_async_b128_2vtile_4wave_kernel<
                    false><<<paired_2vtile_grid, 4 * WAVE, 0, stream>>>(
                    value, ws_kd, ws_kr, ws_decay, ws_inv, beta_cache,
                    affine_a, affine_b, nullptr, nullptr, nullptr,
                    total_context_groups, total_tiles, t_seq, h_count, nseq,
                    nt_dense, GroupChunks);
            else
                k2_kda_context_gfx1250_paired_ab_async_b128_kernel<false>
                    <<<group_grid, 2 * WAVE, 0, stream>>>(
                    value, ws_kd, ws_kr, ws_decay, ws_inv, beta_cache,
                    affine_a, affine_b, nullptr, nullptr, nullptr,
                    total_context_groups, total_tiles, t_seq, h_count, nseq,
                    nt_dense, GroupChunks);
        }
    }
    else
    {
        if(varlen)
            k2_kda_context_gfx1250_paired_ab_async_b128_kernel<true>
                <<<group_grid, 2 * WAVE, 0, stream>>>(
                value, ws_kd, ws_kr, ws_decay, ws_inv, beta_cache,
                affine_a, affine_b, cu_seqlens, tile_prefix, context_prefix,
                total_context_groups, total_tiles, 0, h_count, nseq, 0,
                GroupChunks);
        else
            k2_kda_context_gfx1250_paired_ab_async_b128_kernel<false>
                <<<group_grid, 2 * WAVE, 0, stream>>>(
                value, ws_kd, ws_kr, ws_decay, ws_inv, beta_cache,
                affine_a, affine_b, nullptr, nullptr, nullptr,
                total_context_groups, total_tiles, t_seq, h_count, nseq,
                nt_dense, GroupChunks);
    }

    const dim3 ksplit_scan_grid(
        static_cast<unsigned int>(nseq * h_count), V_TILES / 2);
    const bool use_lean_scan_8w =
        context_policy::should_use_lean_scan_8w(
            static_cast<uint32_t>(h_count), has_state_out);

    auto launch_state = [&]<bool HI, bool HO, bool FP, bool VL>() {
        if(use_lean_scan_8w)
        {
            if constexpr(HO)
                k2_kda_context_gfx1250_scan_ksplit8_apipeline_kernel<
                    GroupChunks, HI, HO, FP, VL,
                    false, true, false, true, true, false, false>
                    <<<ksplit_scan_grid, 8 * WAVE, 0, stream>>>(
                        affine_a, affine_b, initial_state, final_state,
                        VL ? cu_seqlens : nullptr,
                        VL ? context_prefix : nullptr,
                        total_context_groups, t_seq, h_count, nseq, nt_dense,
                        true);
            else
                throw std::runtime_error(
                    "FlashKDA gfx1250: internal 8-wave scan policy mismatch");
        }
        else
        {
            k2_kda_context_gfx1250_scan_ksplit_kernel<
                GroupChunks, HI, HO, FP, VL>
                <<<ksplit_scan_grid, 4 * WAVE, 0, stream>>>(
                    affine_a, affine_b, initial_state, final_state,
                    VL ? cu_seqlens : nullptr,
                    VL ? context_prefix : nullptr,
                    total_context_groups, t_seq, h_count, nseq, nt_dense,
                    true);
        }

        k2_kda_context_gfx1250_replay_async_b128_kernel<VL>
            <<<group_grid, WAVE, 0, stream>>>(
                value, output, ws_kd, ws_qd, ws_kr, ws_decay, ws_inv,
                ws_mqk, beta_cache, affine_b, final_state,
                VL ? cu_seqlens : nullptr,
                VL ? tile_prefix : nullptr,
                VL ? context_prefix : nullptr,
                total_context_groups, total_tiles, VL ? 0 : t_seq,
                h_count, nseq, VL ? 0 : nt_dense, GroupChunks, HO, FP);
    };
    dispatch_state_mode(
        varlen, has_state_in, has_state_out, state_fp32, launch_state);
}

inline void launch_k2_kda_context_gfx1250(
    int group_chunks,
    const bf16* value,
    bf16* output,
    const bf16* ws_kd,
    const bf16* ws_qd,
    const bf16* ws_kr,
    const float* ws_decay,
    const bf16* ws_inv,
    const bf16* ws_mqk,
    const float* beta_cache,
    bf16* affine_a,
    float* affine_b,
    const void* initial_state,
    void* final_state,
    const int32_t* cu_seqlens,
    const int* tile_prefix,
    const int* context_prefix,
    int total_context_groups,
    int total_tiles,
    int t_seq,
    int h_count,
    int nseq,
    int nt_dense,
    bool has_state_in,
    bool has_state_out,
    bool state_fp32,
    hipStream_t stream)
{
    if(group_chunks == K2_CONTEXT_GFX1250_GROUP4)
    {
        launch_k2_kda_context_gfx1250_impl<K2_CONTEXT_GFX1250_GROUP4>(
            value, output, ws_kd, ws_qd, ws_kr, ws_decay, ws_inv, ws_mqk,
            beta_cache, affine_a, affine_b, initial_state, final_state,
            cu_seqlens, tile_prefix, context_prefix, total_context_groups,
            total_tiles, t_seq, h_count, nseq, nt_dense, has_state_in,
            has_state_out, state_fp32, stream);
    }
    else if(group_chunks == K2_CONTEXT_GFX1250_GROUP8)
    {
        launch_k2_kda_context_gfx1250_impl<K2_CONTEXT_GFX1250_GROUP8>(
            value, output, ws_kd, ws_qd, ws_kr, ws_decay, ws_inv, ws_mqk,
            beta_cache, affine_a, affine_b, initial_state, final_state,
            cu_seqlens, tile_prefix, context_prefix, total_context_groups,
            total_tiles, t_seq, h_count, nseq, nt_dense, has_state_in,
            has_state_out, state_fp32, stream);
    }
    else if(group_chunks == K2_CONTEXT_GFX1250_GROUP16)
    {
        launch_k2_kda_context_gfx1250_impl<K2_CONTEXT_GFX1250_GROUP16>(
            value, output, ws_kd, ws_qd, ws_kr, ws_decay, ws_inv, ws_mqk,
            beta_cache, affine_a, affine_b, initial_state, final_state,
            cu_seqlens, tile_prefix, context_prefix, total_context_groups,
            total_tiles, t_seq, h_count, nseq, nt_dense, has_state_in,
            has_state_out, state_fp32, stream);
    }
    else
    {
        throw std::runtime_error(
            "FlashKDA gfx1250: unsupported context group size");
    }
}
