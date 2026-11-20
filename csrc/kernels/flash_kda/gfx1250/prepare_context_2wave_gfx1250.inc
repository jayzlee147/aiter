// Shared function body for the default and context-specialized gfx1250
// prepare kernels. The including kernel supplies Varlen, ContextOperands,
// UseDppReduction, TotalExpHoist, PrepareVectorDrain, PrepareQkAsyncCopy,
// PrepareFactorVectorDrain, PrepareGatePairPrefix, the ordinary prepare
// arguments, and ws_beta.
    // Experimental WG64 context prepare.  Keep the native wave32 lane map
    // explicit: threadIdx.x is a workgroup thread id, while every shuffle and
    // WMMA fragment is indexed by the lane within its own wave.
    constexpr int PREP_WAVES = 2;
    constexpr int PREP_THREADS = PREP_WAVES * WAVE;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (WAVE - 1);
    const int wave = tid / WAVE;
    int head = 0;
    int ht = 0;
    int token0 = 0;
    int active = 0;
    if constexpr(Varlen)
    {
        const int global_tile = static_cast<int>(blockIdx.x);
        head = static_cast<int>(blockIdx.y);
        int lo = 0;
        int hi = nseq;
        while(hi - lo > 1)
        {
            const int mid = (lo + hi) >> 1;
            if(tile_prefix[mid] <= global_tile)
                lo = mid;
            else
                hi = mid;
        }
        const int tile_in_seq = global_tile - tile_prefix[lo];
        const int bos = cu_seqlens[lo];
        const int eos = cu_seqlens[lo + 1];
        const int seq_len = eos - bos;
        const int nt = (seq_len + C - 1) / C;
        if(tile_in_seq >= nt)
            return;
        token0 = bos + tile_in_seq * C;
        active = seq_len - tile_in_seq * C;
        active = active < C ? active : C;
        ht = head * total_tiles + global_tile;
    }
    else
    {
        const int nt = static_cast<int>(blockIdx.x);
        const int seq_head = static_cast<int>(blockIdx.y);
        const int seq = seq_head / h_count;
        head = seq_head - seq * h_count;
        const int chunks = static_cast<int>(gridDim.x);
        token0 = seq * t_seq + nt * C;
        active = t_seq - nt * C;
        active = active < C ? active : C;
        ht = seq_head * chunks + nt;
    }
    const int q_head = head / (h_count / h_q_count);

    alignas(16) __shared__ bf16 k_norm[C * KD_STRIDE];
    alignas(16) __shared__ bf16 q_norm[C * KD_STRIDE];
    __shared__ float gate_cumsum[C * D];
    alignas(16) __shared__ bf16 kd[C * KD_STRIDE];
    __shared__ bf16 k_inv[C * KD_STRIDE];
    alignas(16) __shared__ bf16 kr[C * KD_STRIDE];
    __shared__ float g_total[D];
    __shared__ float beta_act[C];
    __shared__ _Float16 lower[C * C];
    alignas(16) __shared__ float inverse[C * C];
    alignas(16) __shared__ bf16 mqk[C * C];

    // Direct global-to-LDS Q/K publication is deliberately isolated from the
    // gate arithmetic.  Rebase the global handles to this tile/head so the
    // gfx1250 async instruction sees only a small, tile-local 32-bit offset.
    // Every thread owns exactly four b128 vectors.  Tail rows are explicitly
    // zero-filled in LDS, preserving the historical padded C16 tile contract.
    if constexpr(PrepareQkAsyncCopy)
    {
        constexpr int QK_VECTOR = 8;
        constexpr int QK_VECTORS_PER_ROW = D / QK_VECTOR;
        constexpr int QK_TRIPS_PER_THREAD =
            C * QK_VECTORS_PER_ROW / PREP_THREADS;
        static_assert(D % QK_VECTOR == 0);
        static_assert(QK_VECTOR * sizeof(bf16) == 16);
        static_assert((D * sizeof(bf16)) % 16 == 0);
        static_assert((KD_STRIDE * sizeof(bf16)) % 16 == 0);
        static_assert(
            C * QK_VECTORS_PER_ROW ==
            PREP_THREADS * QK_TRIPS_PER_THREAD);
        static_assert(QK_TRIPS_PER_THREAD == 4);

        const int64_t qk_base =
            (static_cast<int64_t>(token0) * h_q_count + q_head) * D;
        const int qk_row_stride = h_q_count * D;
        auto k_global = opus::make_gmem(k + qk_base);
        auto q_global = opus::make_gmem(q + qk_base);
        auto k_lds = opus::make_smem(k_norm);
        auto q_lds = opus::make_smem(q_norm);
        const bf16 zero = f32_to_bf16(0.0f);
        #pragma unroll
        for(int trip = 0; trip < QK_TRIPS_PER_THREAD; ++trip)
        {
            const int vector = tid + trip * PREP_THREADS;
            const int row = vector / QK_VECTORS_PER_ROW;
            const int col =
                (vector - row * QK_VECTORS_PER_ROW) * QK_VECTOR;
            const int sidx = row * KD_STRIDE + col;
            if(row < active)
            {
                const int source = row * qk_row_stride + col;
                k_global.async_load<QK_VECTOR>(k_norm + sidx, source);
                q_global.async_load<QK_VECTOR>(q_norm + sidx, source);
            }
            else
            {
                k_lds.store<QK_VECTOR>(zero, sidx);
                q_lds.store<QK_VECTOR>(zero, sidx);
            }
        }
    }

    const float a = ex2(a_log[head] * LOG2E);
    for(int idx = tid; idx < C * D; idx += PREP_THREADS)
    {
        const int row = idx / D;
        const int col = idx - row * D;
        const int sidx = row * KD_STRIDE + col;
        if(row < active)
        {
            if constexpr(!PrepareQkAsyncCopy)
            {
                const int64_t qk_off =
                    (static_cast<int64_t>(token0 + row) * h_q_count + q_head) *
                        D + col;
                const int64_t vg_off =
                    (static_cast<int64_t>(token0 + row) * h_count + head) * D + col;
                k_norm[sidx] = k[qk_off];
                q_norm[sidx] = q[qk_off];
                if constexpr(!PrepareGatePairPrefix)
                {
                    const float raw_gate =
                        bf16_to_f32(g[vg_off]) + dt_bias[head * D + col];
                    gate_cumsum[idx] =
                        gate_scale * sigmoid_tanh(a * raw_gate);
                }
            }
            else
            {
                const int64_t vg_off =
                    (static_cast<int64_t>(token0 + row) * h_count + head) * D + col;
                if constexpr(!PrepareGatePairPrefix)
                {
                    const float raw_gate =
                        bf16_to_f32(g[vg_off]) + dt_bias[head * D + col];
                    gate_cumsum[idx] =
                        gate_scale * sigmoid_tanh(a * raw_gate);
                }
            }
        }
        else
        {
            if constexpr(!PrepareQkAsyncCopy)
            {
                k_norm[sidx] = f32_to_bf16(0.0f);
                q_norm[sidx] = f32_to_bf16(0.0f);
            }
            if constexpr(!PrepareGatePairPrefix)
                gate_cumsum[idx] = 0.0f;
        }
    }
    if constexpr(PrepareGatePairPrefix)
    {
        // One thread owns two adjacent columns.  Rebase both global handles
        // to this head/tile, load the invariant bias pair once, then fuse the
        // gate transform with its row-ordered prefix scan.  The two scalar
        // dependency chains preserve the historical per-column add order;
        // paired b32/b64 memory operations only reduce instruction count.
        constexpr int GATE_PAIR = 2;
        static_assert(D == PREP_THREADS * GATE_PAIR);
        static_assert(GATE_PAIR * sizeof(bf16) == 4);
        static_assert(GATE_PAIR * sizeof(float) == 8);
        static_assert((D * sizeof(bf16)) % 4 == 0);
        static_assert((D * sizeof(float)) % 8 == 0);

        const int col = tid * GATE_PAIR;
        const int64_t gate_base =
            (static_cast<int64_t>(token0) * h_count + head) * D;
        const int gate_row_stride = h_count * D;
        auto gate_gmem = opus::make_gmem(g + gate_base);
        auto bias_gmem = opus::make_gmem(dt_bias + head * D);
        auto prefix_lds = opus::make_smem(gate_cumsum);
        auto total_lds = opus::make_smem(g_total);
        const auto bias_pair = bias_gmem.load<GATE_PAIR>(col);
        float running0 = 0.0f;
        float running1 = 0.0f;
        // The sigmoid/tanh body is large.  Keeping this fixed 16-row loop
        // rolled avoids replicating it sixteen times in every gate-pair
        // specialization and protects the Prepare I-cache footprint.
        #pragma nounroll
        for(int row = 0; row < C; ++row)
        {
            float gate0 = 0.0f;
            float gate1 = 0.0f;
            if(row < active)
            {
                const auto gate_pair =
                    gate_gmem.load<GATE_PAIR>(row * gate_row_stride + col);
                const float raw0 = bf16_to_f32(gate_pair[0]) + bias_pair[0];
                const float raw1 = bf16_to_f32(gate_pair[1]) + bias_pair[1];
                gate0 = gate_scale * sigmoid_tanh(a * raw0);
                gate1 = gate_scale * sigmoid_tanh(a * raw1);
            }
            running0 += gate0;
            running1 += gate1;
            const opus::vector_t<float, GATE_PAIR> prefix = {
                running0, running1};
            prefix_lds.store<GATE_PAIR>(prefix, row * D + col);
        }
        const opus::vector_t<float, GATE_PAIR> total = {
            TotalExpHoist ? ex2(running0) : running0,
            TotalExpHoist ? ex2(running1) : running1};
        total_lds.store<GATE_PAIR>(total, col);
    }
    // Interleave the sixteen independent beta rows over both waves.  The
    // scalar sigmoid for each row is unchanged.
    if(lane < C / PREP_WAVES)
    {
        const int row = lane * PREP_WAVES + wave;
        beta_act[row] = row < active
            ? sigmoid_tanh(beta[static_cast<int64_t>(token0 + row) * h_count + head])
            : 0.0f;
    }
    if constexpr(PrepareQkAsyncCopy)
        k2_context_gfx1250_wait_async_b128();
    __syncthreads();

    // Four elements per lane cover K=128 on wave32.
    #pragma unroll
    // Each wave owns alternating rows.  Within an owned row, the four scalar
    // products per lane and the XOR reduction tree are bit-for-bit identical
    // to the original wave32 kernel.
    for(int row = wave; row < C; row += PREP_WAVES)
    {
        float kv[4];
        float qv[4];
        float sum = 0.0f;
        #pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const int col = lane + i * WAVE;
            kv[i] = bf16_to_f32(k_norm[row * KD_STRIDE + col]);
            qv[i] = bf16_to_f32(q_norm[row * KD_STRIDE + col]);
            sum += kv[i] * kv[i];
        }
        float k_inv_norm;
        float q_inv_norm;
        if constexpr(UseDppReduction)
            k_inv_norm =
                rsqrtf(wave_reduce_sum_prepare_dpp(sum) + 1.0e-6f);
        else
            k_inv_norm = rsqrtf(wave_reduce_sum(sum) + 1.0e-6f);
        sum = 0.0f;
        #pragma unroll
        for(int i = 0; i < 4; ++i)
            sum += qv[i] * qv[i];
        if constexpr(UseDppReduction)
            q_inv_norm =
                rsqrtf(wave_reduce_sum_prepare_dpp(sum) + 1.0e-6f);
        else
            q_inv_norm = rsqrtf(wave_reduce_sum(sum) + 1.0e-6f);
        #pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const int col = lane + i * WAVE;
            k_norm[row * KD_STRIDE + col] = f32_to_bf16(kv[i] * k_inv_norm);
            q_norm[row * KD_STRIDE + col] = f32_to_bf16(qv[i] * q_inv_norm);
        }
    }

    if constexpr(!PrepareGatePairPrefix)
    {
        for(int col = tid; col < D; col += PREP_THREADS)
        {
            float running = 0.0f;
            #pragma unroll
            for(int row = 0; row < C; ++row)
            {
                running += gate_cumsum[row * D + col];
                gate_cumsum[row * D + col] = running;
            }
            if constexpr(TotalExpHoist)
                g_total[col] = ex2(running);
            else
                g_total[col] = running;
        }
    }
    __syncthreads();

    const bf16 scale_bf = f32_to_bf16(scale);
    for(int idx = tid; idx < C * D; idx += PREP_THREADS)
    {
        const int row = idx / D;
        const int col = idx % D;
        const int sidx = row * KD_STRIDE + col;
        const float gc = gate_cumsum[idx];
        const bf16 positive = f32_to_bf16(ex2(gc));
        const bf16 negative = f32_to_bf16(ex2(-gc));
        bf16 total;
        if constexpr(TotalExpHoist)
            total = f32_to_bf16(g_total[col]);
        else
            total = f32_to_bf16(ex2(g_total[col]));
        const float kn = bf16_to_f32(k_norm[sidx]);
        const float qn = bf16_to_f32(q_norm[sidx]);
        const bf16 kd_value = f32_to_bf16(kn * bf16_to_f32(positive));
        const bf16 ki_value = f32_to_bf16(kn * bf16_to_f32(negative));
        kd[sidx] = kd_value;
        k_inv[sidx] = ki_value;
        kr[sidx] = f32_to_bf16(bf16_to_f32(ki_value) * bf16_to_f32(total));
        const bf16 q_tmp = f32_to_bf16(qn * bf16_to_f32(positive));
        q_norm[sidx] = f32_to_bf16(bf16_to_f32(q_tmp) * bf16_to_f32(scale_bf));
    }
    __syncthreads();

    // The two contractions are independent but share k_inv.  A register
    // fragment cannot cross a wave boundary, so each wave reloads the same B
    // fragment from LDS and retains only one accumulator.  Panel order within
    // each contraction remains k0={0,32,64,96}.
    const Acc product = wave == 0
        ? contract_last_128<KD_STRIDE, KD_STRIDE>(kd, k_inv, lane)
        : contract_last_128<KD_STRIDE, KD_STRIDE>(q_norm, k_inv, lane);
    #pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        const int row = (lane >> 4) * 8 + i;
        const int col = lane & 15;
        if(wave == 0)
        {
            const _Float16 l = row > col
                ? static_cast<_Float16>(product[i]) *
                      static_cast<_Float16>(beta_act[row])
                : static_cast<_Float16>(0.0f);
            lower[row * C + col] = l;
        }
        else
        {
            mqk[row * C + col] = row >= col
                ? f32_to_bf16(product[i]) : f32_to_bf16(0.0f);
        }
    }
    __syncthreads();

    // Exact forward substitution for (I + L)^-1.  One lane owns one output
    // column; rows are naturally dependent but columns are independent.  Use
    // even/odd columns so the longest dependency chains (columns 0 and 1) are
    // balanced across the two waves.  The row and j order of every column is
    // unchanged.
    if(lane < C / PREP_WAVES)
    {
        const int col = lane * PREP_WAVES + wave;
        #pragma unroll
        for(int row = 0; row < C; ++row)
        {
            float value = row == col ? 1.0f : 0.0f;
            if(col < row)
            {
                #pragma unroll
                for(int j = 0; j < C; ++j)
                {
                    if(j >= col && j < row)
                        value -= static_cast<float>(lower[row * C + j]) *
                                 inverse[j * C + col];
                }
            }
            inverse[row * C + col] = col <= row ? value : 0.0f;
        }
    }
    __syncthreads();

    if constexpr(PrepareVectorDrain)
    {
        // Every padded LDS row begins on a 16-byte boundary
        // (KD_STRIDE=136 BF16), as does every C16xD workspace tile.  Drain
        // eight BF16 values per instruction while keeping the large ht term
        // in the buffer base so the per-thread buffer offset remains small.
        constexpr int DRAIN_VECTOR = 8;
        constexpr int DRAIN_VECTORS_PER_ROW = D / DRAIN_VECTOR;
        static_assert(D % DRAIN_VECTOR == 0);
        static_assert(DRAIN_VECTOR * sizeof(bf16) == 16);
        static_assert((KD_STRIDE * sizeof(bf16)) % 16 == 0);
        static_assert((D * sizeof(bf16)) % 16 == 0);
        const int64_t tile_base = static_cast<int64_t>(ht) * C * D;
        auto kd_lds = opus::make_smem(kd);
        auto qd_lds = opus::make_smem(q_norm);
        auto kr_lds = opus::make_smem(kr);
        auto kd_gmem = opus::make_gmem(ws_kd + tile_base);
        auto qd_gmem = opus::make_gmem(ws_qd + tile_base);
        auto kr_gmem = opus::make_gmem(ws_kr + tile_base);
        for(int vector = tid;
            vector < C * DRAIN_VECTORS_PER_ROW;
            vector += PREP_THREADS)
        {
            const int row = vector / DRAIN_VECTORS_PER_ROW;
            const int col =
                (vector - row * DRAIN_VECTORS_PER_ROW) * DRAIN_VECTOR;
            const int sidx = row * KD_STRIDE + col;
            const int out = row * D + col;
            kd_gmem.store<DRAIN_VECTOR>(
                kd_lds.load<DRAIN_VECTOR>(sidx), out);
            qd_gmem.store<DRAIN_VECTOR>(
                qd_lds.load<DRAIN_VECTOR>(sidx), out);
            kr_gmem.store<DRAIN_VECTOR>(
                kr_lds.load<DRAIN_VECTOR>(sidx), out);
        }
    }
    else
    {
        for(int idx = tid; idx < C * D; idx += PREP_THREADS)
        {
            const int row = idx / D;
            const int col = idx - row * D;
            const int sidx = row * KD_STRIDE + col;
            ws_kd[static_cast<int64_t>(ht) * C * D + idx] = kd[sidx];
            ws_qd[static_cast<int64_t>(ht) * C * D + idx] = q_norm[sidx];
            ws_kr[static_cast<int64_t>(ht) * C * D + idx] = kr[sidx];
        }
    }
    for(int col = tid; col < D; col += PREP_THREADS)
    {
        if constexpr(ContextOperands && !TotalExpHoist)
            ws_gt[static_cast<int64_t>(ht) * D + col] = ex2(g_total[col]);
        else
            ws_gt[static_cast<int64_t>(ht) * D + col] = g_total[col];
    }
    if constexpr(ContextOperands)
    {
        if(lane < C / PREP_WAVES)
        {
            const int row = lane * PREP_WAVES + wave;
            ws_beta[static_cast<int64_t>(ht) * C + row] = beta_act[row];
        }
    }
    if constexpr(PrepareFactorVectorDrain)
    {
        // The C16xC16 factors contain exactly one d8 vector per lane.  Give
        // inverse to wave 0 and mqk to wave 1 so both full tiles drain with
        // aligned b128 LDS/global transactions and no tail predicate.
        constexpr int FACTOR_VECTOR = 8;
        constexpr int FACTOR_VECTORS = C * C / FACTOR_VECTOR;
        static_assert(FACTOR_VECTOR * sizeof(bf16) == 16);
        static_assert(4 * sizeof(float) == 16);
        static_assert(FACTOR_VECTORS == WAVE);
        static_assert(PREP_THREADS == 2 * FACTOR_VECTORS);
        static_assert((C * sizeof(float)) % 16 == 0);
        static_assert((C * sizeof(bf16)) % 16 == 0);
        static_assert((C * C * sizeof(bf16)) % 16 == 0);

        const int idx = lane * FACTOR_VECTOR;
        const int64_t tile_base = static_cast<int64_t>(ht) * C * C;
        if(wave == 0)
        {
            auto inverse_lds = opus::make_smem(inverse);
            auto inverse_gmem = opus::make_gmem(ws_inv + tile_base);
            const auto inverse_lo = inverse_lds.load<4>(idx);
            const auto inverse_hi = inverse_lds.load<4>(idx + 4);
            opus::vector_t<uint32_t, 4> packed_words;
            packed_words[0] = pack_f32x2_to_bf16x2_prepare(
                inverse_lo[0], inverse_lo[1]);
            packed_words[1] = pack_f32x2_to_bf16x2_prepare(
                inverse_lo[2], inverse_lo[3]);
            packed_words[2] = pack_f32x2_to_bf16x2_prepare(
                inverse_hi[0], inverse_hi[1]);
            packed_words[3] = pack_f32x2_to_bf16x2_prepare(
                inverse_hi[2], inverse_hi[3]);
            const auto packed = __builtin_bit_cast(
                opus::vector_t<bf16, FACTOR_VECTOR>, packed_words);
            inverse_gmem.store<FACTOR_VECTOR>(packed, idx);
        }
        else
        {
            auto mqk_lds = opus::make_smem(mqk);
            auto mqk_gmem = opus::make_gmem(ws_mqk + tile_base);
            mqk_gmem.store<FACTOR_VECTOR>(
                mqk_lds.load<FACTOR_VECTOR>(idx), idx);
        }
    }
    else
    {
        for(int idx = tid; idx < C * C; idx += PREP_THREADS)
        {
            ws_inv[static_cast<int64_t>(ht) * C * C + idx] = f32_to_bf16(inverse[idx]);
            ws_mqk[static_cast<int64_t>(ht) * C * C + idx] = mqk[idx];
        }
    }
