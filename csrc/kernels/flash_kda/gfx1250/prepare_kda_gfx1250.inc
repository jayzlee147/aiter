// Shared function body for the default and context-specialized gfx1250
// prepare kernels. The including kernel supplies Varlen, ContextOperands,
// the ordinary prepare arguments, and ws_beta.
    const int lane = static_cast<int>(threadIdx.x);
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

    __shared__ bf16 k_norm[C * KD_STRIDE];
    __shared__ bf16 q_norm[C * KD_STRIDE];
    __shared__ float gate_cumsum[C * D];
    __shared__ bf16 kd[C * KD_STRIDE];
    __shared__ bf16 k_inv[C * KD_STRIDE];
    __shared__ bf16 kr[C * KD_STRIDE];
    __shared__ float g_total[D];
    __shared__ float beta_act[C];
    __shared__ _Float16 lower[C * C];
    __shared__ float inverse[C * C];
    __shared__ bf16 mqk[C * C];

    const float a = ex2(a_log[head] * LOG2E);
    for(int idx = lane; idx < C * D; idx += WAVE)
    {
        const int row = idx / D;
        const int col = idx - row * D;
        const int sidx = row * KD_STRIDE + col;
        if(row < active)
        {
            const int64_t q_off =
                static_cast<int64_t>(token0 + row) * q_token_stride +
                static_cast<int64_t>(q_head) * D + col;
            const int64_t k_off =
                static_cast<int64_t>(token0 + row) * k_token_stride +
                static_cast<int64_t>(q_head) * D + col;
            const int64_t vg_off =
                static_cast<int64_t>(token0 + row) * g_token_stride +
                static_cast<int64_t>(head) * D + col;
            k_norm[sidx] = k[k_off];
            q_norm[sidx] = q[q_off];
            const float raw_gate = bf16_to_f32(g[vg_off]) + dt_bias[head * D + col];
            gate_cumsum[idx] = gate_scale * sigmoid_tanh(a * raw_gate);
        }
        else
        {
            k_norm[sidx] = f32_to_bf16(0.0f);
            q_norm[sidx] = f32_to_bf16(0.0f);
            gate_cumsum[idx] = 0.0f;
        }
    }
    if(lane < C)
    {
        beta_act[lane] = lane < active
            ? sigmoid_tanh(beta[static_cast<int64_t>(token0 + lane) * h_count + head])
            : 0.0f;
    }
    __syncthreads();

    // Four elements per lane cover K=128 on wave32.
    #pragma unroll
    for(int row = 0; row < C; ++row)
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
        const float k_inv_norm = rsqrtf(wave_reduce_sum(sum) + 1.0e-6f);
        sum = 0.0f;
        #pragma unroll
        for(int i = 0; i < 4; ++i)
            sum += qv[i] * qv[i];
        const float q_inv_norm = rsqrtf(wave_reduce_sum(sum) + 1.0e-6f);
        #pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const int col = lane + i * WAVE;
            k_norm[row * KD_STRIDE + col] = f32_to_bf16(kv[i] * k_inv_norm);
            q_norm[row * KD_STRIDE + col] = f32_to_bf16(qv[i] * q_inv_norm);
        }
    }

    for(int col = lane; col < D; col += WAVE)
    {
        float running = 0.0f;
        #pragma unroll
        for(int row = 0; row < C; ++row)
        {
            running += gate_cumsum[row * D + col];
            gate_cumsum[row * D + col] = running;
        }
        g_total[col] = running;
    }
    __syncthreads();

    const bf16 scale_bf = f32_to_bf16(scale);
    for(int idx = lane; idx < C * D; idx += WAVE)
    {
        const int row = idx / D;
        const int col = idx % D;
        const int sidx = row * KD_STRIDE + col;
        const float gc = gate_cumsum[idx];
        const bf16 positive = f32_to_bf16(ex2(gc));
        const bf16 negative = f32_to_bf16(ex2(-gc));
        const bf16 total = f32_to_bf16(ex2(g_total[col]));
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

    const AccPair products =
        contract_pair_last_128<KD_STRIDE, KD_STRIDE, KD_STRIDE>(
            kd, q_norm, k_inv, lane);
    const Acc l_acc = products.first;
    const Acc m_acc = products.second;
    #pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        const int row = (lane >> 4) * 8 + i;
        const int col = lane & 15;
        const _Float16 l = row > col
            ? static_cast<_Float16>(l_acc[i]) * static_cast<_Float16>(beta_act[row])
            : static_cast<_Float16>(0.0f);
        lower[row * C + col] = l;
        mqk[row * C + col] = row >= col
            ? f32_to_bf16(m_acc[i]) : f32_to_bf16(0.0f);
    }
    __syncthreads();

    // Exact forward substitution for (I + L)^-1.  One lane owns one output
    // column; rows are naturally dependent but columns are independent.
    if(lane < C)
    {
        const int col = lane;
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

    for(int idx = lane; idx < C * D; idx += WAVE)
    {
        const int row = idx / D;
        const int col = idx - row * D;
        const int sidx = row * KD_STRIDE + col;
        ws_kd[static_cast<int64_t>(ht) * C * D + idx] = kd[sidx];
        ws_qd[static_cast<int64_t>(ht) * C * D + idx] = q_norm[sidx];
        ws_kr[static_cast<int64_t>(ht) * C * D + idx] = kr[sidx];
    }
    for(int col = lane; col < D; col += WAVE)
    {
        ws_gt[static_cast<int64_t>(ht) * D + col] = ContextOperands
            ? ex2(g_total[col]) : g_total[col];
    }
    if constexpr(ContextOperands)
    {
        if(lane < C)
            ws_beta[static_cast<int64_t>(ht) * C + lane] = beta_act[lane];
    }
    for(int idx = lane; idx < C * C; idx += WAVE)
    {
        ws_inv[static_cast<int64_t>(ht) * C * C + idx] = f32_to_bf16(inverse[idx]);
        ws_mqk[static_cast<int64_t>(ht) * C * C + idx] = mqk[idx];
    }
