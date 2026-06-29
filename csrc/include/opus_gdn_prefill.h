// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <torch/extension.h>

void opus_gdn_prefill_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor o,
    float scale,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    bool has_initial_state,
    bool output_final_state,
    int BT,
    int BV,
    int num_warps,
    int k1_algo,
    bool pipeline = false,
    int occ_hint = 0);

void opus_gdn_prefill_split_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor o,
    float scale,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    bool has_initial_state,
    bool output_final_state,
    int BT,
    int BV,
    int num_warps);

void opus_gdn_wavefront_h_fwd(
    torch::Tensor k,
    torch::Tensor w_bar,
    torch::Tensor u_bar,
    torch::Tensor g_cumsum,
    torch::Tensor h_out,
    torch::Tensor v_new_out,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    bool has_initial_state,
    bool output_final_state,
    int S,
    int BT,
    torch::Tensor q,
    torch::Tensor o,
    float scale);
