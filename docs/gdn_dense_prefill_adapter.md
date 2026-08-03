# GDN dense prefill 适配说明

本适配层将 dense GDN prefill 拆为两个可独立构建的后端：

- W/U 后端：`opus_gdn_wu_prefill_fwd`，包含 WF（W/U fused）与 WS（W/U split）。
- C 后端：`opus_gdn_c_prefill_fwd`，包含 CF（C fused）与 CS（C split）。

统一入口为：

```python
from aiter.ops.gdn_prefill import gdn_prefill

o, final_state = gdn_prefill(
    q,
    k,
    v,
    o=output_buffer,
    g=g,
    beta=beta,
    initial_state=initial_state,
    output_final_state=True,
    path="auto",
)
```

接口参数与 `chunk_gated_delta_rule_opt_vk` 保持一致，只在末尾增加关键字参数
`path`。可选值为 `auto`、`cf`、`cs`、`wf`、`ws`、`c`、`wu` 和
`triton`。模型侧若已有自己的 shape 策略，可直接指定四条底层路径；`triton`
用于强制走原实现。

## 自动路由范围

`auto` 只在 MI308X/gfx942、80 CU、BF16、K=V=128、T 为 64 的整数倍、
等长 dense 输入、FP32 state、无 decode prefix 的已验证范围内启用 Opus。
不满足条件时，所有参数和张量对象都会原样传给
`chunk_gated_delta_rule_opt_vk`，不会先做类型转换、连续化或输出分配。

对于 2026-07-31 closeout 的 477 个实测 shape，路由键为
`(B, T, H, state)`，并精确选择数据中的 CF/CS/WS/WF winner。不能把键压缩为
`(T, B*H, state)`：相同 `B*H` 的不同 B/H 分解已经出现过不同 winner。
只有 initial state 和 final state output 同时开启才对应 `state=on`；单边状态
I/O 没有测量数据，因此按 W/U 的固定 WS/WF 包络处理。

未命中 477 点表时，适配层使用代码内固定的 W/U 包络并强制选择 WS 或 WF，
不会调用读取 `OPUS_GDN_SPLIT_THRESHOLD` 的底层 auto mode。CF/CS 也通过显式
`c_mode` 选择，不依赖 `OPUS_GDN_K2C_SPLIT`。因此四路径 family 的选择是逐调用
确定的。统一入口还会关闭底层 backend 的 benchmark/debug 环境覆盖；即使模型
进程继承了 `OPUS_GDN_WF_VARIANT`、`OPUS_GDN_OUT_VARIANT`、
`OPUS_GDN_K2C_VARIANT` 等实验变量，也仍然执行 closeout 的发布配置。直接调用
W/U 或 C 底层 API 时，`use_env_overrides=True` 的默认值继续保留 A/B 能力。

## 数据来源

- 数据：`dense_closeout_all.json`
- SHA-256：`3e333710940d3c6d9feec6f8792fd500be5d0ac28423dc165ef80622cd712d89`
- 有效 shape：477
- winner：CF 196、CS 25、WS 182、WF 74
- 平台：AMD Instinct MI308X / gfx942 / 80 CU

路由数据以排序后的 32-bit packed records 固化在代码中，运行时不读取外部
JSON。内核或测量环境变化后，应重新跑完整矩阵并重新生成该表，不能按相邻
shape 静默插值。

## 当前边界

varlen、非 64 对齐的自动路径、BF16 state、GQA/MQA、内核内 Q/K L2Norm 和
decode prefix 仍走现有实现。显式 Opus 路径若输入不受支持会直接报错，不会
悄悄换成另一条实现。这样在后续切换到 varlen 时，dense 快速路径与 fallback
的语义边界保持清晰。
