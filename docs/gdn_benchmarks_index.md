# GDN Prefill 性能文档索引

Gated-DeltaNet (GDN) prefill 前向 kernel 的性能对比与优化文档汇总。对比对象:**Triton (aiter autotune)** vs **Opus (手写 HIP 融合,K1+K2 两 launch)**。

## 文档列表

| 文档 | 平台 | 内容 |
|---|---|---|
| [bench_gdn_mi300x_report.md](bench_gdn_mi300x_report.md) | **MI300X / gfx942 (CDNA3)** | 逐 kernel 对比;opus 全胜 44/44(dense geomean 1.83×,varlen 1.38×) |
| [bench_gdn_mi350_report.md](bench_gdn_mi350_report.md) | **MI350 / gfx950 (CDNA4)** | 同一配置矩阵;**按 workload 切换**(dense opus 胜 17/28,geomean 0.788) |
| [gdn_2x_optimization_analysis.md](gdn_2x_optimization_analysis.md) | 通用 | 2× 目标优化分析(K1 Neumann 平方法、K2 split-scan、瓶颈拆解) |

两份 bench 报告由同一脚本生成(`op_tests/triton_tests/bench_gdn_mi3{00x,50}.py`),配置矩阵一致,可直接横向对比。

## 一句话结论

| 维度 | MI300X (gfx942) | MI350 (gfx950) |
|---|---|---|
| 32×32×16 MFMA | 无 | **有** → Triton K5 scan 更强 |
| Dense 端到端 | opus 全胜(1.5–1.9×) | 按 workload 切换(短/头多 opus 胜,长/头少 Triton 胜) |
| 后段 k56 长序列 | opus K2 赢 | **Triton K5 反超(~1.4–1.5×)** |
| 前段 k14 | opus 恒赢 ~2.8×(VGPR=36,Neumann 平方法) | 同左 |
| 最优策略 | opus 两段全用 | **混血:opus-k14 + Triton-k56** |
| LDS / BT 上限 | 64 KB / BT≤64 | 160 KB / BT≤128 |
| BT=32 退化 | +27% | +226% |

**共同点**:两平台 **BT=64 都是最优默认**;opus 前段 K1 都凭"单 kernel 融合 + LDS staging + 低寄存器(VGPR=36)"结构性领先 Triton。
**关键差异**:gfx950 的 32×32×16 MFMA 让 Triton 后段 scan 变强,从而长序列总账反超 opus —— 这是 gfx950 需要"混血"、而 gfx942 不需要的根因。

## 复现

```bash
cd /home/sijieli2/aiter
python op_tests/triton_tests/bench_gdn_mi300x.py   # gfx942
python op_tests/triton_tests/bench_gdn_mi350.py    # gfx950
```
