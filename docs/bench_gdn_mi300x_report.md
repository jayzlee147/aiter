# Gated-DeltaNet Prefill: Triton vs Opus Fused — MI300X (gfx942) 性能对比报告

两套 Gated-DeltaNet prefill 前向实现的逐 kernel 对比:

- **Triton** (`aiter.ops.triton._triton_kernels.gated_delta_rule`): Triton autotune, 4 kernel launch。
- **Opus** (`aiter.ops.opus_gdn_prefill`): 手写 HIP, 静态模板特化, 2 kernel launch (K1+K2)。

> 注意: 这是"算法 + 实现"两个变量混在一起的端到端对比。Triton=compiler + autotune, Opus=手写 HIP + MFMA intrinsics。

## 测试环境

- **设备**: AMD Instinct **MI300X** (**gfx942** / CDNA3), 304 CU (80 × 4 XCD), wave64, 512 VGPR/lane/SIMD, **64 KB LDS/CU**, 16× GPU, ROCm/HIP 7.2。
- **基准脚本**: `bench_gdn_mi300x.py`, `warmup=10 iters=20 with_init=True`, 时间取 **median**, 单位 **µs**。
- **数据口径**: 端到端 + 分段 (k14 前段, k56 后段), BT=64 默认; §3 给出 BT=32 vs BT=64 对照。
- **正确性**: opus 输出与 Triton 在 dense 下逐项比对, 误差在 bf16 容差内; 本报告只谈性能。

## 两条 Pipeline 与 Kernel 映射

| 阶段 | Triton (4 launch) | Opus HIP (2 launch) | 本报告记号 |
|---|---|---|---|
| cumsum + KKT + trisol + WY | K1(`cumsum`) + K2(`KKT`) + K3(`solve_tril`) + K4(`recompute_w_u`) | **K1** (`gdn_k1`, 融合 K1-K4) | **k14** |
| chunk-serial h 递推 + 输出 o | K5(`chunk_delta_h`) + K6(`chunk_o`) | **K2** (`gdn_k2`, 融合 K5+K6) | **k56** |

即: **opus k14 ↔ triton K1-K4**, **opus k56 ↔ triton K5+K6**。

模型配置: `35B-tp1` = K=V=128, H=32; `35B-tp2` = K=V=128, H=16; `FULL-tp1` = K=V=128, H=64; `FULL-tp2` = K=V=128, H=32。

比值约定: `opus/triton`, **< 1 表示 opus 更快**。

---

## TL;DR

1. **MI300X 上 opus 全面领先 Triton — 所有 44 个 case 全胜**, 与 MI35x 的"按 workload 切换"截然不同。
   - **Dense**: opus/triton 端到端 geomean **≈ 0.546** (即 **~1.83× 加速**)。范围 0.410–0.679。
   - **Packed Varlen**: opus/triton geomean **≈ 0.722** (即 **~1.38× 加速**)。范围 0.671–0.861。

2. **结构性差异 vs MI35x**: MI35x 上 dense 长序列 aiter K5 (Triton) 凭软件流水反超 opus K2; **MI300X 上这一优势不存在** — Triton K5 在 gfx942 缺乏 32x32x16 MFMA 加持, autotune 效果受限, opus K2 始终更快。

3. **分段看两后端差距方向一致**: opus 前段 k14 (~0.31–0.60×) 和后段 k56 都更快。前段是结构性优势 (LDS staging → 低寄存器 → 高占用); 后段在 gfx942 上 opus 反而也领先 (不同于 MI35x)。

4. **BT=64 仍是最优默认**: BT=32 比 BT=64 慢 ~27% (锚点 35B-tp1 T=8192: 2580 vs 2030 µs)。

5. **gfx942 独有限制**: 无 32x32x16 MFMA; split-K2 (ref_fwd_h OCC=1) 已验证更慢; wavefront fused 也更慢; 64KB LDS 限制 BT≤64。

---

## 1. Dense 端到端对比 (cu_seqlens=None, 单序列)

### 全量 28 case: opus 全胜 (28/28)

| 模型 | T | H | triton k14 | triton k5 | triton k6 | triton 总计 | opus 总计 | opus/triton |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 35B-tp1 | 1024 | 32 | 324 | 207 | 123 | 654 | **284** | **0.434** |
| 35B-tp1 | 2048 | 32 | 426 | 399 | 225 | 1050 | **495** | **0.472** |
| 35B-tp1 | 4096 | 32 | 680 | 697 | 475 | 1852 | **964** | **0.521** |
| 35B-tp1 | 8192 | 32 | 1230 | 1338 | 943 | 3512 | **2006** | **0.571** |
| 35B-tp1 | 16384 | 32 | 2489 | 2753 | 1862 | 7106 | **4058** | **0.571** |
| 35B-tp1 | 32768 | 32 | 4916 | 5404 | 3717 | 14034 | **8048** | **0.573** |
| 35B-tp1 | 65536 | 32 | 9817 | 10754 | 7447 | 28021 | **15997** | **0.571** |
| 35B-tp2 | 1024 | 16 | 255 | 165 | 85 | 504 | **207** | **0.410** |
| 35B-tp2 | 2048 | 16 | 329 | 211 | 137 | 675 | **341** | **0.505** |
| 35B-tp2 | 4096 | 16 | 438 | 408 | 263 | 1109 | **609** | **0.549** |
| 35B-tp2 | 8192 | 16 | 730 | 717 | 530 | 1977 | **1191** | **0.603** |
| 35B-tp2 | 16384 | 16 | 1305 | 1360 | 1078 | 3744 | **2500** | **0.668** |
| 35B-tp2 | 32768 | 16 | 2599 | 2736 | 2133 | 7469 | **5069** | **0.679** |
| 35B-tp2 | 65536 | 16 | 5190 | 5457 | 4244 | 14890 | **10063** | **0.676** |
| FULL-tp1 | 1024 | 64 | 400 | 406 | 226 | 1033 | **501** | **0.485** |
| FULL-tp1 | 2048 | 64 | 631 | 708 | 458 | 1797 | **964** | **0.536** |
| FULL-tp1 | 4096 | 64 | 1146 | 1362 | 950 | 3458 | **2019** | **0.584** |
| FULL-tp1 | 8192 | 64 | 2207 | 2907 | 1910 | 7024 | **4022** | **0.573** |
| FULL-tp1 | 16384 | 64 | 4422 | 5528 | 3791 | 13729 | **7995** | **0.582** |
| FULL-tp1 | 32768 | 64 | 9025 | 10969 | 7622 | 27605 | **15896** | **0.576** |
| FULL-tp1 | 65536 | 64 | 18069 | 22251 | 16211 | 56536 | **31963** | **0.565** |
| FULL-tp2 | 1024 | 32 | 314 | 207 | 123 | 643 | **284** | **0.441** |
| FULL-tp2 | 2048 | 32 | 424 | 398 | 224 | 1047 | **496** | **0.473** |
| FULL-tp2 | 4096 | 32 | 679 | 695 | 474 | 1848 | **962** | **0.521** |
| FULL-tp2 | 8192 | 32 | 1227 | 1337 | 945 | 3509 | **2024** | **0.577** |
| FULL-tp2 | 16384 | 32 | 2482 | 2750 | 1858 | 7089 | **4043** | **0.570** |
| FULL-tp2 | 32768 | 32 | 4929 | 5419 | 3732 | 14081 | **8064** | **0.573** |
| FULL-tp2 | 65536 | 32 | 9831 | 10740 | 7444 | 28012 | **16059** | **0.573** |

> **关键观察**: 随 T 增长, opus/triton 比值从 ~0.41 (T=1024) 上升到 ~0.57–0.68 (T=65536), 但始终 < 1。这说明 opus 在短序列的优势更大 (launch overhead 占比高, opus 2 launch vs Triton 4 launch), 长序列时差距收窄但 opus 仍胜。

### 锚点 (35B-tp1 T=8192, 即 B=1 H=32 的标准 config)

| 段 | triton | opus | opus/triton |
|---|---:|---:|---:|
| 前段 k14 | 1230 µs | — | — |
| 后段 k5+k6 | 1338+943=2281 µs | — | — |
| **总计** | **3512 µs** | **2006 µs** | **0.571** |

> opus 加速比 **1.75×**, 距 2× 目标 (1756 µs) 还差 ~250 µs。

---

## 2. Packed Varlen 端到端对比 (T_total=32768, N 条等长序列)

### 全量 16 case: opus 全胜 (16/16)

| 模型 | fpl | N | triton k14 | triton k5 | triton k6 | triton 总计 | opus 总计 | opus/triton |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 35B-tp1 | 1024 | 32 | 4313 | 3408 | 3730 | 11451 | **8074** | **0.705** |
| 35B-tp1 | 2048 | 16 | 4325 | 3304 | 3762 | 11393 | **8091** | **0.710** |
| 35B-tp1 | 4096 | 8 | 4295 | 3474 | 3759 | 11527 | **8072** | **0.700** |
| 35B-tp1 | 8192 | 4 | 4293 | 3966 | 3726 | 11987 | **8062** | **0.673** |
| 35B-tp2 | 1024 | 32 | 2266 | 1730 | 1892 | 5886 | **5070** | **0.861** |
| 35B-tp2 | 2048 | 16 | 2274 | 1798 | 1893 | 5966 | **5051** | **0.847** |
| 35B-tp2 | 4096 | 8 | 2265 | 2011 | 1889 | 6164 | **5061** | **0.821** |
| 35B-tp2 | 8192 | 4 | 2241 | 2000 | 1880 | 6123 | **5069** | **0.828** |
| FULL-tp1 | 1024 | 32 | 8925 | 6940 | 7597 | 23458 | **15909** | **0.678** |
| FULL-tp1 | 2048 | 16 | 9005 | 6752 | 7594 | 23350 | **15877** | **0.680** |
| FULL-tp1 | 4096 | 8 | 8852 | 6684 | 7586 | 23125 | **15866** | **0.686** |
| FULL-tp1 | 8192 | 4 | 8912 | 7163 | 7595 | 23674 | **15892** | **0.671** |
| FULL-tp2 | 1024 | 32 | 4302 | 3405 | 3722 | 11430 | **8074** | **0.706** |
| FULL-tp2 | 2048 | 16 | 4285 | 3305 | 3720 | 11310 | **8095** | **0.716** |
| FULL-tp2 | 4096 | 8 | 4312 | 3474 | 3721 | 11508 | **8074** | **0.702** |
| FULL-tp2 | 8192 | 4 | 4289 | 3967 | 3765 | 12021 | **8064** | **0.671** |

> **观察**: 与 MI35x 不同, varlen 下 opus 的领先幅度 (~0.67–0.86) **反而小于** dense (~0.41–0.68)。原因: Triton 在 packed batch 下 K5+K6 的并行度提升明显 (拆链效应), 而 opus 的 K2 改善幅度不如 dense→varlen 的差距大。

---

## 3. BT=32 vs BT=64 Opus 对比

锚点: 35B-tp1 (H=32, K=V=128), T=8192

| BT | opus 总计 (µs) | vs BT=64 |
|---:|---:|---:|
| 32 | 2580 | +27% |
| **64** | **2030** | baseline |

> BT=64 明确更优。与 MI35x 结论一致: chunk 数 ∝ 1/BT, 后段 K2 串行递推是主导项, BT 越大 chunk 越少越快。BT=128 在 gfx942 因 64KB LDS 限制不可用。

---

## 4. 分段分析

### 前段 k14: Opus 的结构性优势

- Opus K1 将 cumsum + KKT + 三角求逆 + WY 因子重算融合为 **单 kernel**
- 矩阵 staging 进 LDS, 寄存器仅 ~74 VGPR → 高占用
- Triton 分 4 个 kernel launch, 中间 A 矩阵落 HBM, launch overhead 叠加
- **K1 优化: Neumann squaring** (15→6 MFMA), 锚点 K1 已优化到 ~500 µs

### 后段 k56: 串行递推瓶颈 (K2 占总时延 ~73%)

- Opus K2 融合 h 递推 + 输出, T=8192/BT=64 = 128 chunk 串行, 每 chunk 14 `__syncthreads`
- Grid=64 WG on 304 CU = **21% 利用率**, LATENCY/OCC-BOUND
- PMC 实测 (锚点 config): MFMA 利用率 11.9%, HBM BW 7%, MemStall 0.4%
- 瓶颈是结构性的: 串行 chunk 链 + barrier-heavy 内循环, 不是 compute 或 BW

### 与 MI35x 对比: 为什么 MI300X 上 dense 也是 opus 胜

MI35x 上 dense 长序列 Triton K5 凭 `num_stages` 软件流水 + autotune 反超 opus K2 (~1.25–2.15×)。MI300X 上这一优势消失, 可能原因:

1. **gfx942 无 32x32x16 MFMA**: Triton K5 的 GEMM tile 效率更低
2. **gfx942 VGPR 限制更紧**: 512 VGPR/SIMD (vs MI35x 512), 但 64KB LDS (vs MI35x 160KB) 限制了 Triton autotune 选项
3. **Opus K2 在 gfx942 上 OCC=4**: 得益于手动寄存器管控, 而 Triton autotune 在 gfx942 可能选不到最优 tile

---

## 5. MI300X (gfx942) 资源与占用

| 指标 | Triton K1-K4 | Triton K5 | Triton K6 | Opus K1 | Opus K2 |
|---|---|---|---|---|---|
| Launch | 4 kernel | 1 kernel | 1 kernel | 1 kernel | 1 kernel |
| MFMA | 16×16×16 | 16×16×16 | 16×16×16 | 16×16×16 | 16×16×16 |
| VGPR/lane | varies | ~180 | ~120 | ~74 | ~120 |
| LDS/CU | varies | ~32 KB | ~16 KB | ~25 KB | ~24 KB |
| 软件流水 | N/A | `num_stages` | `num_stages` | 无 | 无 |

### gfx942 特殊限制

- **无 32x32x16 MFMA**: 仅 16×16×16, split-K2 的 ref_fwd_h 效率受限
- **ref_fwd_h OCC=1**: VGPR+AGPR=264 → 仅 1 wave/SIMD, split-K2 已验证更慢 (1453 vs fused 1388 µs)
- **Wavefront fused 更慢**: AGPR 128→132 导致 OCC 4→3, 原子同步开销, 已验证 S=128 (串行) 最快
- **64KB LDS 硬上限**: BT=128 不可用 (LDS 超限)
- **CU 数 304** (vs MI35x 256): 但 K2 grid=64 WG 在两者上都严重利用不足

---

## 6. 配置默认

- **BT=64** 为两后端最优默认 (chunk 数 ∝ 1/BT, 后段主导)
- **Opus K2**: BT=64 时 BV=64, num_warps=4 (gfx942 默认)
- **Opus K1**: 固定 4 warp, Neumann squaring 优化 (15→6 MFMA)
- **k2_mode=0** (fused): gfx942 唯一可用模式; k2_mode=2 (split) 已验证更慢
- **k1_algo=1**: 默认 Neumann squaring 路径

---

## 7. 结论与优化建议

### 关键结论

1. **MI300X 上 opus 全面领先 Triton (44/44 case)**, 与 MI35x "按 workload 切换" 不同。
2. **加速比**: Dense geomean 1.83×, Varlen geomean 1.38×。锚点 (B=1 T=8192 H=32) 达 **1.75×**, 距 2× 目标差 ~250 µs。
3. **K2 (后段) 仍是绝对瓶颈 (~73% 总时延)**: 串行 128-chunk 递推 + 21% CU 利用率。
4. **MI300X 不需要 "混血" 方案**: 不同于 MI35x 的 "opus-k14 + triton-k56" 建议, MI300X 上 opus 两段都更快。

### 后续优化方向

1. **K2 内循环 prefetch 流水**: 参考 Triton K5 的 `num_stages`, 在 HIP 层面实现 per-chunk 数据预取, 重叠 MFMA 与 HBM 读取。
2. **降 fp32 h 寄存器压力**: 当前 h-state 128×BV = 8192 个 fp32 值常驻寄存器, 探索 bf16 中间表示或 LDS spill。
3. **降 LDS bank 冲突**: profile bank conflict 率, 优化 padding。
4. **填充 CU 空闲**: Grid=64 WG / 304 CU = 21% 利用率; wavefront 方案在 gfx942 已验证无收益, 需要新的并行化思路。
5. **split-K2 仅在 gfx950 可行**: gfx942 的 ref_fwd_h OCC=1 是硬伤, 不宜在 gfx942 上追求 split。

---

## 复现

```bash
cd /home/sijieli2
python bench_gdn_mi300x.py
```

原始数据: `bench_gdn_mi300x_raw.json`
