# Gated-DeltaNet Prefill: Triton vs Opus Fused — MI350 (gfx950) 性能对比报告

两套 Gated-DeltaNet prefill 前向实现的逐 kernel 对比(MI300X 报告的姊妹篇,见 [`bench_gdn_mi300x_report.md`](bench_gdn_mi300x_report.md)):

- **Triton** (`aiter.ops.triton._triton_kernels.gated_delta_rule`): Triton autotune, 4 kernel launch。
- **Opus** (`aiter.ops.opus_gdn_prefill`): 手写 HIP, 静态模板特化, 2 kernel launch (K1+K2)。

> 注意: 这是"算法 + 实现"两个变量混在一起的端到端对比。Triton=compiler + autotune, Opus=手写 HIP + MFMA intrinsics。

## 测试环境

- **设备**: AMD Instinct **MI350 / MI35x** (**gfx950** / CDNA4), 256 CU, wave64, 512 VGPR/lane/SIMD, **160 KB LDS/CU**, 统一 VGPR/AGPR, ROCm/HIP。
- **基准脚本**: `op_tests/triton_tests/bench_gdn_mi350.py`, `warmup=10 iters=20 with_init=True`, 时间取 **median** (CUDA event), 单位 **µs**。
- **逐段 GPU 时间**: 另用 `rocprofv3 --kernel-trace` 采集 (§4 锚点),与 event 口径略有差异。
- **数据口径**: 端到端 + Triton 分段 (k14 前段, k5/k6 后段), BT=64 默认; §3 给出 BT=32 vs BT=64 对照。
- **正确性**: opus 输出与 Triton 在 dense 下逐项比对, 误差在 bf16 容差内; 本报告只谈性能。
- **代码**: aiter 分支 `fgdn_prefill_neumann` (K1 Neumann 平方法 + K2 gfx950 split-scan)。

## 两条 Pipeline 与 Kernel 映射

| 阶段 | Triton (4 launch) | Opus HIP (2 launch) | 本报告记号 |
|---|---|---|---|
| cumsum + KKT + trisol + WY | K1(`cumsum`) + K2(`KKT`) + K3(`solve_tril`) + K4(`recompute_w_u`) | **K1** (`gdn_k1_neumann`, 融合) | **k14** |
| chunk-serial h 递推 + 输出 o | K5(`chunk_delta_h`) + K6(`chunk_o`) | **K2** (split: `chunk_gated_delta_rule_fwd_h` scan + `gdn_k2_out`) | **k56** |

即: **opus k14 ↔ triton K1-K4**, **opus k56 ↔ triton K5+K6**。

模型配置 (dense, 单 H, K=V=128): `35B-tp1` = H=32; `35B-tp2` = H=16; `FULL-tp1` = H=64; `FULL-tp2` = H=32。

比值约定: `opus/triton`, **< 1 表示 opus 更快**。

---

## TL;DR

1. **MI350 上没有单一赢家 — 按 workload 切换**(与 MI300X 的 opus "全胜 44/44" 截然不同)。
   - **Dense**: opus 胜 **17/28**, geomean opus/triton **≈ 0.788**。短序列 / 头多时 opus 大胜(最快 0.333),长序列 / 头少时 Triton 反超(最慢 1.469)。
   - **Packed Varlen** (T_total=32768): opus 仅胜 **2/16**, geomean **≈ 1.30** —— Triton 在 gfx950 上拿下 varlen(见 §2 口径说明)。

2. **结构性根因 vs MI300X**: gfx950 **有 32x32x16 MFMA**,Triton 的 K5 (`chunk_delta_h`) scan 在长序列下效率大增,**dense 长序列 K5 凭软件流水 + autotune 反超 opus K2**。这一优势在 gfx942 上不存在(那里 opus 后段也赢)。

3. **分段看,前后段方向相反**:opus 前段 **k14 恒赢 ~2.8×**(`opus/triton k14 ≈ 0.36`,结构性优势,见 §4);后段 **k56 长 dense 输给 Triton**(~1.4–1.5×),短序列翻盘。

4. **理论最优是"混血" opus-k14 + triton-k56**:锚点 35B-tp1 T=8192 ≈ **97 + 334 ≈ 431µs**,优于纯 opus(579)与纯 Triton(592)。这与 gfx942 上"opus 两段都赢、不需要混血"相反。

5. **BT=64 仍是最优默认**: BT=32 比 BT=64 慢 **+226%**(锚点 35B-tp1 T=8192: 1900 vs 583 µs)。opus 在 BT=32 拿不到 K2 的 `num_warps=8` 且 chunk 数翻倍,gfx950 dense 上退化尤其猛。BT=128 在 gfx950 可用(160KB LDS)。

---

## 1. Dense 端到端对比 (cu_seqlens=None, 单序列)

> 全量 28 case。`triton k14/k5/k6` 为 CUDA-event 分段, `opus 总计` 为 e2e。**← = opus 更快**。

| 模型 | T | H | triton k14 | triton k5 | triton k6 | triton 总计 | opus 总计 | opus/triton |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 35B-tp1 | 1024 | 32 | 110.9 | 84.1 | 26.0 | 221.4 | **86.9** | 0.393 **←** |
| 35B-tp1 | 2048 | 32 | 123.4 | 77.1 | 33.4 | 233.3 | **136.2** | 0.584 **←** |
| 35B-tp1 | 4096 | 32 | 152.9 | 101.6 | 56.1 | 311.1 | **253.1** | 0.813 **←** |
| 35B-tp1 | 8192 | 32 | 256.5 | 219.2 | 114.3 | 591.7 | **579.4** | 0.979 **←** |
| 35B-tp1 | 16384 | 32 | 448.3 | 437.6 | 224.7 | **1112.5** | 1143.0 | 1.027 |
| 35B-tp1 | 32768 | 32 | 850.9 | 871.7 | 428.6 | **2152.1** | 2257.8 | 1.049 |
| 35B-tp1 | 65536 | 32 | 1686.5 | 1708.9 | 849.5 | **4244.2** | 4641.2 | 1.094 |
| 35B-tp2 | 1024 | 16 | 106.2 | 89.6 | 20.2 | 216.2 | **72.0** | 0.333 **←** |
| 35B-tp2 | 2048 | 16 | 112.6 | 83.7 | 27.6 | 224.0 | **104.6** | 0.467 **←** |
| 35B-tp2 | 4096 | 16 | 124.1 | 79.8 | 37.9 | 242.1 | **174.6** | 0.721 **←** |
| 35B-tp2 | 8192 | 16 | 152.7 | 114.2 | 58.1 | **324.7** | 334.4 | 1.030 |
| 35B-tp2 | 16384 | 16 | 251.9 | 222.8 | 115.4 | **588.7** | 788.6 | 1.339 |
| 35B-tp2 | 32768 | 16 | 429.8 | 452.2 | 222.6 | **1103.6** | 1579.5 | 1.431 |
| 35B-tp2 | 65536 | 16 | 828.7 | 874.0 | 431.8 | **2134.1** | 3135.9 | 1.469 |
| FULL-tp1 | 1024 | 64 | 116.5 | 80.6 | 33.4 | 230.8 | **107.4** | 0.466 **←** |
| FULL-tp1 | 2048 | 64 | 153.0 | 101.7 | 55.8 | 310.1 | **202.0** | 0.651 **←** |
| FULL-tp1 | 4096 | 64 | 254.4 | 215.1 | 115.6 | 584.3 | **433.2** | 0.741 **←** |
| FULL-tp1 | 8192 | 64 | 462.6 | 434.6 | 223.5 | 1121.5 | **844.4** | 0.753 **←** |
| FULL-tp1 | 16384 | 64 | 896.1 | 849.9 | 430.2 | 2180.3 | **1662.7** | 0.763 **←** |
| FULL-tp1 | 32768 | 64 | 1809.9 | 1681.9 | 850.3 | 4343.4 | **3346.1** | 0.770 **←** |
| FULL-tp1 | 65536 | 64 | 3619.7 | 3449.9 | 1715.4 | 8782.4 | **6872.4** | 0.783 **←** |
| FULL-tp2 | 1024 | 32 | 108.8 | 82.9 | 25.7 | 217.5 | **85.0** | 0.391 **←** |
| FULL-tp2 | 2048 | 32 | 119.3 | 77.1 | 33.1 | 229.3 | **135.2** | 0.590 **←** |
| FULL-tp2 | 4096 | 32 | 147.7 | 101.1 | 55.3 | 304.5 | **255.1** | 0.838 **←** |
| FULL-tp2 | 8192 | 32 | 246.2 | 213.1 | 112.3 | **571.3** | 581.2 | 1.017 |
| FULL-tp2 | 16384 | 32 | 439.0 | 435.8 | 224.8 | **1098.1** | 1151.0 | 1.048 |
| FULL-tp2 | 32768 | 32 | 851.3 | 873.7 | 431.1 | **2157.4** | 2283.8 | 1.059 |
| FULL-tp2 | 65536 | 32 | 1710.8 | 1725.5 | 857.0 | **4291.1** | 4649.9 | 1.084 |

> **geomean opus/triton = 0.788, opus 胜 17/28。**
> **关键观察**:
> 1. **短序列 opus 通吃**(T≤2048 全部 opus 胜,launch 少 + k14 占比高);
> 2. **头多保 opus**:`FULL-tp1` (H=64) **7/7 全胜**(k14 占比随头数升高,后段 K5 反超不动总账);头少 `35B-tp2` (H=16) 长序列 Triton 大胜(后段主导,1.34–1.47×);
> 3. **随 T 增长 opus/triton 单调上升**:k56(Triton 强项)占比变大 → 长 dense 偏向 Triton。

### 锚点 (35B-tp1 T=8192, B=1 H=32 标准 config) — rocprofv3 逐段

| 段 | triton | opus | opus/triton |
|---|---:|---:|---:|
| 前段 k14 | 243.0 | **97.4** | **0.40** |
| 后段 k56 | 308.1 | 473.8 | **1.54** |
| **总计** | 551.1 | 571.2 | 1.04 |

> 一前一后正好相反 → **混血**(opus k14 + triton k56)= 97.4 + 308.1 ≈ **406µs**,比纯 opus(571)、纯 Triton(551)都快 ~26%。

---

## 2. Packed Varlen 端到端对比 (T_total=32768, N 条等长序列)

> **口径说明(重要)**:本 bench 的 opus 路径**未接入 `cu_seqlens`**(对打包张量按单条长序列跑 dense),Triton 路径用 cu-aware scan。因此本组实为"同一打包张量:Triton 拆链 scan vs opus dense scan"。在 gfx950 上 Triton 的 32x32x16 scan + 拆链并行双重得利,故 opus 多数落后。MI300X 报告同一脚本下 opus 反而赢,是因为 gfx942 Triton scan 弱。

| 模型 | fpl | N | triton k14 | triton k5 | triton k6 | triton 总计 | opus 总计 | opus/triton |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 35B-tp1 | 1024 | 32 | 802.9 | 501.8 | 430.2 | **1730.5** | 2264.6 | 1.309 |
| 35B-tp1 | 2048 | 16 | 812.4 | 491.8 | 432.2 | **1737.3** | 2264.1 | 1.303 |
| 35B-tp1 | 4096 | 8 | 810.8 | 489.1 | 431.8 | **1733.3** | 2261.0 | 1.304 |
| 35B-tp1 | 8192 | 4 | 806.6 | 493.2 | 429.9 | **1730.2** | 2258.0 | 1.305 |
| 35B-tp2 | 1024 | 32 | 416.1 | 263.3 | 221.6 | **899.8** | 1578.3 | 1.754 |
| 35B-tp2 | 2048 | 16 | 425.6 | 256.1 | 222.8 | **904.9** | 1579.5 | 1.746 |
| 35B-tp2 | 4096 | 8 | 420.5 | 254.4 | 219.7 | **891.7** | 1579.3 | 1.771 |
| 35B-tp2 | 8192 | 4 | 412.4 | 430.7 | 213.4 | **1056.3** | 1572.6 | 1.489 |
| FULL-tp1 | 1024 | 32 | 1584.9 | 1008.4 | 846.7 | 3441.0 | **3346.8** | 0.973 **←** |
| FULL-tp1 | 2048 | 16 | 1595.9 | 970.6 | 853.3 | **3417.0** | 3427.2 | 1.003 |
| FULL-tp1 | 4096 | 8 | 1597.7 | 946.3 | 854.4 | 3398.6 | **3353.0** | 0.987 **←** |
| FULL-tp1 | 8192 | 4 | 1601.3 | 936.4 | 847.8 | **3386.2** | 3424.8 | 1.011 |
| FULL-tp2 | 1024 | 32 | 811.9 | 512.4 | 433.1 | **1757.5** | 2284.8 | 1.300 |
| FULL-tp2 | 2048 | 16 | 814.1 | 494.2 | 433.4 | **1740.2** | 2284.6 | 1.313 |
| FULL-tp2 | 4096 | 8 | 813.3 | 485.6 | 434.2 | **1735.8** | 2266.1 | 1.306 |
| FULL-tp2 | 8192 | 4 | 812.7 | 491.6 | 429.4 | **1733.6** | 2270.3 | 1.310 |

> **geomean opus/triton = 1.30, opus 胜 2/16**(仅 FULL-tp1 H=64,k14 占比最高)。varlen 下 N(序列条数)对两后端 e2e 影响都很小(打包总 token 固定)。

---

## 3. BT=32 vs BT=64 Opus 对比

锚点: 35B-tp1 (H=32, K=V=128), T=8192

| BT | opus 总计 (µs) | vs BT=64 |
|---:|---:|---:|
| 32 | 1899.7 | **+226%** |
| **64** | **583.4** | baseline |

> BT=64 明确更优(gfx950 上 BT=32 退化幅度远大于 gfx942 的 +27%):chunk 数 ∝ 1/BT,且 BT=32 拿不到 K2 `num_warps=8`。**BT=128 在 gfx950 可用**(160KB LDS),gfx942 因 64KB 上限不可用。

---

## 4. 分段分析

### 前段 k14 (K1): Opus 的结构性优势(~2.8×)

- Opus K1 (`gdn_k1_neumann`) 将 cumsum + KKT + 三角求逆 + WY 因子重算融合为 **单 kernel**,A 矩阵不落 HBM。
- 三角求逆用 **Neumann 平方法** `(I+B)(I+B²)(I+B⁴)(I+B⁸)`,**6 次 MFMA**(原 Horner 15 次);cumsum 用单 warp `__shfl_up` 前缀和,去掉多个 block barrier。
- 矩阵 staging 进 ~25 KB LDS,**实测 VGPR 仅 36**(rocprofv3 编译产物)→ 高占用。
- Triton 分 4 个 kernel launch,中间 A 矩阵落 HBM,launch overhead 叠加(`recompute_w_u` 单 kernel 就 116 VGPR)。
- 结果:`opus/triton k14` 在锚点 = **0.40**(rocprof),全 dense 稳定 ~0.32–0.40。

### 后段 k56 (K2): gfx950 上 Triton 长序列反超

- Opus K2 走 **split scan**:`chunk_gated_delta_rule_fwd_h`(寄存器常驻 H + `buffer_load_b128`,VGPR=80,LDS=48KB)+ `gdn_k2_out`(VGPR=60,LDS=60KB),BV 自适应 16/32 填满 256 CU。
- Triton K5 (`chunk_delta_h`) 在 gfx950 上吃到 **32x32x16 MFMA** + `num_stages` 软件流水 + autotune → 长序列 scan 效率高,**dense 长序列反超 opus K2 ~1.4–1.5×**。
- 短序列 / 小 grid 下 Triton 多-launch 固定开销盖过流水红利,opus K2 反而更快。

### 与 MI300X 对比:为什么 gfx950 dense 长序列是 Triton 胜

| | MI300X (gfx942) | MI350 (gfx950) |
|---|---|---|
| 32x32x16 MFMA | 无 | **有** → Triton K5 scan 强 |
| Triton K5 长 dense | 弱(opus K2 全胜) | **反超 opus K2(1.4–1.5×)** |
| Opus K2 路径 | fused(split OCC=1 更慢) | **split scan(填满 256 CU)** |
| 最优策略 | opus 两段全用 | **混血:opus-k14 + triton-k56** |
| LDS / CU | 64 KB(BT≤64) | 160 KB(BT≤128) |

---

## 5. MI350 (gfx950) 资源与占用 (rocprofv3, BT=64)

| kernel | 角色 | VGPR/lane | LDS/WG |
|---|---|---:|---:|
| opus `gdn_k1_neumann` | k14(全融合) | **36** | ~25 KB |
| opus `chunk_gated_delta_rule_fwd_h` | k56 scan | 80 | 48 KB |
| opus `gdn_k2_out` | k56 out | 60 | ~60 KB |
| triton `recompute_w_u` | k14(最重) | 116 | 16 KB |
| triton `chunk_scaled_dot_kkt` | k14 | 44 | 24 KB |
| triton `parallel_scan_v2` | k5(×2/iter) | 112 | 40 KB |
| triton `chunk_fwd_kernel_o` | k6 | 60 | 48 KB |

### gfx950 特性

- **有 32x32x16 MFMA**:Triton K5 长序列 scan 更高效(dense 长序列反超 opus)。
- **split-K2 为默认**:ref_fwd_h scan + gdn_k2_out,BV 自适应填满 256 CU(gfx942 上 ref_fwd_h OCC=1 反而更慢,故 gfx942 用 fused)。
- **160KB LDS**:BT=128 可用。
- **统一 VGPR/AGPR (CDNA4)**:寄存器调度更灵活。

---

## 6. 配置默认

- **BT=64** 为两后端最优默认(chunk 数 ∝ 1/BT,后段主导);BT=128 gfx950 可用但 chunk 太大收益有限。
- **Opus K2**: gfx950 走 split-scan,`(BV, num_warps)` 由 launcher 按 BT 内部自适应(传入形参被覆盖)。
- **Opus K1**: 固定 4 warp,Neumann 平方法(15→6 MFMA),VGPR=36。
- **k2_mode=0 / k1_algo=1**: 默认路径。

---

## 7. 结论与优化建议

1. **MI350 上无单一最优后端,按 workload 选**:短序列 / 头多(FULL-tp1 H=64)用 opus;长 dense / 头少(35B-tp2 H=16)用 Triton。与 MI300X 的 "opus 全胜" 形成对照。
2. **理论最优 = opus-k14 + triton-k56 混血**(锚点 ~406µs,优于两个纯后端),是 gfx950 上压时延最值得做的方向。
3. **opus 前段 k14 结构性领先 ~2.8×**(VGPR=36、单 kernel 融合 + Neumann 平方法),已接近极限,空间有限。
4. **opus 后段 k56 在 gfx950 输给 Triton K5**:Triton 吃到 32x32x16 MFMA + 软件流水;opus K2 split-scan 内循环缺 per-chunk prefetch 流水,是主要短板。
5. **两后端默认 BT=64**;BT=32 在 gfx950 退化 +226%。

---

## 复现

```bash
cd /home/sijieli2/aiter
# 端到端 + 分段 (§1/§2/§3, BT=64)
python op_tests/triton_tests/bench_gdn_mi350.py
# 逐段 GPU 时间 (§4 锚点, rocprofv3)
rocprofv3 --kernel-trace -d /tmp/po -o t -- python prof/one.py opus   1 8192 32
rocprofv3 --kernel-trace -d /tmp/pt -o t -- python prof/one.py triton 1 8192 32
```

> 数据采集日期 2026-06-30,gfx950 / MI35x,aiter@`fgdn_prefill_neumann`。原始数据: `bench_gdn_mi350_raw.json`。
