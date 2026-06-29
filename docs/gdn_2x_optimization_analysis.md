# GDN Prefill 2x 性能优化分析

**日期**: 2026-06-27 (更新)
**测试配置**: B=1, T=8192, H=32, K=V=128, MI300X (gfx942)
**基线**: Triton 串行 GDN prefill = 3.643 ms
**目标**: ≤ 1.822 ms (2x 加速)

---

## 1. 当前性能全景

### 1.1 端到端性能对比

| 配置 | 总耗时 (ms) | K1 (ms) | K2 (ms) | vs Triton |
|---|---|---|---|---|
| Triton 全流程 | 3.643 | — | — | 1.00x |
| **HIP K1+K2 OCC=3 (BT=64 BV=64 nw=8 algo=1)** | **1.928** | **0.519** | **1.403** | **1.89x** |
| HIP K1+K2 OCC=2 (BT=64 BV=64 nw=8 algo=1) | 2.229 | 0.746 | 1.482 | 1.63x |
| HIP K1+K2 (BT=64 BV=64 nw=8 algo=0) | 2.428 | 0.922 | 1.506 | 1.50x |
| HIP K1+K2 (BT=64 BV=64 nw=4) | 2.680 | — | — | 1.36x |
| HIP K1+K2 (BT=64 BV=32 nw=8) | 2.885 | — | — | 1.26x |
| HIP K1+K2 (BT=64 BV=32 nw=4) | 4.547 | — | — | 0.80x |
| Pipeline 模式 (BT=64 nw=8) | 4.741 | 2.483 | 5.065 | 0.77x (失败) |
| HIP K1 + 拆分 K2 (scan+output) | 3.013 | 0.549 | 2.483 | 1.21x |
| **2x 目标** | **1.822** | — | — | **2.00x** |

**关键结论**: K1 OCC=3 优化（寄存器缓存 C_inv）将总耗时降至 1.928ms = **1.89x**。距 2x 目标差 **0.106 ms (5.5%)**。

### 1.2 K1/K2 分离计时（torch.profiler 实测）

```
HIP K1 Neumann OCC=3 (algo=1): 0.519 ms  (27% of total)
  4096 WGs, nw=4 (256 threads/WG), OCC_HINT=3
  寄存器缓存 C_inv，消除 s_C_bf16 LDS 缓冲区
  arch_vgpr=76, accum_vgpr=0, LDS=18,176 bytes
  (OCC=2 时: 0.746ms, LDS=26,112 → 提升 30%)

HIP K2 fused (BV=64, nw=8): 1.403 ms  (73% of total)
  64 WGs, nw=8 (512 threads/WG), OCC_HINT=1
  128 chunks × ~11.0 μs/chunk
  arch_vgpr=142, accum_vgpr=0, LDS=43,776 bytes
  VALU 主导：737 VALU vs 32 MFMA (静态指令计数)

Triton K1 各子步骤 (共 1.204 ms):
  cumsum:     0.020 ms  ( 2%)
  KKT:        0.194 ms (16%)
  trisol:     0.465 ms (39%)
  WY (w,u):   0.565 ms (47%)
```

### 1.3 K1 算法对比

| 算法 | K1 耗时 (ms) | 总耗时 (ms) | vs Triton |
|---|---|---|---|
| Neumann (algo=1) | 0.746 | 2.229 | 1.63x |
| Forward-sub (algo=0) | 0.922 | 2.428 | 1.50x |

Neumann 比 forward-sub **快 19%**。Neumann 用 MFMA Horner 求值计算 16×16 对角块（每块 15 次迭代，4 个 warp 并行计算 4 个块）+ Schur 补合并（3 层）。Forward-sub 每行有 O(BT) 的串行依赖。

### 1.4 为什么 K2-nw=8 优于 Wavefront-nw=4

| | K2 (nw=8) | WF fused (nw=4) |
|---|---|---|
| Warps/WG | 8 | 4 |
| OCC_HINT | 1 | 2 |
| WGs/CU | 1 | 2 |
| 有效 warps/CU | 8 | 8 |
| MFMA 吞吐/CU | 相同 | 相同 |
| 同步开销 | 无 | h_pass 读写 + atomic flags |
| 可用 VGPRs | 512 | 256 (OCC=2 限制) |
| 寄存器溢出风险 | 低 | 高 |

**结论**: 对当前工作负载（64 条独立链，128 个 chunk），串行 K2-nw=8 优于 wavefront-nw=4，因为：
1. 每 CU 的 MFMA 吞吐相同（都是 8 个 wave）
2. 零 wavefront 同步开销
3. 更多可用寄存器（OCC=1）
4. 更简单的控制流

**Wavefront 仅在以下情况有优势**: N_flat >> CU 数量时，或 S 足够小使流水线开销可被摊销。当前配置（64 链，304 CU），串行 scan + nw=8 更优。

---

## 2. HBM 流量分析

### 2.1 数据流

```
输入（读一次）:
  q:     B*T*H*K * 2B = 67.1 MB
  k:     67.1 MB
  v:     67.1 MB
  g:     B*T*H * 4B   = 1.05 MB
  beta:  1.05 MB
  输入合计: 203.4 MB

输出（写一次）:
  o:     67.1 MB

K1 → K2 中间结果（K1 写，K2 读）:
  w_bar:    67.1 MB × 2 = 134.2 MB
  u_bar:    67.1 MB × 2 = 134.2 MB
  g_cumsum: 1.05 MB × 2 = 2.1 MB
  中间结果合计: 270.5 MB

K2 内部（串行 scan，保持在寄存器中）:
  h 快照: 未物化（串行 K2 内联计算输出）
  v_new:  未物化（立即消费）

HBM 总流量: 203.4 + 67.1 + 270.5 = 541.0 MB
带宽极限 (5.3 TB/s): 0.102 ms
```

**kernel 严重受计算限制**（实际 1.928ms vs 带宽极限 0.102ms = 5.3% 带宽利用率）。

### 2.2 流量优化可能性

| 优化 | 节省 | 影响 |
|---|---|---|
| 将 K1 融合进 K2（消除中间结果） | 270.5 MB | -0.051 ms（仅带宽） |
| 但 K1 融合会串行化 K1 计算 | | +~1.0 ms（更慢） |

**K1-K2 融合是反效果的**: K1 有 4096 路并行（128 chunks × 32 heads），K2 仅 64 路。融合后 K1 被迫在 K2 的低并行度下运行，每个 WG 增加约 1.0ms 串行 K1 计算。

---

## 3. 文献调研

### 3.1 Tiled Flash Linear Attention (TFLA)

**论文**: arXiv:2503.14376 (NeurIPS 2025)
**代码**: NX-AI/mlstm_kernels

核心思路：两层 tiling 打破 chunk size 瓶颈。
- **第一层**（inter-chunk）: 标准 chunk 分解，但 **BT=128-256**
- **第二层**（intra-chunk）: 沿 seq 和 embedding 维度 tile BT×BT attention 矩阵，保持在 SRAM 中

**对 GDN 的适用性**: 论文确认兼容 DeltaNet（附录 A.2）。但我们的 BT=128 实验因寄存器压力（251 VGPRs）表现更差。

### 3.2 cuLA (基于 CUTLASS 的线性注意力)

**代码**: inclusionAI/cuLA (2026)
**结果**: 在 Hopper 上比 FLA Triton 平均加速 1.58x

使用 CUTLASS/CuTe DSL。仅支持 NVIDIA，但模块化 kernel 分解模式可借鉴。

### 3.3 MoonMath HIP Attention Kernel (2026)

在 MI300X 上比 AITER v3 快 1.08-1.26x。关键洞察：**存储位置很重要** — K 放 LDS，V 热数据在 L1，Q 在寄存器。

### 3.4 XCD 感知的 Tile Swizzling (SwizzlePerf)

MI300X 有 8 个 XCD，各有独立 L2 cache。将协作 tile 放在同一 XCD 可最大化 L2 复用。线性注意力的可预测访问模式非常适合静态 swizzling。

---

## 4. 拆分 K2 实验（失败）

### 4.1 方案

将融合 K2 kernel 拆为两个独立 kernel：
- **Scan kernel**: phase a (h 快照存储), b (retrieve GEMM w×h), b' (v_new=u-retrieve), d (h 更新 k_gated^T×v_new)。chunk 串行，head 并行。Grid: 64 WGs。
- **Output kernel**: phase c (跨 chunk GEMM q×h), e (chunk 内 QK^T+AV attention)。完全 chunk 并行。Grid: 8192 WGs。

**动机**: Scan kernel 不含 output phase，VGPR 应更少 → OCC=2（≤128 VGPRs）→ 更好的延迟隐藏。Output kernel 是完全并行的。

### 4.2 结果

| 指标 | 融合 K2 | 拆分 K2 | 比值 |
|--------|----------|----------|-------|
| K2 耗时 | 1.403 ms | 2.483 ms (scan+output) | 1.8x 更慢 |
| 端到端 | 1.928 ms | 3.013 ms | 1.6x 更慢 |
| 正确性 | — | 完全匹配 | ✓ |

### 4.3 根因分析

**1. Scan kernel VGPR 仍超过 OCC=2 阈值**

| Kernel | nw=8 VGPRs | OCC |
|--------|-----------|-----|
| 融合 K2 (BV=64) | 142 | 1 |
| Scan-only (BV=64) | 142 | 1 (需 ≤128) |
| Output-only (BV=64) | 68 | 2 ✓ |

Scan kernel 142 VGPRs 超过 OCC=2 的 128 阈值。尝试 `__attribute__((amdgpu_num_vgpr(128)))` 被编译器完全忽略。根本原因：phase b 和 d 各需要 h-state 寄存器 + MFMA 累加器 + 操作数缓冲区，总计必然 >128 VGPRs。

**2. 物化中间状态导致 HBM 流量激增**

| 缓冲区 | 融合 K2 | 拆分 K2 | 说明 |
|--------|----------|----------|-------|
| h_snap (fp32) | 未物化 | 256 MB (写+读) | [B,NT,H,K,V] fp32 |
| v_new (bf16) | 未物化 | 64 MB (写+读) | [B,T,H,V] bf16 |
| q, k 重复读 | 读一次 | 多读 67+67 MB | output kernel 重读 |
| **HBM 总计** | **~516 MB** | **~1,288 MB** | **2.5x 更多** |

融合 kernel 将 h-state 和 v_new 保持在寄存器中立即消费。拆分后强制物化到 HBM，增加 772 MB 流量。

### 4.4 BV=32 子实验（同样失败）

BV=32 使 scan kernel VGPRs 降至 118（满足 OCC=2 VGPR 限制），但 LDS 含 persistent q 时为 35,072 × 2 WGs = 70,144 > 65,536 字节限制。

**结果**: BT=64 BV=32 nw=8 = 2.885 ms（比 BV=64 的 1.928ms 慢 50%）。

---

## 5. K1-K2 流水线重叠实验（失败）

### 5.1 方案

使用两个 HIP stream 重叠 K1 和 K2 执行。K2 在 K1 完成前启动，通过 atomic flag 自旋等待每个 chunk 的 K1 完成信号。

### 5.2 结果

| 指标 | 串行 | 流水线 | 比值 |
|--------|-----------|----------|-------|
| K1 耗时 | 0.746 ms | 2.483 ms | 3.33x 更慢 |
| K2 耗时 | 1.482 ms | 5.065 ms | 3.42x 更慢 |
| 总计 | 2.229 ms | 4.741 ms | 2.13x 更慢 |
| 正确性 | — | max_diff=0.000000 | 完全匹配 |

### 5.3 根因：CU 争用

K2 的 64 个 WG 预占 64 个 CU。当线程 0 自旋等待 atomic flag 时，整个 WG 在 `__syncthreads()` 处阻塞——512 个线程全部空闲，却霸占 CU。

K1 只剩 304 - 64 = 240 个 CU，从 0.75ms 降速到 2.48ms。K2 也受损，因为其早期 chunk 必须等待已被减速的 K1。

**无法修复**: HIP 没有协作式抢占，自旋等待的 WG 无法释放 CU。K1-K2 流水线重叠对此问题配置根本不可行。

---

## 6. 硬件计数器深度分析

### 6.1 K2 融合 Kernel 计数器

```
配置: BT=64 BV=64 nw=8, Grid=(2,32)=64 WGs, 512 threads/WG

每 wave 每 chunk 指令分布:
  VALU:    737 条（含 ~120 NaN 规范化） ← 主导
  MFMA:     32 条
  LDS:     ~130 条
  VMEM_RD: ~24 条
  VMEM_WR: ~8 条

VALU 构成分析:
  ~360 VALU: MFMA tile 加载的 LDS 地址计算
    (5 个 GEMM × 2 bk × 12 loads × 3 VALU/load)
  ~120 VALU: NaN 规范化 (v_cmp_u_f32 + v_cndmask_b32 × 40 处)
  ~160 VALU: VMEM 地址计算
  ~85  VALU: 逐元素运算 (gate-scale, v_new, causal mask)
  ~12  VALU: 循环控制

资源使用:
  arch_vgpr=142, accum_vgpr=0, LDS=43,776 bytes
  CU 利用率: 64 WGs / 304 CUs = 21%
  Barrier 开销: ~14 barriers/chunk × 128 chunks × ~50 cycles = 42.7μs (2.2%)
```

### 6.2 VALU 主导性分析

每个 `load_mfma_tile` 调用需要计算: `addr = (row_base + (lane_id & 15)) * stride + col_base + ((lane_id >> 4) << 2)` — 每次 tile load 需 3 条 VALU。5 个 GEMM × 2 bk 子 tile × ~12 loads/GEMM = 120 loads × 3 = 360 VALU。

**关键洞察**: MFMA tile load 的地址计算才是主要开销，而非 MFMA 运算本身。这是 gfx942 架构限制——MFMA 要求在 VALU 中显式计算每次 LDS tile load 的地址。

### 6.3 K1 Neumann Kernel 计数器

```
配置: BT=64 nw=4, Grid=(128,32)=4096 WGs, 256 threads/WG

每 wave 指令分布:
  VALU:    2,404 条 ← 同样 VALU 主导
  MFMA:      115 条
  LDS:       281 条
  VMEM_RD:    13 条
  VMEM_WR:    64 条

寄存器使用:
  OCC=3 (当前): arch_vgpr=76, accum_vgpr=0, LDS=18,176 bytes
  OCC=2 (之前): arch_vgpr=72, accum_vgpr=0, LDS=26,112 bytes
```

### 6.4 accum_vgpr=132 阻止 K2 OCC=2（但无关紧要）

K2 的 accum_vgpr=132 超过 OCC=2 的 128/wave 限制。峰值 accum 活跃数：h1[2]+h2[2]+r_retrieve[2]+r_o_cross[2] = 32 tiles = 128 accum VGPRs，编译器额外加了 4 个（用于 accum_to_src 转换临时变量）。

**但 OCC=2 对 K2 无意义**: 64 WGs / 304 CUs = 21% 利用率，没有足够 WG 填充第二个 occupancy slot。

---

## 7. BT=128 可行性分析 (gfx942)（不可行）

### 7.1 K2 BT=128 LDS

```
含 persistent q（当前设计）:
  总计 = 86,016 bytes > 65,536 ← 溢出

不含 persistent q:
  总计 = 51,200 bytes ✓（可行）
  但需从 HBM 重读 q（增加 ~33 MB 流量）
```

### 7.2 K1 BT=128 LDS（致命溢出）

```
smem_A = BT × A_STRIDE × 4 = 128 × 129 × 4 = 66,048 bytes > 65,536 ← 溢出

A_STRIDE = BT+1 = 129（padding 避免 bank conflict）
去掉 padding: 128 × 128 × 4 = 65,536（恰好到极限）
但 stride=128 会导致灾难性 bank conflict
```

K1 BT=128 在 gfx942 上**不可能**。混合 BT（K1=64, K2=128）数学上不合法——C_inv 矩阵是 [BT, BT]，K2 BT=128 要求 K1 计算 128×128 的 C_inv。仅 gfx950 (128KB LDS) 可支持 BT=128。

---

## 8. 通往 2x 的优化路径

### 8.1 优化路线图

**当前差距分析（K1 OCC=3 后）:**
```
当前最优:  K1≈0.519ms + K2≈1.403ms = 1.928ms (1.89x)
目标 (2x): 1.822ms
差距:      0.106ms (5.5% 的缩减)
K2 占总耗时 73% → K2 优化是唯一杠杆
```

| # | 优化项 | 预估收益 | 难度 | 状态 |
|---|---|---|---|---|
| **P0** | **降低 K2 VALU 开销** | 0.1-0.3 ms | 高 | 主要瓶颈 |
| ~~P1~~ | ~~K1: OCC 提升至 3~~ | **0.227 ms** | — | **已完成** ✓ (0.746→0.519ms) |
| **P2** | **K2: 消除 h-state LDS 往返** | ~0.05-0.1 ms | 高 | 下一目标 |
| P3 | K1 XCD 感知 grid swizzling | 0.05-0.1 ms | 低 | 未探索 |
| ~~P4~~ | ~~BT=128~~ | — | — | 不可行（K1 LDS 溢出） |
| ~~P5~~ | ~~K1-K2 流水线重叠~~ | — | — | 失败（CU 争用） |
| ~~P6~~ | ~~拆分 K2~~ | — | — | 失败（HBM 流量） |
| ~~P7~~ | ~~激进编译器选项~~ | — | — | 失败（寄存器压力） |
| ~~P8~~ | ~~fast_f32_to_bf16~~ | — | — | 失败（编译器忽略） |
| ~~P9~~ | ~~BV=32~~ | — | — | 失败（慢 50%） |
| ~~P10~~ | ~~K2 OCC=2 via waves-per-eu 强制~~ | — | — | 失败（编译器忽略所有属性） |

### 8.2 P0: 降低 K2 VALU 开销（主要瓶颈）

K2 是 VALU 主导的：每 chunk 每 wave 737 VALU vs 32 MFMA。VALU 指令主要来自 MFMA tile load 的 LDS 地址计算（49%）和 VMEM 地址计算（22%）。

**为什么困难**: 每个 `load_mfma_tile` 需要显式 VALU 地址算术。5 个 GEMM × 2 bk × ~12 loads = 120 次 tile load，每次 3 条 VALU。这是 gfx942 显式 LDS 寻址的架构限制。

**可能方案**:
1. **寄存器缓存地址**: 预计算 lane 相关偏移量，跨 load 复用。编译器应该已经做了，需 ISA dump 验证。
2. **更宽 MFMA (32×32×8)**: 更少更大的 tile → 更少地址计算。但分析显示加载次数相同，无净收益（§11.7）。
3. **减少 GEMM 数量**: 当前每 chunk 5 个 GEMM。若能合并 retrieve 和 cross 为单个大 GEMM 则可减少。
4. **Inline ASM**: 手写 LDS 加载序列，优化地址生成。

### 8.3 P1: K1 OCC=3 — 已完成 ✓

**实现**: 寄存器缓存 C_inv tile，消除 LDS 中的 `s_C_bf16`。

每个 warp 在寄存器中计算并保留自己的 16×16 C_inv tile，而非存储在 LDS（8,704 bytes）中。关键洞察：每个 warp 只需自己的对角块用于 Neumann 迭代，Schur 补合并只需通过 LDS 临时区（1,024 bytes）交换块对角元素。

**LDS 减少**: 25,856 → 18,176 bytes (−30%)，低于 OCC=3 阈值 21,845。
**寄存器影响**: arch_vgpr 72 → 76（仍低于 OCC=3 限制 85）。
**性能**: K1 0.746 → 0.519 ms (−30%)，总计 2.229 → 1.928 ms。
**正确性**: max_diff=0.000000，完全匹配参考实现。

修改的文件:
- `gdn_k1_bt64_neumann_kernel_template.hpp`: `__launch_bounds__(256, 3)` + 寄存器缓存 C_inv
- `gdn_defs.h`: `smem_size_bytes()` 对 BT≥64 排除 c_bf16_bytes

### 8.4 P2: K2 h-state LDS 往返消除（下一目标）

K2 的 phase b（retrieve）中，h-state 寄存器被溢出到 LDS 以构造 MFMA 源操作数，然后再读回。包括:
- 8 × ds_write_b128 将 h-state tile 溢出到 LDS
- Barrier
- 8 × ds_read_b128 按 MFMA 兼容布局重新加载
- ~152 条 VALU 用于地址计算

**可能方案**: 使用 `ds_permute_b32` / `ds_bpermute_b32` 进行跨 lane 寄存器 shuffle，直接构造 MFMA 源操作数，避免 LDS 写+读往返。

**预估收益**: ~0.05-0.1 ms（K2 耗时的 3-5%），可弥合约一半的 2x 差距。

### 8.5 失败的优化尝试

#### 8.5.1 激进编译器选项

尝试 `-O3 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false`:
- K1 **退步** 从 ~500μs 到 622μs
- 原因：激进内联增加寄存器压力，可能将 OCC 从 3 降回 2
- **已回滚** 至 `-ffast-math -std=c++20`

#### 8.5.2 fast_f32_to_bf16（绕过 NaN 规范化）

编译器对每次 fp32→bf16 转换插入 `v_cmp_u_f32 + v_cndmask_b32`（NaN 规范化），即使已设 `-ffast-math`。约 40 处 = ~120 条额外 VALU。

尝试通过 `__builtin_bit_cast` 整数算术绕过（round-to-nearest-even，不检查 NaN）。**结果**: 编译器生成完全相同的 ISA——bitcast 方式被识别并经同一转换路径降低。通过 fatbin MD5 哈希对比验证（完全相同）。

**结论**: 消除 NaN 规范化需要 inline assembly，源码级别无法实现。

#### 8.5.3 BV=32（更多 WG，更小 tile）

BV=32 使 WG 数翻倍（128 vs 64）但总工作量等比增加:
- BV=32 nw=8: 2.885 ms (1.26x) — 比 BV=64 **慢 50%**
- BV=32 nw=4: 4.547 ms (0.80x) — 更差

根因：更小 BV 意味着每个 GEMM 更多 MFMA tile 但总 FLOPs 相同，plus 启动开销增加和寄存器复用降低。

---

## 9. 可达性能预估

### 9.1 当前状态（K1 OCC=3 后）

```
K1 Neumann OCC=3 (algo=1): 0.519 ms  (27% of total, 4096 WGs, OCC=3)
K2 fused (nw=8):           1.403 ms  (73% of total, 64 WGs, OCC=1)
总计:                       1.928 ms  (1.89x vs Triton 3.643ms)
距 2x 差距:                  0.106 ms  (5.5%)
```

K1 已充分优化（OCC=3，降低 30%）。
K2 占总耗时 73%，VALU 主导——唯一剩余杠杆。

### 9.2 可达路径（K2 小幅 VALU 优化）

```
K1 OCC=3:                          ~0.519 ms
K2 消除 h-state LDS 往返 (-7%):     ~1.305 ms
总计:                                1.824 ms (2.00x) ← 达成 2x
```

需在 K2 中每 chunk 每 wave 减少约 100 条 VALU（从 ~737 降至 ~637）。

### 9.3 激进路径（Inline ASM）

```
K1 OCC=3:                          ~0.519 ms
K2 消除 NaN 规范化 (-16%):          ~1.179 ms
总计:                                1.698 ms (2.15x)
```

需要 inline assembly 绕过编译器 NaN 规范化（节省约 120 条 VALU）。难度极高，维护负担重。

### 9.4 已排除的路径

| 路径 | 原因 | 章节 |
|------|------|------|
| 拆分 K2 | 2.5x HBM 流量, 3.013ms | §4 |
| BV=32 | 比 BV=64 慢 50% | §8.5.3 |
| K1-K2 流水线重叠 | CU 争用致两 kernel 慢 3x | §5 |
| gfx942 上 BT=128 | K1 LDS 溢出 (66,048 > 65,536) | §7 |
| Wavefront nw=4 | 无优势 vs 串行 nw=8 | §1.4 |
| K2 OCC=2 | VGPRs=167, 编译器忽略所有 OCC/VGPR 属性 | §6.4, §11.10 |
| 激进编译器选项 | 寄存器压力增加, K1 退步 | §8.5.1 |
| fast_f32_to_bf16 | 编译器生成完全相同 ISA | §8.5.2 |
| 32×32×8 MFMA | 加载数相同, FLOPs 相同 | §11.7 |

---

## 10. 推荐下一步

### 当前状态: 1.928ms = 1.89x（距 2x 差 0.106ms / 5.5%）

### 高优先级

1. **K2: 消除 h-state LDS 往返** — 用 `ds_permute_b32` / `ds_bpermute_b32` 从 h-state 寄存器直接构造 MFMA 源操作数，避免当前的 LDS write+barrier+read 模式。预估节省 ~0.05-0.1ms。详见 §8.4。

2. **K2: Inline ASM 绕过 NaN 规范化** — 编译器为每次 fp32→bf16 转换插入 `v_cmp_u_f32 + v_cndmask_b32`（约 40 处, ~120 VALU）。源码级别无法消除（§8.5.2）。Inline `v_cvt_pk_bf16_f32` 或手动位操作可节省 ~0.1ms。

### 中优先级

3. **K2: 更深的预取** — 当前预取下一个 chunk。仅 2 waves/SIMD 下 VMEM 延迟 (~500 cycles) 隐藏不足。需额外寄存器。

4. **K1: XCD 感知 grid swizzling** — 4096 WGs 分布在 8 个 XCD 上。静态 swizzling 可提升 L2 复用。

### 已完成

- ~~K1: OCC=3~~ — **已完成** ✓ 0.746→0.519ms (§8.3)

### 已排除

- ~~gfx942 上 BT=128~~ — K1 LDS 溢出 (§7)
- ~~K1-K2 流水线重叠~~ — CU 争用 (§5)
- ~~拆分 K2~~ — HBM 流量暴增 (§4)
- ~~K2 OCC=2~~ — accum_vgpr=132 > 128, 仅 64 WGs (§6.4)
- ~~BV=32~~ — 慢 50% (§8.5.3)
- ~~Wavefront nw=4~~ — 无优势 (§1.4)
- ~~激进编译器选项~~ — 寄存器压力退步 (§8.5.1)
- ~~fast_f32_to_bf16~~ — 编译器生成相同 ISA (§8.5.2)
- ~~32×32×8 MFMA~~ — 加载数/FLOPs 相同 (§11.7)
- ~~K2 OCC=2 via 编译器属性~~ — 所有属性被忽略 (§11.10)

---

## 11. 实验附录

### 11.1 配置扫描结果（20 warmup + 50 runs 修正后）

```
K1 OCC=3 + BV=64 nw=8:  1.928 ms  (1.89x) ← 当前最优
BT=64 BV=64 nw=8:       2.229 ms  (1.63x) ← OCC=3 前
BT=64 BV=64 nw=4:       2.680 ms  (1.36x)
BT=64 BV=32 nw=8:       2.885 ms  (1.26x)
BT=64 BV=32 nw=4:       4.547 ms  (0.80x)
拆分 K2 (nw=8):          3.013 ms  (1.21x) ← 拆分失败
Triton 全流程:            3.643 ms  (1.00x)

注: 早期测量使用 5 warmup + 20 runs，结果偏高
（如 2.051ms 而非真实 1.928ms）。以上数值均使用修正后方法。
```

### 11.2 Wavefront 融合模式各 S 值性能

```
S=128: scan=1.153ms  fused=1.851ms
S= 64: scan=1.168ms  fused=1.884ms
S= 32: scan=1.305ms  fused=2.033ms
S= 16: scan=1.348ms  fused=2.073ms
S=  8: scan=1.479ms  fused=2.161ms
```

### 11.3 K1 子步骤分解（Triton, BT=64）

```
cumsum:     0.020 ms  ( 2%)
KKT:        0.194 ms (16%)
trisol:     0.465 ms (39%)
WY (w,u):   0.565 ms (47%)
总计:       1.204 ms
```

### 11.4 HBM 流量汇总

```
必要 I/O:       270.5 MB  (0.051 ms @ 5.3 TB/s)
K1→K2 中间结果: 270.5 MB  (0.051 ms @ 5.3 TB/s)
总计:           541.0 MB  (0.102 ms @ 5.3 TB/s)
实际耗时:       1.928 ms  → 计算受限 (5.3% 带宽利用率)
```

### 11.5 寄存器使用（实测）

```
gfx942 occupancy 阈值（每 wave, arch_vgpr 和 accum_vgpr 独立计算）:
  OCC=1: ≤512 arch_vgpr, ≤512 accum_vgpr, ≤65536 LDS
  OCC=2: ≤128 arch_vgpr, ≤128 accum_vgpr, ≤32768 LDS/WG
  OCC=3: ≤85  arch_vgpr, ≤85  accum_vgpr, ≤21845 LDS/WG

K1 Neumann kernel:
  OCC=3 (当前): arch_vgpr=76, accum_vgpr=0, LDS=18,176 → OCC=3 ✓
  OCC=2 (之前): arch_vgpr=72, accum_vgpr=0, LDS=25,856 → OCC=2

融合 K2 kernel (phases a-e):
  BT=64 BV=64 nw=8: 142 VGPRs, 0 AGPRs, 0 spills → OCC=1
  BT=64 BV=32 nw=8: 118 VGPRs, 0 spills → OCC=2 (VGPR 可以但 LDS 超限)
  BT=64 BV=64 nw=4: 256 VGPRs, 80 spills → OCC=1 (溢出严重)

拆分 scan kernel (phases a/b/b'/d):
  BT=64 BV=64 nw=8: 142 VGPRs, 0 spills → OCC=1 (需 ≤128 才能 OCC=2, 失败)

融合 K2 OCC=2 强制 (SERIALIZE_BC, 独立函数):
  __launch_bounds__(512,2): 167 VGPRs, 0 spills (忽略)
  + amdgpu_waves_per_eu(4,4): 167 VGPRs, 0 spills (忽略)
  + amdgpu_num_vgpr(128): 167 VGPRs, 0 spills (忽略, SGPRs 有 33 spills)

拆分 output kernel (phases c/e):
  BT=64 BV=64 nw=8:  68 VGPRs, 0 spills → OCC=2 ✓

注: ISA 通过 objcopy --dump-section + llvm-objdump 提取。
```

### 11.6 拆分 K2 实验详情

```
拆分 K2 分解 (BT=64 BV=64 nw=8, 重新测量):
  K1 (HIP OCC=3): 0.549 ms
  K2-scan kernel:  1.208 ms  (64 WGs, 128 chunks 串行)
  K2-output kernel:1.275 ms  (8192 WGs, chunk 并行)
  K2 合计:         2.483 ms
  总计:            3.013 ms  (1.21x vs Triton, 比融合慢 57%)

HBM 流量对比:
  融合 K2:  ~516 MB
  拆分 K2:  ~1,288 MB (2.5x 更多)
    h_snap 物化: 256 MB (fp32 [B,NT,H,K,V] 写+读)
    v_new 物化:   64 MB (bf16 [B,T,H,V] 写+读)
    q,k,g 重复读: 135 MB (output kernel 重读)

核心教训: 当拆分需要物化中间状态（h_snap, v_new）
而融合 kernel 可以保持在寄存器中时, 拆分必然更慢。
```

### 11.7 32×32×8 MFMA 分析

```
对比: mfma_f32_16x16x16_bf16 vs mfma_f32_32x32x8_bf16

16×16×16:
  - 输入: 4×bf16 A, 4×bf16 B → 4×fp32 C (per lane)
  - FLOPs: 16×16×16×2 = 8,192 / 指令
  - 吞吐: 每 8 cycles 一条 (gfx942)
  - 64×64 GEMM (K=128): 4×4×8 = 128 条 MFMA
  - 源加载: 4×8 A-tiles + 4×8 B-tiles = 64 次 tile load

32×32×8:
  - 输入: 4×bf16 A, 4×bf16 B → 16×fp32 C (per lane)
  - FLOPs: 32×32×8×2 = 16,384 / 指令 (2x)
  - 吞吐: 每 16 cycles 一条 (gfx942) — 2x 更慢
  - 64×64 GEMM (K=128): 2×2×16 = 64 条 MFMA (减半)
  - 源加载: 2×16 A-tiles + 2×16 B-tiles = 64 次 tile load (相同)

结论: 加载次数相同 (64/GEMM), 总 FLOPs 相同。
32×32×8 MFMA 指令更少但每条 VALU 加载开销不变。
净收益: 约为零。不再追求。
```

### 11.8 基准测试方法修正

```
早期测量 warmup 不足:
  5 warmup + 20 runs → 2.051 ms（偏高, GPU 未充分预热）
  20 warmup + 50 runs → 1.928 ms（稳定, 正确）

~6% 偏差原因:
  1. 初始迭代时 GPU 频率爬升
  2. 反复 kernel launch 的 cache 预热
  3. 早期调用的 HIP 运行时初始化开销

建议: 始终使用 ≥20 次 warmup 迭代 + ≥50 次测量。
最终取测量轮的中位数。
```

### 11.9 NaN 规范化 ISA 分析

```
编译器为每次 fp32→bf16 转换插入的 NaN 规范化:
  v_cmp_u_f32 vcc, v_src, v_src      ; 检查是否 NaN (无序比较)
  v_cndmask_b32 v_dst, v_result, 0x7FC0, vcc  ; 替换为规范 NaN

K2 中约 40 处转换 × 3 条指令 = ~120 额外 VALU/chunk。
占 K2 VALU 总数的约 16% (737 条静态指令)。

尝试消除方式:
  1. -ffast-math: 已设置, 对 NaN 规范化无效
  2. __builtin_bit_cast 整数舍入: 编译器识别该模式
     并生成完全相同 ISA (fatbin MD5 对比验证)
  3. 源码级 fast_f32_to_bf16(): 无 ISA 变化

唯一可行方案: inline assembly (v_cvt_pk_bf16_f32 或
手动 v_lshrrev_b32 + v_add_u32 位操作)
```

### 11.10 K2 OCC=2 强制实验（失败）

```
目标: 通过编译器属性强制 K2 VGPR ≤128, 开启 OCC=2
前提: SERIALIZE_BC 路径下 LDS=26,368 bytes (≤32,768 OCC=2 限制)
      LDS 不是瓶颈, VGPR 是唯一限制

实现: 独立 gdn_k2_kernel_occ2() 函数 (非模板)
      使用 K2_OCC2_Traits = gdn_k2_traits<64,128,128,64,8,2>
      SERIALIZE_BC=true: 串行化 retrieve+cross GEMM, 复用 LDS sub buffer

尝试的编译器属性:
  方法 1: __launch_bounds__(512, 2)
    结果: VGPRs=167, spills=0 (编译器忽略 occupancy hint)

  方法 2: + __attribute__((amdgpu_waves_per_eu(4, 4)))
    结果: VGPRs=167, spills=0 (编译器忽略)

  方法 3: + __attribute__((amdgpu_num_vgpr(128)))
    结果: VGPRs=167, spills=0 (编译器忽略)
    注: SGPRs 有 33 spills, 但 VGPRs 完全不受影响

对比 NVIDIA: nvcc 的 __launch_bounds__(maxThreads, minBlocks)
  会积极溢出寄存器来满足 occupancy 目标。
  ROCm hipcc 则视所有这些属性为可选提示。

正确性验证: max_diff=0.000750 (SERIALIZE_BC 路径)
性能 (无 VGPR 减少时): 2.075 ms vs 1.901 ms baseline = 9.2% 更慢
  原因: 串行化 b/c GEMM 增加延迟, 但无 OCC 提升抵消

结论: 在 ROCm/gfx942 上, 编译器无法/不愿将 K2 kernel 的
VGPRs 从 167 强制降至 128。K2 的 5 GEMM 结构 (h-state 寄存器
+ MFMA 累加器 + 操作数缓冲区) 的最低 VGPR 需求超过 128。
除非手写 inline assembly 管理寄存器分配, 否则 K2 OCC=2 不可行。
```

---

## 12. 参考文献

1. **Tiled Flash Linear Attention (TFLA)**: arXiv:2503.14376, NeurIPS 2025. 两层 tiling 加速线性注意力。代码: NX-AI/mlstm_kernels
2. **cuLA**: inclusionAI/cuLA (2026). 基于 CUTLASS 的 FLA kernel。Hopper 上 1.58x vs Triton。
3. **GDN-2**: arXiv:2605.22791 (2026). Channel-wise gating, 扩展 GDN 和 KDA。
4. **Flash Linear Attention (FLA)**: fla-org/flash-linear-attention. Triton 参考实现。
5. **SwizzlePerf**: arXiv:2508.20258. MI300X XCD 感知 tile 调度。
6. **MoonMath HIP Attention**: AMD MI300X 存储位置优化。
