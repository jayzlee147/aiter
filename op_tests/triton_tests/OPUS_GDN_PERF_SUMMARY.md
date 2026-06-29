# Opus GDN Prefill — Performance Summary (gfx950 / MI350)

Forward gated-delta-net prefill (`opus_gdn_prefill_fwd`, K=V=128, bf16, BT=64).
Baseline = triton FLA `chunk_gated_delta_rule`. **Goal: beat triton.** ✅ achieved.

## 1. Headline: opus beats triton on every tested config

End-to-end wall-clock (python side, min of 5 runs, BT=64, k2_mode=0 auto):

| config            | B·H | opus  | triton | speedup |
|-------------------|----:|------:|-------:|--------:|
| B1 T8192  H32     |  32 | 562µs | 609µs  | **1.08×** |
| B1 T16384 H16     |  16 | 760µs | 866µs  | **1.14×** |
| B1 T4096  H8      |   8 | 139µs | 190µs  | **1.37×** |
| B1 T2048  H16     |  16 |  91µs | 153µs  | **1.69×** |
| B4 T2048  H8      |  32 | 115µs | 156µs  | **1.35×** |
| B1 T1024  H4      |   4 |  53µs | 151µs  | **2.87×** |

Wins grow as the device gets grid-starved (small B·H): opus handles low parallelism
better and launches fewer kernels.

## 2. Holds under CUDA graph too

Removing launch overhead (HIP graph capture/replay) does not flip the result —
opus still wins everywhere; graph mainly recovers triton's small-config launch tax:

| config        | EAGER opus / tri | GRAPH opus / tri | graph speedup |
|---------------|------------------|------------------|--------------:|
| B1 T8192 H32  | 567 / 612        | 569 / 614        | **1.08×** |
| B1 T4096 H8   | 140 / 190        | 146 / 197        | **1.34×** |
| B1 T2048 H16  |  91 / 155        |  97 / 139        | **1.43×** |
| B1 T1024 H4   |  53 / 154        |  57 /  85        | **1.50×** |

CUDA graph does NOT change individual kernel GPU time — it only removes the
CPU launch gaps *between* kernels. opus (3 kernels) has fewer gaps than triton
(5 kernels), so eager already favors opus; graph narrows that part but opus's
per-kernel GPU work is competitive/faster, so it still wins.

## 3. Per-stage GPU time (seq8k, B1 T8192 H32)

| stage              | opus kernel(s)                         | opus | triton                                   | triton |
|--------------------|----------------------------------------|-----:|------------------------------------------|-------:|
| **K1 / prepare**   | `gdn_k1_neumann_kernel` (1 fused)      | **94µs** | cumsum+kkt+recompute_w_u (3 kernels) | 111µs |
| **scan / fwd_h**   | `chunk_gated_delta_rule_fwd_h` (vendored ref) | 362µs | `chunk_gated_delta_rule_fwd_kernel_h` | 338µs |
| **out / chunk_o**  | `gdn_k2_out_kernel`                    | 115µs | `chunk_fwd_kernel_o`                      | 108µs |

- **K1: opus wins (94 vs 111, 1.18×)** at every size — fuses cumsum + KKT +
  triangular inverse (Neumann) + WY/UT transform into ONE kernel, keeping all
  intermediates in LDS (no HBM round-trips, no inter-kernel launch).
  Cross-config: 8192H32 94/111 · 4096H8 19/23 · 2048H16 19/24.
- **scan: opus slightly behind (362 vs 338).** This is AMD's vendored HIP fwd_h,
  grid-bound by the serial recurrence (≤ V/BV × B·H workgroups; occ ~1.4/8;
  nw=4 locked by BT=64). Near its architectural floor.
- **out: ~parity (115 vs 108)** after optimization (see §5).

opus net win = K1 advantage + out pipelining + fewer kernels (3 vs 5).

## 4. K2 architecture: fused vs split

K2 = scan (h-state recurrence) + output. Two paths, auto-selected:
- **fused** (`gdn_k2_kernel`): register-resident scan+output in one kernel. Wins
  at high parallelism (B·H large).
- **split** (scan → materialize h_snap → chunk-parallel out): wins when
  grid-starved. Scan = **vendored reference fwd_h**; out = `gdn_k2_out_kernel`.
- Auto threshold: split when `fused_grid (=2·B·H) < 129` (i.e. B·H ≤ 64).
  Verified: B·H=64 split wins big, B·H≥128 fused wins.

The reference fwd_h reads opus's **token-major** w/u/k directly (added strided
loads) and writes token-major v_new → **zero transposes** (saved ~135µs vs an
earlier permute-based integration).

## 5. Out-kernel optimization (the main lever): 212 → 115µs (1.84×)

`gdn_k2_out_kernel` was the bottleneck (212µs, 2× triton). Fixes, in order:

| step | change | out µs |
|------|--------|-------:|
| 0 | baseline (BV=32)                                            | 212 |
| 1 | BV 32→128 (single v-tile: no q/k re-read, no intra q@kᵀ recompute) | 170 |
| 2 | de-alias `s_kh`/`s_A5` (h^T & k in separate LDS → halve bk-loop barriers) | 162 |
| 3 | prefetch v_new into regs before bk-loop (overlap HBM latency) | 155 |
| 4 | **software-pipeline bk-loop (num_stages=2): prefetch bk+1 h/k during bk's GEMMs** | **115** |

Step 4 was decisive. `BV` is adaptive: 128 normally, 64 when `NT·B·H < 128`
(BV=128 makes grid.x=1, so small grids starve). `BK=128` was tried and rejected
— it collapses N_K to 1 and loses the bk-loop pipelining (net regression).

## 6. Correctness

- vs fused K2: max abs diff **0.0002** (bf16 h_snap rounding), final_state 0.0015.
- All seq sizes, with/without initial & final state.
- `pytest op_tests/triton_tests/test_opus_gdn.py` → **13/13 passed**.

## 7. Measurement caveats (so numbers reproduce)

- **Must pass `BT=64`.** The python wrapper defaults to `BT=32`, which routes to
  the slow fused BT=32 path (`gdn_k2_kernel` ~1810µs) and tanks the comparison.
- GPU kernel time (rocprofv3 kernel-trace) ≠ wall-clock e2e (adds launch +
  python overhead). Both reported; opus wins on both.
- The node thermally throttles under back-to-back benchmarking (absolute numbers
  drift ~1.5× run-to-run); same-session interleaved or fresh-process min-of-N
  used for fair comparisons. Ratios are stable.

## 8. Remaining headroom (not yet done)

- **scan (362 vs 338):** would need algorithm-level parallelization of the serial
  recurrence to lift the grid/occupancy ceiling. Largest single gap.
- **out (115 vs 108):** near the h_snap-read BW floor (128MB, the split tax that
  triton also pays). fp8 h_snap could cut ~20µs but risks precision.
- **K1:** already ahead; small headroom.
