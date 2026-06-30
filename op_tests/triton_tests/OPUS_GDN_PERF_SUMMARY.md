# Opus GDN Prefill — Performance Summary & Closeout

Forward gated-delta-net prefill (`opus_gdn_prefill_fwd`, K=V=128, bf16, BT=64).
Baseline = triton FLA `chunk_gated_delta_rule`. Branch: `fgdn_prefill_neumann`.

Two GPUs, two different optimal paths (LDS & register-file architecture differ):
- **gfx950 / MI350** — 256 CU, 160KB LDS/CU, unified VGPR/AGPR, 32×32×16 bf16 MFMA.
- **gfx942 / MI300X** — ~304-320 CU, 64KB LDS/CU, separate VGPR/AGPR, 16×16×16 only.

---

## 1. Headline results

| | gfx950 (MI350) | gfx942 (MI300X) |
|---|---|---|
| **vs triton** | **1.08–2.87×** (eager & CUDA-graph) | **1.83×** |
| K1 / prep | ~95µs (beats triton 111) | 500µs (beats triton prep) |
| K2 path | **split** (ref_fwd_h scan + out), ~475µs | **fused** (gdn_k2_kernel), 1388µs |
| K2 alternatives | fused ~1400, fused-wf 1075 | split 1453, fused-wf 1688 |

The K2 winner flips by arch — see §3.

---

## 2. K1 — optimized, wins on both arches

K1 fuses cumsum + KKT + triangular-inverse + WY into one kernel (vs triton's 3).

| optimization | effect | commit |
|---|---|---|
| Arch-gated OCC | gfx942: register-cached C_inv → OCC 2→3 (−30%, LDS-bound there). gfx950: keep LDS C_inv OCC=2 (160KB not the limiter; OCC=3 a no-op) | 888e3974 |
| Neumann squaring | triangular inverse `(I+B)(I+B²)(I+B⁴)(I+B⁸)` = Σⁿ₌₀¹⁵Bⁿ in **6 MFMAs vs 15** Horner | 4f3ee267 |
| shfl cumsum | g prefix-sum in one warp via `__shfl_up` (BT==warpSize) → drops ~6 block barriers | ff340626 |

gfx942 K1: 746 (OCC=2) → 519 (OCC=3) → 508 (cumsum) → **500µs** (squaring).
gfx950 K1: ~95µs unchanged by squaring/cumsum (K1 is **not** compute/barrier-bound there).

---

## 3. K2 — the arch split, and the gfx942 wall

K2 = h-state scan (serial recurrence over chunks) + output. Approaches tried:

| approach | what it does | gfx950 | gfx942 |
|---|---|---|---|
| **fused** `gdn_k2_kernel` | register-resident scan+output, 1 kernel | ~1400 | **1388 (best)** |
| **split** (ref_fwd_h scan + out kernel) | materialize h_snap, chunk-parallel output | **~475 (best)** | 1453 |
| **wavefront fused** `gdn_wf_h_kernel` | super-chunk grid + atomic h handoff, fused output | 1075 | 1688 |

### Why split wins on gfx950 but loses on gfx942
ref_fwd_h scan uses 264 VGPR+AGPR. gfx950's **unified** register file (CDNA4) tolerates it → OCC OK → 360µs. gfx942's **separate** VGPR/AGPR (CDNA3) caps it at OCC=1 → 785µs. The out kernel needs ≥38KB LDS → 1 WG/CU on gfx942's 64KB (vs 2-4 on gfx950's 160KB) → 668µs.

### Why 2× is unreachable on gfx942 (the three-axis wall)
fused K2 is **grid-starved** (grid = `ceil(V/BV)·B·H` = 64 WG on ~320 CU = 20% utilization; achieved OCC 19.5%, MFMA 11.9%, HBM 7% — pure latency-bound). The fix is "more WG to fill idle CUs", but every parallelism axis costs more than it saves:

1. **V-axis** (smaller BV → more V-tiles): q/k are V-independent but re-read per V-tile → redundant loads + retrieve/h-update recompute. *Measured BV=32 was 2× slower.*
2. **chunk-axis via ref_fwd_h** (split): 264-register serial scan → OCC=1 on gfx942.
3. **chunk-axis via wavefront** (super-chunk pipeline): atomic-handoff overhead grows with N_super (fastest config is degenerate serial), fused output path heavier than fused K2's, AGPR 132 → OCC 4→3. *Measured 1688 > 1388.*

Barrier audit of fused K2: 12 `__syncthreads`, **all true cross-warp GEMM-staging deps** (not implementation artifacts like K1's cumsum); one already removed in e989bb0c. Removing the lone candidate (g_cumsum load) is ~1-3%, not the needed 12%.

**Conclusion: 1.83× is the practical ceiling for this algorithm family on gfx942.** The only remaining theoretical path is a true associative parallel-scan (Blelloch up/down-sweep over chunks) that is *leaner* than the wavefront — high risk, since wavefront already showed per-chunk-boundary overhead dominates.

---

## 4. Final configuration (defaults)

- **K1**: arch-gated neumann (gfx942 OCC=3 register C_inv / gfx950 OCC=2 LDS), squaring inverse, shfl cumsum. Runtime `gdn_is_gfx950()` picks the dynamic-smem size.
- **K2 auto** (`k2_mode=0`): split when grid-starved (`fused_grid < 129`, i.e. B·H ≤ 64), else fused. On gfx950 split (ref_fwd_h, BV adaptive 16/32) wins grid-starved; on gfx942 fused is best (split is gated/available but slower).
- **out kernel** (split path): BV adaptive (128 if `NT·B·H ≥ 128` else 64), de-aliased barriers, v_new prefetch, bk-loop num_stages=2 pipeline (out 212→115µs on gfx950).
- **Research / arch-gated, off by default**: `scan32` (32×32×16 MFMA, gfx950-only, `OPUS_GDN_SCAN32`), pure-HIP single-warp scan (`OPUS_GDN_HIP_SCAN`), wavefront (`opus_gdn_wavefront_*_fwd`), `opus_gdn_k2_split.cu`/`opus_gdn_wf_h.cu` (kept for reference).

### Reproduce
- **Must pass `BT=64`** (python default BT=32 → slow fused BT=32 path).
- GPU-kernel time (rocprofv3 kernel-trace) ≠ wall-clock e2e (adds launch+python). opus wins on both; CUDA-graph confirmed.
- The node thermally throttles under back-to-back runs (~1.5× drift); use fresh-process min-of-N or same-session interleaving. Ratios are stable.

---

## 5. Future work (if revisiting gfx942 2×)
- Associative parallel-scan (Blelloch) over chunks for the H-state recurrence (`h_t = decay_t·h_{t-1} + kᵀv`, associative). Must be far leaner than the wavefront's atomic pipeline to beat fused's 1388µs — the combine (matrix decay-scale + add) overhead is the risk.
- Anything that raises fused K2's WG count without per-tile redundancy is the goal; none found so far.
