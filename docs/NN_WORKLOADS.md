# Neural-network workloads on the Tram CGRA (measured)

Two kernels are verified bit-exactly on the RTL. Both were re-run and re-checked
on 2026-11-07; the numbers below are from those runs, not from earlier sessions.

## 1. conv2d_3x3 — convolution layer (spatial flow)

* Kernel: 3x3, coefficients `[[2,-3,4],[5,6,7],[-8,-9,10]]`, 16x16 image
  `A[i] = i+1`, 14x14 interior output.
* Flow: `cgra-compiler` `conv2d_3x3` benchmark (SA mapping, 9 input IOB windows,
  AG-driven), dual-slot cin, run as `eg3.riscv`.
* Result: **42/42 defined outputs bit-exact** against the C reference, with the
  control (row-major tap order) at 0/42 — the compiler maps taps column-major,
  so the effective kernel is `C^T`. Checker: `tools/s2/check_conv2d.py`.
* Cost: 9 input windows + one 892-byte store; the layer is throughput-oriented
  (AG streams windows while the array computes), unlike the bridge's
  constant-stream kernels.

## 2. nn3x2 — dense layer (spatio-temporal flow)

* `y_o = sum_i w[o][i] * x_i + b[o]`, 3 activations, 2 neurons,
  `W = [[2,2,1],[2,3,2]]`, `b = [1,0]`, `x = [1,2,3]`.
* Flow: `tools/s2/gen_nn.py` → MCTS placement → S2 routing → dual-slot cin.
* Result: **`[10, 14]` bit-exact** (independent evaluator), `exec=5 store=10
  total=34` cycles, 1 capture channel.

## 3. nn4x2 — wider dense layer (16 nodes)

`gen_nn.py --inputs 4 --outputs 2`, `W = [[1,1,1,1],[2,1,3,2]]` (seed 0),
`x = [1,2,3,4]`, expected `[18, 21]`.

* Result: **`[18, 21]` bit-exact**, `load=98 exec=5 store=29 total=143`, 2
  capture channels, mapped at the first attempt (the 20-node ceiling was broken
  the same day by dropping the mod-4 delivery constraint).

### Utilisation, stated plainly

| layer | compute ops | total cycles | ops/cycle | % of 64 ops/cycle |
|---|---|---|---|---|
| nn3x2 | 12 | 34 | 0.35 | 0.55 % |
| nn4x2 | 16 | 143 | 0.11 | 0.17 % |

Wider is **worse**, and the phase split says why: `load` went 25 → 98 cycles
while `exec` stayed at 5. The kernel pays for its configuration stream and input
DMA in proportion to the graph, and the array only runs for the few cycles the
values are in flight. **Correctness is demonstrated; high performance is not.**
Two things are needed for a throughput claim:

1. amortise the launch cost — many windows (or many iterations) per configuration,
   i.e. II scheduling over loop iterations, or a steady AG stream;
2. a layer whose work is large relative to its configuration, which the 20-node
   mapping ceiling no longer prevents but which this flow still cannot feed from
   memory.

## 3.4 Measured throughput (2026-11-07, corrected)

Two `nn4x2` runs differing only in the AG window (after fixing the DMA burst
bug — a single `load_data` covers only 256 B):

| `S2_CAP_CYCLES` | samples | ops | load | store | total | value | ops/cycle |
|---|---|---|---|---|---|---|---|
| 64 | 64 | 1024 | 98 | 731 | 867 | [18, 21] ✓ | 1.18 |
| 128 | 128 | 2048 | 226 | 1002 | 1302 | [18, 21] ✓ | 1.57 |
| 256 | 256 | 4096 | 1407 | 623 | 2066 | [18, 21] ✓ | **1.98** |

```
batch 64 -> 128 : +0.39 ops/cycle   (marginal 6.8 cycles/sample)
batch 128 -> 256: +0.41 ops/cycle   (marginal 5.97 cycles/sample ≈ 2.7 ops/cycle)
```

The trend is the point: **throughput keeps rising with batch size because the
launch cost amortises**, and the marginal rate (6 cycles/sample) is dominated by
moving four operands and storing two results, not by the array. At 256 samples
the kernel sustains **1.98 ops/cycle end-to-end**, 3.1 % of the 64 ops/cycle
ceiling — still launch/DMA-bound, and II>1 scheduling is what would lift it.

Two things this establishes with verified-correct runs:

* the **launch cost amortises**: doubling the batch raises end-to-end
  throughput by 33 % with no kernel change;
* the marginal rate is 6.8 cycles/sample for a four-stream layer, dominated by
  moving four operands per sample — not by the ALU.

Earlier figures (0.52 cycles/sample, 31 ops/cycle) came from runs whose values
were wrong and are withdrawn.

## 4. What a LeNet-class workload needs

| Requirement | Status |
|---|---|
| conv layer bit-exact on RTL | done (conv2d_3x3, 42/42) |
| dense layer bit-exact on the temporal flow | done (nn3x2, 2/2 outputs) |
| dense layer with > 12 nodes | **blocked** by the mapping ceiling |
| sustained multiply-accumulate | not demonstrated; needs streaming inputs and II > 1 |

The next engineering step is therefore capacity, not arithmetic: either the
MRRG + II scheduling sketched in `RELATED_WORK.md`, or negotiated-congestion
rip-up to replace the 12-attempt remap.
