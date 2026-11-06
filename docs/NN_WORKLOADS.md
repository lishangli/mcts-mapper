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

## 3.4 Measured streaming throughput (2026-11-07)

Two runs of `nn4x2` with only the AG window length changed (same flow, same
emitter, corrected fence-before-store ordering):

| `S2_CAP_CYCLES` | samples | load | exec | store | total |
|---|---|---|---|---|---|
| 64 | 64 | 98 | 6 | 733 | 846 |
| 128 | 128 | 98 | 5 | 765 | 879 |

Slope = 33 cycles for 64 extra samples ≈ **0.52 cycles/sample**, i.e. the array
absorbs ~1.9 samples per cycle once it is streaming. `nn4x2` computes 16
operations per sample, so:

```
16 ops/sample ÷ 0.52 cycles/sample ≈ 31 ops/cycle
31 / 64 ops/cycle ceiling ≈ 48 % of peak, sustained
```

This is the first measured throughput figure rather than a lower bound, and it
changes the picture: **the array is not the problem — the launch path is.** The
same runs spend 733–765 cycles in the capture DMA, so a 16-operation kernel
achieves 16/846 ≈ 0.02 ops/cycle end-to-end while its steady-state rate is ~31.

Caveats: the two runs have independent MCTS placements, so the slope carries
placement noise; and 0.52 cycles/sample is below `exe_ii = 2`, which means the
AG streams faster than one context per sample and the array pipelines several
samples per context window. Both need a VCD-level check before being quoted as
a hard number.

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
