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


## 1b. Measured convolution throughput: 19.8 ops/cycle

The same conv2d_3x3 kernel, instrumented with `rdcycle` around `cgra_execute0`
(driver copied to `tools/conv/main_conv_timed.c`):

```
CONV_PERF cycles=178
42/42 defined outputs bit-exact with the transposed tap order (C^T)
ops = 196 outputs x 9 MACs x 2 ops = 3528
throughput = 3528 / 178 = 19.8 ops/cycle   (31 % of the 64 ops/cycle ceiling)
```

**Repeatability**: three runs of the same ELF give 178 cycles each, i.e. the
number is deterministic in this configuration (0.91 cycles per output, 9.9
MAC/cycle, ~1 output per cycle). Building through `tools/conv/run.sh` (which
takes the execute variant from the benchmark directory) gives 195 cycles and the
same 42/42, i.e. 18.1 ops/cycle — the range 18.1–19.8 is build-variant noise, not
run-to-run noise.

This is the high-performance datapoint and it explains the others: the
convolution reuses every input across nine taps and streams 196 outputs from one
configuration, so the launch cost and the input DMA are amortised over real
arithmetic. The dense-layer flow (`gen_nn`, one multiply per activation per
neuron) reaches only 3.71 ops/cycle for exactly the opposite reason — it is
memory-bound by construction.

| kernel | ops | cycles | ops/cycle | status |
|---|---|---|---|---|
| conv2d_3x3 (spatial, AG streaming) | 3528 | 178–195 | **18.1–19.8** | 42/42 bit-exact |
| l4w5o3c3 dense DAG (256 samples, 1 readback) | 5120 | 1378 | 3.71 | ch0 bit-exact |
| nn4x2 (256 samples) | 4096 | 2066 | 1.98 | [18, 21] bit-exact |

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

With one IOB per input value (stream sharing, 2026-11-07) `nn4x2` drops from 889
to **653 cycles** for the same `[18, 21]`, i.e. 16 ops / 653 = 0.0245 ops/cycle
end-to-end on a 64-sample window.

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


### 3.4b Amortisation curve and the flow's ceiling

The same 20-operation DAG at four batch sizes (1 readback channel except where
noted):

> **Retired 2026-11-07**: the 20-node `l4w5o3c3` case below no longer maps
> (32 minutes with a 1800 s budget, sharing on and off), so these rows are kept
> only as history. The current throughput evidence is §3.4b's small kernels.

| samples | ops | total cycles | ops/cycle |
|---|---|---|---|
| 128 (3 caps) | 2560 | 698 | 3.67 |
| 256 | 5120 | 1378 | 3.71 |
| 512 | 10240 | 2226 | **4.60** |
| 1024 | 20480 | does not complete | — |

The marginal cost between 256 and 512 is **3.31 cycles/sample**, i.e. moving a
sample through this kernel costs ~3.3 cycles no matter how long the batch is.
With 20 operations per sample that caps the kernel at

```
20 ops / 3.31 cycles ~= 6.0 ops/cycle   (asymptote, ~9 % of the 64 ops/cycle ceiling)
```

and the measured 4.60 at 512 samples is already 77 % of that asymptote. So this
flow is **saturated at ~6 ops/cycle**, and the reason is structural: each sample
arrives by DMA (two operand streams, one result), while conv2d's operand window
comes from the AG, which is why it sustains 18-19.

1024 samples does not complete: the emitted program does not terminate within
300 s, on three consecutive attempts, twice. That is the practical limit.

### 3.4c Amortisation curve on a kernel that still maps (nn4x2, 2026-11-07)

16 operations per sample (4 inputs x 2 outputs x 2), stream sharing on, both
capture channels read back:

| samples | ops | total cycles | ops/cycle |
|---|---|---|---|
| 64 | 1024 | 653 | 1.57 |
| 128 | 2048 | 915 | **2.24** |

Marginal cost between the two points is (915 - 653) / 64 = **4.09 cycles per
sample**, so 16 ops / 4.09 gives an asymptote of **~3.9 ops/cycle** for this
16-op layer. The 64-sample point is 40 % of that asymptote and the 128-sample
point 57 %, i.e. this kernel is still launch-dominated at both sizes, which is
exactly what the per-sample DMA model predicts.

256 samples did not complete: the case hit its 1800 s budget without emitting
(1862 s wall clock). So for this kernel the curve stops at 128 too.

### 3.5 The 20-node DAG at batch 128 (best verified throughput)

`dfg_dag_l4w5o3c3` (20 compute nodes, 3 capture channels) with a 128-sample
window:

| kernel | samples | ops | load | store | total | ops/cycle | value |
|---|---|---|---|---|---|---|---|
| `l4w5o3c3` | 128 | 2560 | 61 | 579 | 698 | **3.67** | [6, 6, 4] ✓ |

That is the best verified figure so far (4.6 % of the 64 ops/cycle ceiling), and
it is higher than `nn4x2` at the same idea (1.98) simply because the kernel has
20 operations per sample instead of 16 with a similar launch cost — the same
amortisation argument, measured.

Two routing experiments are recorded as negative results so they are not
repeated blindly:

* routing the constant streams **before** the compute edges (to avoid
  `no const IOB`) makes the 24-node case fail *earlier* (28 nodes instead of
  46): the streams' reserved pins and wires are exactly what the compute edges
  need, so the order matters in the opposite direction;
* the spatial rip-up keeps `l4w5o3c3` and the 20-node result intact (both
  re-verified after the change).

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
