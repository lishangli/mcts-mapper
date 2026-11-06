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

### Utilisation, stated plainly

```
12 compute ops / 34 cycles = 0.35 ops/cycle
array ceiling (§1.3 of EXECUTION_MODEL.md) = 128 PEs / 2 cycles = 64 ops/cycle
utilisation = 0.55 %
```

The layer is latency-bound, not ALU-bound: 34 cycles is roughly one 64-cycle AG
window, so the kernel ends before the array can be filled. **Correctness is
demonstrated; high performance is not** — and the binding constraint is the
12-node mapping ceiling, which forces layers this small.

## 3. What a LeNet-class workload needs

| Requirement | Status |
|---|---|
| conv layer bit-exact on RTL | done (conv2d_3x3, 42/42) |
| dense layer bit-exact on the temporal flow | done (nn3x2, 2/2 outputs) |
| dense layer with > 12 nodes | **blocked** by the mapping ceiling |
| sustained multiply-accumulate | not demonstrated; needs streaming inputs and II > 1 |

The next engineering step is therefore capacity, not arithmetic: either the
MRRG + II scheduling sketched in `RELATED_WORK.md`, or negotiated-congestion
rip-up to replace the 12-attempt remap.
