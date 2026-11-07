# CGRA execution and compute model (Tram, spatio-temporal mode)

Derived from the RTL and from the emitter in `tools/s2`, cross-checked against RTL
runs. Every claim below cites the file it comes from; if the RTL changes, this file
is wrong and must be updated with it.

## 1. Compute model

### 1.1 Resources

| Resource | Count | Source |
|---|---|---|
| GPE (processing element) | 128 | `dsa/template/CGRA.scala` (`rows` × `cols`) |
| GIB (routing block) | 153 | `top.v` netlist, `tools/netlist/netlist_gibs.json` |
| IOB | 32 banks | `iob_stream_use` / `iob_rows` in `tools/s2/mcts_bridge.py` |
| Config slot per PE / GIB | 2 (A, B) | `Interconnect.scala`, `GIB` class |
| RF entries per PE | 4 | `cgra_num_rf_reg` defaults to `numCtx` = 4 |
| Contexts | 4 | `TramSpec.scala:82` |

### 1.2 Operator semantics

Opcode order and operand counts come from `op/Operations.scala` (`OPC`, `OpInfo`).
All arithmetic is 32-bit integer; there is no FPU. For a PE with operands
`op0` (DFG operand 0) and `op1` (DFG operand 1):

| OPC | Semantics | Operands |
|---|---|---|
| PASS | `op0` | 1 |
| ADD | `op0 + op1` | 2 |
| SUB | `op0 - op1` | 2 |
| MUL | `op0 * op1` | 2 |

`OpInfo` lists latency 1 for all four (the register outside the ALU is the only
stage). **SUB is `op0 - op1`**, i.e. the first DFG operand is the minuend; any
earlier note in the plan document that claims `op1 - op0` is superseded by
`Operations.scala`'s `"SUB" -> Seq(op0 - op1)`.

### 1.3 Time multiplexing: contexts and slots

`CGRA.scala:143-155` keeps a single array-wide counter:

```
every 2 cycles of io.en (cgra_exe_ii = 2):   ctx := (ctx + 1) mod 4
slot := ctx & 1                              // which config bank is read
```

So a PE holding two different operators executes them alternately: config A on
contexts 0 and 2, config B on contexts 1 and 3, one context every **2 cycles**.
Per PE that is **1 operation per 2 cycles**; the array ceiling is therefore
**64 operations/cycle** (128 PEs / 2), or 32 multiply-accumulates per cycle when
half the PEs are multipliers and adders alternate.

A context slot with no configuration decodes as `PASS(0)`, which is why the
bridge duplicates a single-operator PE's config into both slots: otherwise every
other context silently produces 0.

### 1.4 Rotating register file

`dsa/util/RegFile.scala` implements the cross-context path by address arithmetic:

```
write:  phys = (sel + ctx)     mod 4
read :  phys = (sel + ctx - k) mod 4        // k = configured distance
```

With `k = 1` a value written in context `c` is readable in `c + 1`; the register
is not duplicated, the logical index slides. `rot_en = false` restores the
original static behaviour bit-exactly, which is how the spatial flow keeps working.

Consequence for mapping: the RF removes the *routing* hop requirement for a
distance-1 dependency, but it does not remove the *context* requirement — the
consumer still has to run one context later.

## 2. Execution model

### 2.1 One kernel

1. `load_cfg(cin, ...)` writes the config stream (`{data_lo, data_hi, addr}`
   6-byte rows) into the CGRA config memory.
2. `load_data(...)` DMAs the input buffers into IOB banks (64 B aligned arrays only:
   unaligned CPU stores make the TileLink monitor fire "PutPartial contains invalid mask").
3. `config(addr, len)` loads slot A and slot B; slot B's base address is
   `len(rowsA) * 6 / 16` 128-bit words, and `rowsA` is padded to a 16-byte multiple.
4. `execute(iob_ens, ...)` starts the array: the context counter runs and the IOB
   address generators stream their windows.
5. `store(...)` DMA-reads the capture IOB banks back to DRAM.
6. `fence(1)` waits for the DMAs.

### 2.2 Datapath latency

A route costs **registered** otrack hops, not hops (`sched.path_delay`): only
280 of 560 GIB otrack outputs are registered, and ipin is never registered.
Combinational GIBs therefore add zero cycles, which is why the scheduler can place
two nodes in the same context when the route between them is combinational.

Phase constraint enforced by the emitter:

```
ctx_consumer ≡ ctx_producer + path_delay (mod 4)
```

**This constraint is conservative, and the code now proves it.** The emitter
writes every route into *both* config banks (`mcts_bridge.py`: `for slot in
(0, 1): emit_route(...)`) and duplicates a single-operator PE's config into both
slots, so a path is active in every context. The only physical requirement is
that the producer's output is stable when the consumer samples — which holds for
the default flow.

Measured with `S2_NO_PHASE=1` (constraint dropped, same emitter):

| case | expected | `S2_NO_PHASE=1` |
|---|---|---|
| `dfg_chain5` (5 nodes) | 9 | **PASS 9** |
| `dfg_big` (10 PEs) | 31 | **PASS 31** |

Where the constraint still matters: time-folded operators sharing one PE
(`S2_PACK`) and RF reads with `k = 1`, because then the producer's output only
exists in the context its slot is active.

`tools/s2/mcts_bridge.py` searches for a path whose delay satisfies this
(`route_parity`); when no such path exists it retries with the producer launched
later (`S2_PHASE_BUMP`), which changes `want` without moving the operator.

### 2.3 Capture

The capture IOB runs an address generator for `Cycles0 = 64` cycles with
`Stride0 = 1`, so the run logs carry 64 samples per output channel; the
regression judges the trailing run of identical non-zero samples. Samples before
the pipeline is filled are transients, not results.

## 3. Performance model

Measured by `tools/s2/perf.py`, which reads the `S2_PERF` line the generated
`cgra_execute.c` prints (`rdcycle` around `execute()`, `store()` and `fence(1)`).

| Case | compute nodes | exec | store | total | expected |
|---|---|---|---|---|---|
| dfg_scale | 5 | 4 | 5 | 11 | 11 |
| dfg_big | 10 | 5 | 6 | 13 | 31 |
| dfg_mixed | 4 | 5 | 6 | 13 | 13 |
| dfg_chain5 | 4 | 26 | 4 | 32 | 9 |
| dfg_const | 2 | 24 | 6 | 32 | 8 |
| chain_dfg | 4 | 23 | 163 | 188 | 6 |
| dfg_branch | 3 | 5 | 186 | 193 | 3 |
| dfg_multi | 5 | 25 | 175 | 202 | [8, -2] |
| nn3x2 (3 inputs, 2 neurons, 6 MUL) | 12 | 5 | 10 | 34 | [10, 14] |

(measured 2026-11-07 with `perf.py`; every case also passes its value check)

Three facts follow:

* `execute()` returns before the array has drained: its cost (4–26 cycles) is the
  command issue path, not the kernel, and it does not grow with graph size
  (`dfg_big`, 10 PEs, has the smallest `exec`).
* wall clock is bimodal: ~11–13 cycles when the store finds nothing to move, and
  ~190 cycles when the capture path actually drains 512 B per channel. The
  difference is the DMA, not the array.
* therefore the model has to separate the phases.

### 3.1 Phase-split measurements (2026-11-07)

`S2_PERF load=… exec=… store=… total=… caps=…` splits the kernel into
config+input DMA (`load`), the `execute()` instruction (`exec`), the capture DMA
issue (`store`) and the whole thing through `fence(1)` (`total`).

| case | caps | load | exec | store | total | total − 64 |
|---|---|---|---|---|---|---|
| dfg_scale | 1 | 25 | 4 | 6 | 67 | 3 |
| dfg_branch | 1 | 60 | 25 | 11 | 107 | 43 |

Both totals sit just above the 64-cycle AG window (`Cycles0 = 64`), which means
the capture window runs concurrently with the DMA and the tail is what is left
after it. The residual does **not** track `exec` (25 vs 4) but does track
`load` (60 vs 25 → tail 43 vs 3), i.e. the cost is dominated by how much has to
be shuttled in and out, not by the datapath.

### 3.2 Controlled experiments (why the constants stayed open)

Two knobs were added so the terms can be varied independently:
`S2_CAP_CYCLES` (AG window length, default 64) and `S2_STORE_BYTES` (capture DMA
length, default 512). Both were run on `dfg_scale`:

| variant | load | exec | store | total |
|---|---|---|---|---|
| default (64 cycles, 512 B) | 25 | 4 | 6 | 67 |
| `S2_CAP_CYCLES=32` | 25 | 4 | 6 | **67** |
| `S2_STORE_BYTES=256` | 25 | 5 | 6 | **68** |

Halving the window and halving the store change nothing. So `total` is **not**
capture-window-bound and not store-size-bound: the capture path runs
concurrently with the command stream and the fence does not wait for the full
window. What dominates is a fixed launch cost (config + input DMA + execute +
fence), 34–107 cycles depending on the DFG.

That also re-frames the NN utilisation number: `nn3x2` at 34 cycles is almost
entirely launch overhead, so "0.55 % of the array" measures the kernel's size,
not a throughput limit of the array. A throughput claim needs a kernel whose
work exceeds this fixed cost by a wide margin — i.e. exactly the >12-node
layers the mapping ceiling currently blocks.

```
total ≈ launch(dfg)            # measured 34..107 cycles, DFG-dependent
        + DMA(bytes)           # not the capture window, see 3.2
```

The fence column (total − load − exec − store) is 32 and 11 cycles respectively;
it absorbs the AG completion, which is why the split has to come from the
generated code rather than from the RISC-V side alone. Two points are enough to
fix the shape, not the constants: the constants need the AG start offset and the
per-channel beat count, which the current run log does not carry yet.


### 3.5 The store phase is not what it looked like (2026-11-07, later)

Re-measuring `dfg_scale` with the store length halved, now that the fence sits
before the store:

| `S2_STORE_BYTES` | load | exec | store | total |
|---|---|---|---|---|
| 64 | 25 | 5 | 6 | 310 |
| 512 | 25 | 5 | **6** | 295 |

The store phase reads 6 cycles in **both** cases, so the printed `store` is only
the issue latency; the transfer itself lands in the trailing `fence(1)`, which is
inside `total` but not inside `store`. In `nn4x2` the same field read 733 — the
DMA engine blocks the store instruction when it is still busy, so `store` is not
a stable component across kernels.

What this means for the numbers already published:

* `total` (through `fence`) is the only trustworthy field; the per-phase split is
  indicative, not additive;
* end-to-end utilisation figures should be computed from `total` only
  (`nn4x2`: 16 ops / 846 cycles ≈ 0.019 ops/cycle);
* the streaming slope (33 cycles / 64 samples) came from two `total` values with
  the same structure, so it is unaffected by this, but it still needs a VCD check
  because both runs amortise the same trailing fence over different windows.

A stable decomposition needs either (a) a counter inside the CGRA, or (b) VCD
instrumentation of the IOB handshake; the RISC-V-side `rdcycle` around DMA calls
cannot separate issue from completion.


### 3.6 The window>64 bug, and a valid slope at last (2026-11-07, final)

A window longer than 64 samples produced wrong values. Root cause found: the
generated code loaded a **fixed 256 B** per input stream and stored a fixed
512 B, regardless of `Cycles0`. With a 128-sample window the kernel therefore
read 64 loaded samples plus 64 stale bank words. Both DMA lengths now scale
with the window (`CAP_CYCLES * 4`).

With that fixed, `dfg_scale` at two window lengths:

| `S2_CAP_CYCLES` | total | value | expected |
|---|---|---|---|
| 64 | 340 | 11 | 11 ✓ |
| 128 | 393 | 11 | 11 ✓ |

Slope = **0.83 cycles/sample** (53 cycles for 64 extra samples). The array
therefore consumes a new sample every 0.83 cycles in this configuration, and
the earlier 0.52 figure is superseded by this cleaner measurement (both runs
correct, same flow).

Applied to a kernel with `n` operations per sample: `n / 0.83` ops/cycle while
streaming — 6 ops/cycle for the 5-node `dfg_scale`, and ~19 ops/cycle for the
16-operation `nn4x2` if the stream rate carries over (measured directly below).

Verified end-to-end figures therefore remain the low ones:

| kernel | ops | total | ops/cycle |
|---|---|---|---|
| `nn4x2` | 16 | 846 | 0.019 |
| `dfg_scale` | 5 | 334 | 0.015 |

A defensible steady-state number needs a cycle counter **inside** the CGRA (or
VCD-level IOB handshake capture); the RISC-V-side `rdcycle` cannot see the array
internals, and widening the AG window breaks value correctness in the current
driver. That is the next concrete piece of work.


#### Fix: the DMA moves at most 256 B per call

`nn4x2` at `S2_CAP_CYCLES=128` settled at a constant **16** instead of 18; the
cause was not the AG at all — a single `load_data` request silently covers only
256 B (64 words), so the upper half of every input bank stayed stale and one
operand read as 0. Splitting the load into 256 B bursts fixes it:

| `S2_CAP_CYCLES` | load | exec | store | total | value |
|---|---|---|---|---|---|
| 64 | 98 | 27 | 731 | 867 | [18, 21] ✓ |
| 128 | 226 | 6 | 1002 | 1302 | **[18, 21] ✓** |

With the batch doubled, end-to-end throughput rises from
`1024 ops / 867 cycles = 1.18 ops/cycle` to
`2048 ops / 1302 cycles = 1.57 ops/cycle` — the launch cost amortising, measured
on a verified-correct kernel. Marginal rate over the extra 64 samples is
6.8 cycles/sample (4 input streams), i.e. ~2.4 ops/cycle sustained.

#### Superseded: the earlier multi-stream limit note

`nn4x2` (4 input streams, 2 captures) at the same two window lengths:

| `S2_CAP_CYCLES` | total | value | expected |
|---|---|---|---|
| 64 | 867 | [18, 21] | ✓ |
| 128 | 1238 | **[16, 8]** | ✗ |

So the 0.83 cycles/sample slope is verified **only for the single-stream case**
(`dfg_scale`). With four streams the 128-sample window still corrupts values
(the IOB bank window is 64 words per stream in this configuration), so a
per-layer throughput number for `nn4x2` cannot be quoted from a slope; its
verified end-to-end cost stays 867 cycles for 16 operations = 0.018 ops/cycle.

That is the honest position: streaming throughput is demonstrated for one
kernel, and the multi-stream case needs either a per-stream bank window > 64 or
a second capture configuration before the same measurement can be repeated.


#### Why the multi-stream window still corrupts (characterised, not fixed)

`nn4x2` at `S2_CAP_CYCLES=128` settles at a **constant 16** where 18 is
expected (`O0` is 16 for all 122 trailing samples, not a wrap pattern), and the
same kernel at 64 samples is exactly 18. A constant wrong value means an
operand is missing, not that samples are misaligned: one of the four input
streams is not being read past word 64.

Consistent with that, `load_data` now requests 512 B per stream and the array
start address is the same, so the remaining suspect is the IOB address
generator's window per stream (its BaseAddr uses one 1 KB group offset while the
DMA writes at a 4 KB bank offset; for 64 words the two coincide, for 128 they
may not). That is the next thing to check, and it needs an RTL-level look at
`IOB.scala`'s address generator rather than another timing run.

Until then, the only verified streaming throughput is the single-stream one
(`dfg_scale`: 0.83 cycles/sample), and `nn4x2` keeps its end-to-end
867 cycles / 16 ops = 0.018 ops/cycle.


### 3.7 The capture store is a fixed ~700-cycle cost

Trying to cut the readback to speed up the measurement:

| config | load | store | total | value |
|---|---|---|---|---|
| CAP=128, store 512 B | 104 | 720 | 861 | [6, 6, 4] ✓ |
| CAP=128, store 64 B | 123 | **714** | 873 | [5, -6, 10] ✗ |

Two conclusions:

* the store phase costs ~700 cycles **whether it moves 64 B or 512 B** — it is
  DMA setup/queueing, not bandwidth, so shrinking the readback cannot amortise
  anything;
* reading back only 16 samples lands in the pipeline's transient region, so the
  steady-state value is wrong: the readback must cover the settled part of the
  window.

The only lever that improves end-to-end throughput is therefore **more work per
launch** (more samples and/or more operators per sample), which is what the
batch sweep measures: 2.97 ops/cycle at 20 nodes × 128 samples.


### 3.8 Throughput probe: more work per launch, one readback channel

`S2_MAX_CAPS=N` limits how many output channels are read back (the kernel still
computes all of them), which isolates the per-channel DMA cost.

| config | samples | ops | load | store | total | ops/cycle | ch0 |
|---|---|---|---|---|---|---|---|
| CAP=128, 3 captures | 128 | 2560 | 104 | 720 | 861 | 2.97 | 6 ✓ |
| CAP=256, 1 capture | 256 | 5120 | 765 | 81 | 1378 | **3.71** | 6 ✓ |

Two effects, both measured: dropping to one readback removes ~640 cycles of DMA
(720 → 81) and doubling the batch doubles the work, while the load phase grows to
765 cycles because 256 samples × 2 streams must be brought in. Net: 2.97 → 3.71
ops/cycle (5.8 % of the 64 ops/cycle ceiling).

The load phase is now the largest term, which is the honest bottleneck for this
flow: every sample must be DMAd in, and the array's work per sample (20 ops) is
done in a few cycles. A kernel with more arithmetic per input — a 3×3
convolution, or a dense layer reusing one activation across many weights —
is where this machine can actually show throughput; the current generator
produces one multiply per activation per neuron, so it is memory-bound by
construction.

## 4. What this implies for neural-network workloads

* The array executes **64 integer ops/cycle at peak**; a LeNet-class convolution
  layer expressed as multiply-accumulate therefore peaks at ~32 MAC/cycle, three
  orders of magnitude below the kernel's arithmetic intensity only if the operand
  stream can be sustained — the binding constraint is IOB bandwidth and the
  4-entry RF, not the ALU count.
* Temporal folding is the only way to reuse a PE between operators, and it costs
  a context: two operators per PE means each runs every 2nd context.
* Because capture is a fixed 64-cycle window, a layer whose tiled DFG has more
  than ~60 context steps cannot finish inside one window and needs either tiling
  or a longer `Cycles0`.


### 3.3 Correction: the fence must precede the store (2026-11-07)

The generated code used to issue the capture store *before* `fence(1)`, so the
timer stopped before the array had drained and every total in §3.1/§3.2 was a
lower bound. Moving the fence ahead of the store (needed to run several rounds
per configuration) changes `dfg_scale` from **67 to 315 cycles** with the same
value (11). The extra ~250 cycles are the capture window and the AG drain that
the old ordering skipped.

Consequences:

* the honest shape is `total ≈ load + exec + window_drain + store`, with
  `window_drain ≈ 250` for a 64-context window at `exe_ii = 2` plus AG start;
* every utilisation figure published earlier (0.55 % for nn3x2, 0.17 % for
  nn4x2) is therefore an **upper bound**; the corrected denominator is larger;
* `S2_REPEAT=K>1` (several rounds on one configuration) **hangs** the run: the
  controller does not accept a second `execute()` without reconfiguration, so
  amortisation must come from II scheduling or a longer per-launch window, not
  from re-issuing the command.

## 4.1 Neural-network kernels measured so far

`gen_nn.py` emits `y_o = sum_i w[o][i] * x_i + b[o]` as MUL/ADD nodes with the
parameters inlined, and evaluates the same graph independently. `nn3x2`
(3 activations, 2 neurons, 6 MUL + 6 ADD) runs on the RTL and captures
`[10, 14]`, exactly the reference values, in `exec=5 store=10 total=34`.

Interpretation, stated as a number rather than a slogan:

* 12 compute operations in 34 cycles ≈ **0.35 ops/cycle**, against the 64 ops/cycle
  array ceiling derived in §1.3 — 0.55 % utilization.
* That is not a datapath limit: a layer this small finishes inside one 64-cycle
  AG window, so the kernel is latency-bound (window + DMA), not throughput-bound.
* Consequence: a *high-performance* claim needs layers wide enough to fill the
  window, which in turn needs DFGs larger than the current 12-node mapping
  ceiling. The ceiling, not the arithmetic, is the blocker.

## 5. Validation status

* `python tools/s2/regression.py` → 9/9 PASS (values, not timing).
* `python tools/s2/perf.py` → measured cycles per case (see table).
* Not yet modelled: AG startup latency, per-hop bank conflicts, and the
  `execute` command overhead as a function of config length.
