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
| branch | 3 | 22 | 153 | (pending) | 3 |
| chain_dfg | 4 | 25 | 162 | (pending) | 6 |
| dfg_chain5 | 4 | 5 | 6 | (pending) | 9 |
| dfg_scale | 5 | 4 | 5 | (pending) | 11 |
| dfg_big | 10 | 4 | 6 | (pending) | 31 |

Two facts already follow from the split:

* `execute()` returns before the array has drained: its cost (4–25 cycles) is the
  command issue path, not the kernel.
* the capture window dominates the wall clock, so **`total` (through `fence`)** is
  the metric to model; the split is kept because it separates command latency from
  memory latency.

The model being fitted is

```
total ≈ t_start + ctx_granularity * (max_ctx + fill) + capture_cycles + dma_cycles
```

with `ctx_granularity = exe_ii = 2`, `capture_cycles = 64`, and `dma_cycles`
proportional to the store length (512 B per output channel per IOB).

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

## 5. Validation status

* `python tools/s2/regression.py` → 9/9 PASS (values, not timing).
* `python tools/s2/perf.py` → measured cycles per case (see table).
* Not yet modelled: AG startup latency, per-hop bank conflicts, and the
  `execute` command overhead as a function of config length.
