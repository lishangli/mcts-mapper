# Status: what is proven, and what is open (2026-11-07)

A single page that separates measured facts from open work, so a reader (or a
later session) does not have to reconstruct it from the experiment logs.

## Proven on the RTL simulator

| Claim | Evidence |
|---|---|
| Convolution layer runs bit-exactly | conv2d_3x3: 42/42 defined outputs, transposed tap order; `tools/conv/`, `tools/s2/check_conv2d.py` |
| Convolution sustains **18.1–19.8 ops/cycle** (28–31 % of the 64 ops/cycle ceiling) | 3528 ops in 178 (one execute variant) / 195 cycles (the one `tools/conv/run.sh` builds), deterministic per build; `tools/conv/run.sh` reproduces it in one command |
| Dense layers run bit-exactly | `nn3x2` [10,14]; `nn4x2` [18,21] at 64/128/256 samples |
| A 20-node DAG maps and runs bit-exactly | `dfg_dag_l4w5o3c3` [6,6,4]; **3.67 ops/cycle** at 128 samples (698 cycles) after the near-IOB placement pass |
| Dense-layer throughput scales with batch | 1.18 → 1.57 → 1.98 ops/cycle for 64/128/256 samples |
| The mod-4 delivery constraint was over-conservative | `S2_NO_PHASE=1` passes `dfg_chain5` (9) and `dfg_big` (31); it is now the default |
| The capture store is a fixed ~700-cycle cost | 512 B and 64 B stores both cost ~715 cycles |
| The input DMA moves at most 256 B per call | 128-sample windows read stale data until the load was split into bursts |
| Mapping capacity moved 12 → 20 nodes | `stress_report.md`, MRRG plan |
| The suite still passes after the near-IOB and rip-up changes | **all 12 cases re-run individually on 2026-11-07 and PASS**: chain_dfg 6/353, dfg_chain5 9/424, dfg_branch 3/347, dfg_const 8/261, dfg_mixed 13, dfg_scale 11, dfg_multi [8,-2]/358, dfg_big 31/506, chain12 [4,6]/518, nn3x2 [10,14]/615, nn4x2 [18,21]/889, l4w5o3c3 [6,6,4]/698 (value/cycles) |

## Two ceilings (2026-11-07, consolidated)

| flow | measured | ceiling | why |
|---|---|---|---|
| dense DAG, per-sample DMA | 3.67 / 3.71 / 4.60 ops/cycle at 128 / 256 / 512 samples | **~6 ops/cycle** (marginal 3.31 cycles/sample x 20 ops) | every operand of every sample is DMA'd; ~5-6 cycles per stream per launch, so launch cost grows with the graph |
| conv2d_3x3, AG-streamed | **18.1-19.8 ops/cycle**, 42/42 bit-exact, 195 cycles via `tools/conv/run.sh` | array-limited (31 % of 64) | nine taps are spatially reused, one configuration does 1764 MACs |

1024 samples is out of reach: the emitted program does not terminate (three
300 s timeouts, twice). That is where this flow stops.

**The 20-node figure is retired (2026-11-07)**: `dfg_dag_l4w5o3c3` no longer
maps — 28 minutes with stream sharing, 17 minutes with a bounded stream search,
and **32 minutes with sharing off**, i.e. longer than the 1385 s it once took.
Sharing is therefore not the cause; the DFG has become marginal. Its 698-cycle /
3.67 ops-per-cycle entry is withdrawn rather than quoted as stale.

What replaced it as the throughput evidence: the small kernels, all bit-exact and
all faster after stream sharing — nn3x2 615 -> 530, nn4x2 889 -> **653**,
chain12 487 -> 448, dfg_big 506 -> 390, dfg_multi 358 -> 318.

## Open, with the next concrete action

| Open item | Next action |
|---|---|
| A 24-node DAG (LeNet-class width) does not map — the *correct* next attempt is known | Allowing a rip-up victim to change delay made it map but produced wrong values (`[0,1,2]` vs `[8,4,7,5,4,7]`), proving the rule: two operands of one node must arrive in the same cycle. So a victim may change delay only if **all operands of its node are re-routed together** with a consistent alignment. That is a bounded change and the next thing to implement | 60 of the nodes route (near-IOB placement + cross-node rip-up); it dies on the last one or two edges. Tuning is exhausted — the next step is structural: a time-expanded (PE, ctx) resource graph, or allowing a victim to change delay with its consumers re-scheduled |
| No II > 1 modulo scheduling | Implement a (PE, ctx) time-expanded resource graph; the RTL already rotates 4 contexts with `exe_ii = 2` |
| Dense-layer throughput stays ~3.7 ops/cycle | It is memory-bound by construction (one multiply per activation); the fix is kernels with tap reuse, not more batching |
| `S2_REPEAT > 1` hangs | The controller rejects a second `execute()` without reconfiguration; multi-iteration must come from the schedule |

## A methodological limit worth knowing

Every kernel in this suite feeds **constant** streams (`gen_nn`, `gen_dag` write
one value into the whole input array), so a change that shifts *when* a value
arrives can still produce the right numbers. That is why the failed
delay-freedom experiment produced wrong values only for `nn2x6`'s shape and not
for the smaller kernels: constant inputs cannot validate timing changes.

Any future re-scheduling work (II, delay insertion, node re-timing) needs a
testbench with **data that varies per sample** — the convolution's AG window is
the only such kernel here. Without it, a mapping change can pass every test and
still be wrong on real data.

The ramp testbench now runs and it **fails the default flow**, which is the most
important open correctness result in this repo:

```
nn3x2, 64 samples, S2_INPUT_RAMP=1 (stream i carries base_i + j at sample j)
  channel 0: 49 consecutive samples fit  capture[i] = ref(i-15) + 2
  channel 1: 50 consecutive samples fit  capture[i] = ref(i-12) - 2
```

A constant residual that persists for 50 samples is not noise and not latency:
it means the operands of a node are reading **different sample indices**, i.e.
the mapping is wrong for data that varies per sample even though it is bit-exact
on constant streams.

The obvious fix does not work: re-enabling the mod-4 delivery check
(`S2_NO_PHASE=0`, the constraint that would force every operand of a node into
the same context) makes `nn3x2` **fail to map** (6 remaps, final conflict
`54 otrackS_0`: edge16 wants idx 2, stream5 wants idx 4). So the alignment has to
be restored in the emitter's delivery path rather than by refusing to schedule.

Reproduced on the smallest possible graph, so this is the emitter and not the
generator: `dfg_quad.json` is `y = (x+1)*x` (one stream, one constant, two
compute nodes) and a quadratic grid search fits **skew = -1 sample** (rmse 0.38
over 56 samples) where the aligned hypothesis is off by 30x (rmse 31). A
quadratic kernel is required here: on a linear one, a latency and a skew are
indistinguishable.

**Alignment is implemented but parked (2026-11-07)**: on the minimal kernel it
works for both stimuli (`dfg_quad`: constant 2, ramp skew 0), but at layer scale
`nn3x2` with `S2_ALIGN_OPNDS=1` produces a program that hangs in simulation on
**every** attempt (0/1 PASS after 900 s = 3 x 300 s timeout), and about 1 in 6
`dfg_quad` placements does the same. `perf.py` now bounds each run
(`S2_SIM_TIMEOUT`, `S2_SIM_RETRY`) so a hang is a failed attempt rather than an
infinite wait, which is a real improvement independent of alignment.

**Earlier note: alignment implemented but not enabled**: routing every operand
of a node at the same delay does remove the skew (`dfg_quad`: -1 -> **0**,
rmse 0.00), but on some placements it makes the node read the *same* source on
both operand pins — `dfg_quad` then computes `(x+1)^2` instead of `(x+1)*x`.
That is a worse failure than the skew, so `S2_ALIGN_OPNDS` defaults to **off**
and the flag stays for experiments. The collision is between a stream path and a
compute path on the same mux port and is the next thing to fix.

**Earlier partial fix in the tree (2026-11-07)**: the router routes every operand of
a node at the *same* registered-hop delay (`S2_ALIGN_OPNDS`, default on), using
a new `route_exact` and a delay argument on `iob_stream_route` so a stream pin
matches the node's routed operands too. Effect: `dfg_quad` goes from skew -1
to **skew 0, rmse 0.00**, and `nn3x2`'s ramp residual drops from +2 to -1 —
better, but not zero, so on a multi-node tree some node still fails to pad its
shorter operand (the fallback is silent). The constant regression is intact
(`nn3x2` [10,14] at 665 cycles vs 615 before; `chain12` [4,6] at 487).

The mechanism is now clear: a node's two operands arrive over paths of different
length (the `x+1` branch has an extra hop), and nothing in the emitter equalises
them, so the multiply reads sample j on one input and j-1 on the other.

Scope of the damage, stated precisely: every **throughput** figure in this repo
was measured with constant stimuli and the same schedule, so those numbers stand.
What is now unproven is the stronger claim that the flow is correct for real
(non-constant) data. Next step: ramp one input at a time to identify which
operand carries the skew.

## Retracted claims (kept so they are not repeated)

* 0.52 cycles/sample and 31 ops/cycle — came from runs whose captured values were
  wrong (window > 64 with a single-burst DMA), withdrawn in §3.6.
* "store is bandwidth-bound" — it is a fixed DMA setup cost, §3.7.

## Reproduce

```bash
# dense layers and DAGs (MCTS + netlist routing + RTL)
cd tools/s2 && python perf.py            # 12-case regression with cycle counts
python gen_nn.py --inputs 4 --outputs 2 --name nn4x2

# convolution layer (compiler flow)
# build tools/conv/main_conv_timed.c as described in tools/conv/README.md
``## Root cause of the misalignment (2026-11-07)

Every skew, and every "duplicate operand", since round 1 has one cause:
`S2_NO_PHASE=1` (the old default) drops the mod-4 context-residue constraint, so
a route may take any delay and a node's two operands can sample **different
samples**. The binary probe makes it unambiguous — `y = (x+1) - x`:

    S2_NO_PHASE=1   ->  -2      (wrong)
    S2_NO_PHASE=0   ->   1      (correct)

and `dfg_quad` with `NO_PHASE=0` returns the exact sequence 20, 30, 42, … 3080
(skew 0, rmse 0.50) where it used to return perfect squares.

The default is now `S2_NO_PHASE=0`. The cost is that routes must hit an exact
residue, and it is **paid immediately and completely**: `nn3x2` — the 12-node
dense layer that used to run in 530 cycles — no longer maps at all (1862 s, six
remap attempts), failing on

    conflict 30 otrackE_0 (idx 3 from edge21(20->21) vs idx 0 from stream0(iob11->15))

i.e. a stream wants the same GIB port an edge already drives — and on a different
draw the same case fails as `no parity route 20->21 (bumped 3)` instead. Both are
the same weakness: with an exact residue required, the router either cannot find
a route with that residue or finds one that collides. **The constraint is
satisfiable for 3-5 node graphs (dfg_quad, dfg_add1, dfg_dsub all route and are
now numerically correct) and not for 12.** Making it usable is a router project. Three mitigations were tried and all three
fail at 12 nodes: reserving the stream path before the rip-up victims are
re-routed (conflict just moves), routing the longest operand first (failure moves
to a different edge), and trying all four `c0` launches (failure moves again).
The last two failures report **free resources with no admissible route** —
`kind: congestion, free_hops 9, free_pins [4,5,6,7]` — so the required residue is
simply not reachable by the router's path set.

The default stays 0: large graphs then fail loudly instead of producing silent
wrong data, which is the intended trade.

`