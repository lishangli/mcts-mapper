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

First attempt at such a testbench (`S2_INPUT_RAMP=1`, `check_seq.py`), and what
it did and did not establish:

* the run does show the outputs are data-dependent now (`nn3x2` captures
  `[252, 369]` where the constant stimulus gives `[10, 14]`), so a shifted
  schedule can no longer hide behind identical samples;
* it is **not yet a pass/fail criterion**: the captured sequence starts with the
  pipeline's warm-up zeros and the reference has no warm-up model, so
  `check_seq.py` cannot align the two (it reports "MISMATCH" with no offset for
  both channels). Treat the ramp run as evidence of sensitivity, not as a
  verified correctness test, until the alignment model exists.

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
```
