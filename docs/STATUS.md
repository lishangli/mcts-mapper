# Status: what is proven, and what is open (2026-11-07)

A single page that separates measured facts from open work, so a reader (or a
later session) does not have to reconstruct it from the experiment logs.

## Proven on the RTL simulator

| Claim | Evidence |
|---|---|
| Convolution layer runs bit-exactly | conv2d_3x3: 42/42 defined outputs, transposed tap order; `tools/conv/`, `tools/s2/check_conv2d.py` |
| Convolution sustains **19.8 ops/cycle** (31 % of the 64 ops/cycle ceiling) | 3528 ops in 178 cycles, identical across three runs |
| Dense layers run bit-exactly | `nn3x2` [10,14]; `nn4x2` [18,21] at 64/128/256 samples |
| A 20-node DAG maps and runs bit-exactly | `dfg_dag_l4w5o3c3` [6,6,4]; 2.97 ops/cycle at 128 samples |
| Dense-layer throughput scales with batch | 1.18 → 1.57 → 1.98 ops/cycle for 64/128/256 samples |
| The mod-4 delivery constraint was over-conservative | `S2_NO_PHASE=1` passes `dfg_chain5` (9) and `dfg_big` (31); it is now the default |
| The capture store is a fixed ~700-cycle cost | 512 B and 64 B stores both cost ~715 cycles |
| The input DMA moves at most 256 B per call | 128-sample windows read stale data until the load was split into bursts |
| Mapping capacity moved 12 → 20 nodes | `stress_report.md`, MRRG plan |

## Open, with the next concrete action

| Open item | Next action |
|---|---|
| A 24-node DAG (LeNet-class width) does not map | It now routes all nodes and dies on the last constant stream / last long edge. Options: reserve IOBs at placement time, or allow victims to change delay and re-schedule their consumers |
| No II > 1 modulo scheduling | Implement a (PE, ctx) time-expanded resource graph; the RTL already rotates 4 contexts with `exe_ii = 2` |
| Dense-layer throughput stays ~3.7 ops/cycle | It is memory-bound by construction (one multiply per activation); the fix is kernels with tap reuse, not more batching |
| `S2_REPEAT > 1` hangs | The controller rejects a second `execute()` without reconfiguration; multi-iteration must come from the schedule |

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
