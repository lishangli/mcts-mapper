# Experiment log

One row per round. A result is recorded only if the run that produced it passed
its own correctness check; a run that failed to map is marked as such and counts
as evidence in neither direction.

| round | hypothesis | change | run | result | decision | commit |
|---|---|---|---|---|---|---|
| 0 | the dense path is DMA-bound, so batching cannot lift it past ~5 ops/cycle | none (baseline sweep) | `l4w5o3c3` at 128/256/512 samples, 1-3 readback channels | 3.67 -> 3.71 -> 4.60 ops/cycle (2560/698, 5120/1378, 10240/2226) | queue E3 to fit the slope and predict the ceiling | 29e57bf |
| 0 | a constant-stream stimulus hides timing errors | `S2_INPUT_RAMP` + `check_seq.py` | `nn3x2` with ramp inputs | outputs became data-dependent (`[252,369]` vs `[10,14]`) but the checker cannot align the capture (warm-up zeros, no reference model) | E1 becomes blocking | e18bb22 |

## Retracted

| claim | why it fell | where |
|---|---|---|
| 0.52 cycles/sample, ~31 ops/cycle streaming | capture values were wrong (window > 64 with a single-burst DMA) | `EXECUTION_MODEL.md` section 3.6 |
| "the store phase is bandwidth-bound" | 64 B and 512 B stores both cost ~715 cycles | section 3.7 |
| a rip-up victim may take any delay | `nn2x6` mapped but captured `[0,1,2]` instead of `[8,4,7,5,4,7]`; operands of one node must arrive in the same cycle | `STATUS.md`, `EXECUTION_MODEL.md` section 2.4 |
