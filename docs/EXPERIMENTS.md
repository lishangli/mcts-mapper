# Experiment log

One row per round. A result is recorded only if the run that produced it passed
its own correctness check; a run that failed to map is marked as such and counts
as evidence in neither direction.

| round | hypothesis | change | run | result | decision | commit |
|---|---|---|---|---|---|---|
| 0 | the dense path is DMA-bound, so batching cannot lift it past ~5 ops/cycle | none (baseline sweep) | `l4w5o3c3` at 128/256/512 samples, 1-3 readback channels | 3.67 -> 3.71 -> 4.60 ops/cycle (2560/698, 5120/1378, 10240/2226) | queue E3 to fit the slope and predict the ceiling | 29e57bf |
| 0 | a constant-stream stimulus hides timing errors | `S2_INPUT_RAMP` + `check_seq.py` | `nn3x2` with ramp inputs | outputs became data-dependent (`[252,369]` vs `[10,14]`) but the checker cannot align the capture (warm-up zeros, no reference model) | E1 becomes blocking | e18bb22 |
| 1 | with a per-sample ramp, a correct mapping shows `capture[i] = ref(i-L)` exactly, so any deviation is a constant residual | rewrote `check_seq.py`: bases read from the generated driver, latency chosen by smallest |residual| | `nn3x2` @64 samples, `S2_INPUT_RAMP=1` | ch0: 49 samples at L=15 with **r=+2**; ch1: 50 samples at L=12 with **r=-2** — a constant residual, i.e. the operands of a node do not all carry the same sample index | E1 partial: the detector works and the default flow **fails** it. A/B with `S2_NO_PHASE=0` answered the "just re-enable the check" idea: **it does not map** (6 remaps, final conflict `54 otrackS_0`: edge16 wants idx 2, stream5 wants idx 4) | next round: ramp **one input at a time** to identify which operand carries the skew, since the fix depends on it | 753587a |
| 2 | streams are **per-consumer copies**, so ramping one copy moves exactly one output channel | `S2_RAMP_ONE=k` in the emitter: ramp a single stream, leave the rest constant | `nn3x2` twice: ramp si0 (IOB2) and si3 (IOB3), both carrying x0 | ramp si0 -> `got=[96, 14]` (only ch0 moves); ramp si3 -> `got=[10, 106]` (only ch1 moves). Both ramped channels step by exactly the weight (2/sample), so the ramp does reach the PE through its IOB. ch0's residual is -2 = 2 x (-1): the x0 copy at IOB2 leads by **one sample** | skew is **inside one neuron's operand set**, not cross-talk between copies -> next: per-operand attribution (ramp si1/si2, fix the checker's stream->input map) | (this commit) |

## Retracted

| claim | why it fell | where |
|---|---|---|
| 0.52 cycles/sample, ~31 ops/cycle streaming | capture values were wrong (window > 64 with a single-burst DMA) | `EXECUTION_MODEL.md` section 3.6 |
| "the store phase is bandwidth-bound" | 64 B and 512 B stores both cost ~715 cycles | section 3.7 |
| a rip-up victim may take any delay | `nn2x6` mapped but captured `[0,1,2]` instead of `[8,4,7,5,4,7]`; operands of one node must arrive in the same cycle | `STATUS.md`, `EXECUTION_MODEL.md` section 2.4 |