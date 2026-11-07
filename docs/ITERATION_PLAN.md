# Iteration plan: exploration / exploitation under a 30-minute round

Every round is a **falsifiable experiment**, not an edit. The round is only done
when the result is written down, committed and pushed. Anything not measured is
labelled a hypothesis, and anything measured wrong is retracted on the record.

## 1. Round protocol (30 min, fixed shape)

| slot | activity | rule |
|---|---|---|
| 0-5 min | read the queue, pick the round's experiment, restate the hypothesis and the kill criterion | one hypothesis per round, never two |
| 5-22 min | run the experiment (mapping takes 5-15 min; use `S2_ATTEMPTS=24` for probes) | a run that fails to map is **not** a result: resume or re-draw |
| 22-28 min | update `docs/EXPERIMENTS.md` (one row) + the affected model doc, commit, push | a number without a verified run does not go in a doc |
| 28-30 min | apply the feedback rules (section 3), re-rank the queue, schedule the next round | the queue, not habit, decides the next experiment |

Two hard rules learned the expensive way:

* **mapping is stochastic** - a probe that dies on `no route` says nothing about the
  hypothesis; raise the attempt budget before concluding;
* **constant-stream stimuli cannot detect timing changes** - until E1 lands, any
  result that could be explained by a shifted schedule is inconclusive.

## 2. Exploration / exploitation split

Ratio **2 exploit : 1 explore** over every three rounds, enforced by the queue
below rather than by feel.

* **exploit** = a known path to a known blocker (24-node closure, throughput
  ceiling, II). Expected value is bounded but the path is clear.
* **explore** = one cheap probe of an assumption the plan rests on (arithmetic
  intensity, alignment, schedule slack). Expected value is unbounded, but a probe
  must be killable in one round.

Selection rule when the queue is re-ranked: pick the highest
`information_gain / cost` item whose **dependencies are satisfied**; if two are
equal, take the exploit (lower variance) - exploration is scheduled, not
improvised.

## 3. Feedback rules (how a result changes the plan)

| observation | decision |
|---|---|
| E1 cannot align the ramp reference (no consistent offset) | **freeze** all timing-changing work (E2, E5); escalate to a structural check of the emitter |
| E2 maps 24 nodes but values differ from the reference | the alignment rule is violated somewhere: go back to E1, record the failing edge |
| E2 maps and values match | unblock LeNet-class width, promote `nn4x3`/`nn2x6` into the regression suite |
| E3 fits `total = a + b*N` and the predicted ceiling is < 5 ops/cycle | declare the dense path DMA-bound, move the throughput budget to arithmetic-intensity kernels (E4) |
| E4 shows ops/cycle rising with fanout, roughly linearly | write it as a measured law, extend to the largest fanout that still maps |
| any experiment makes no progress in 3 rounds | mark it blocked with the exact failure string; do not retry without a new idea |

No round may end with "still running" as its conclusion. If a run overruns, the
round reports the previous state and the run is picked up next round.

## 4. Queue (re-ranked every round)

| id | kind | hypothesis | minimal change | pass criterion | cost | gain |
|---|---|---|---|---|---|---|
| **E1** | exploit (blocking) | the capture can be aligned to the ramp reference by modelling warm-up plus a constant offset | add a warm-up model to `check_seq.py`, test on `nn3x2` and `nn4x2` | both kernels match over at least 32 steady samples with one offset | 1 | **high** - unlocks every timing experiment |
| **E1d** | exploit (blocking) | the skew disappears when a node's operands are routed with **equal delay** (pad the shorter path with hops) | in the router: for each node, target = max operand delay, route every operand at exactly that delay, rip-up if needed | `dfg_quad` skew 0 (rmse < 2) **and** `nn3x2` full-ramp residual 0, with the constant-input regression still passing | 1-2 | **high** - this is the correctness fix |
| **E1c** | exploit (blocking) | the +-2 residual is the **stream's** own delay, not a capture artefact | minimal kernel: one stream feeding a 2-input node with a const; ramp it and ramp the const-side path, then map stream->input by base value in the checker (needed for the second copy, si>=3) | residual attributable to a named operand, and reproducible across two placements | 1 | **high** - decides whether the fix is AG/IOB addressing or c0/delay |
| **E2** | exploit | a rip-up victim may change delay if **all operands of its node** are re-routed together and `c0` is re-derived | group routes by node in `try_cross_ripup`, re-route the group, recompute `c0`, re-check consumers | 24-node `nn2x6`/`nn4x3` maps **and** the RTL values match | 2 | high - LeNet-class width |
| **E3** | exploit | dense-layer cost is `a + b*N` with `b` set by input DMA, so the ceiling is predictable | run 128/256/512/1024 samples, one readback channel, fit `b` | residuals < 5 %, `b` stable | 1 | medium - decides whether more batching is worth anything |
| **E4** | **explore** | ops/cycle scales with **fanout** (arithmetic intensity), not with batch | new `fanout` shape in `gen_dag.py`: one input to k multipliers with distinct consts, summed | value bit-exact and ops/cycle at least 2x the k=1 case at k=8 | 1 | high - the only way left to test the performance claim without the missing LLVM front end |
| **E5** | explore | there is measurable schedule slack inside one 64-cycle window (i.e. II > 1 is reachable) | measure the sample rate from a ramp capture's period instead of from a widened window | a period below `exe_ii` = 2 cycles confirms slack | 1 | medium - sizes the II payoff before building it |
| **E6** | exploit | a time-expanded (PE, ctx) resource graph is needed for II > 1 | only after E5 shows slack | schedule with II = 2 on a chain kernel, values match | 3+ | high but expensive |

Dependencies: E1 -> {E2, E5}; E5 -> E6; E3 and E4 are independent.

## 5. Round record

Each round appends one row to `docs/EXPERIMENTS.md`:

```
| round | hypothesis | change | run | result | decision | commit |
```

and the affected model doc (`EXECUTION_MODEL.md` / `NN_WORKLOADS.md` /
`STATUS.md`) is updated in the same commit, so a reader never sees a number
without its provenance.

## 6. Guardrails

* push every round; a round that produced no push produced no knowledge;
* retractions stay in the docs with their reason;
* never quote a figure from a run whose values were wrong, even as a "rough" number;
* one hypothesis per round - bundling two changes makes the result unattributable.
