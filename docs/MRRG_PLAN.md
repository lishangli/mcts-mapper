# MRRG-lite: time-expanded routing occupancy (next milestone)

## Why

The 20-node wall is not capacity. Measured (see `tools/s2/stress_report.md`,
2026-11-07 section): the failing operand has all four pins free, a 3-hop path
exists once occupancy is ignored, and the whole array is only 7–12 % occupied
(113–196 of 1668 muxes). The block is the set of GIB muxes that feed the
destination pins.

`Occ` treats `(gib, port)` as permanently exclusive. Physically a GIB has two
config banks (`cfgA` / `cfgB`) selected by `ctx_ptr(0)` — RTL:
`dsa/template/CGRA.scala` lines 516–521, `cfg_slot_wr` in the GIB class. Two
paths whose value reaches the same wire in contexts of opposite parity can share
it: the mux selects source A in slot 0 and source B in slot 1.

## Algorithm

1. **Slot of a hop.** A value leaves the producer at `ctx_p`; every registered
   otrack hop advances the context by one (`exe_ii = 2` cycles each). So for a
   route of total delay `H = path_delay(path)` ending at consumer context
   `c0`:

   ```
   ctx = c0 - H
   for hop in path (source -> destination):
       if is_reg(hop): ctx += 1
       slot[hop] = ctx & 1
   ```

   `ctx` is the context in which the value is *on that wire*, which is exactly
   the context that selects the config bank the mux must be programmed in.

2. **Occupancy key** becomes `(gib, port, slot)`. Three states per wire:
   free, owned in one slot, owned in both.

3. **Two-pass per node.** The current flow routes freely and repairs parity
   afterwards, so the slot is unknown during the first route. Change it to:
   * pass 1 — route freely (conservative claims, slot = any) to obtain `base_c0`
     from the delays;
   * pass 2 — release this node's claims, re-route every operand with
     `route_parity(want = (base_c0 - ctx_p) % 4)`, claiming with the exact
     per-hop slot from (1).
   Every claim is then slot-exact, so later nodes can legally share the other
   slot.

4. **Config emission** already writes `gib_owner[(slot, blk, dstport)]`, so two
   paths sharing one wire in different slots produce two consistent configs —
   no change needed there.

## Acceptance

| Check | Criterion |
|---|---|
| No regression | `regression.py` 10/10 (values, not just mapping) |
| Capacity | `dfg_dag_l4w5o3c3` maps and captures `[6, 6, 4]` |
| A/B | `S2_SLOT_OCC=0` reproduces today's failure on the same DFG |
| Statistics | `S2_STATS=1` reports how many muxes are shared slot-wise |

## Risk

Silent data corruption if the slot model is off by one: the mux would select the
wrong source in one of the two phases. Mitigation: validate against RTL values
(not just "mapping succeeded") on `dfg_scale`, `dfg_mixed`, `dfg_multi` before
attempting the 20-node case, and keep the feature behind `S2_SLOT_OCC`
(default off) until those pass.

## Out of scope

True modulo scheduling (II > 1 over loop iterations) stays out of scope for this
step: the RTL context counter is free-running over 4 contexts, so the current
model is "II = 4 contexts". MRRG-lite only fixes occupancy, and the design above
does not assume otherwise.
