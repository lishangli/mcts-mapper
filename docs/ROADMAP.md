# Roadmap: from mapper experiments to a measured NN on the CGRA

Status: written 2026-11-08 after 55 experiment rounds in `docs/EXPERIMENTS.md`.
This file exists because those rounds were **tactical** (one hypothesis each) and
had no global plan; the last ~20 of them were spent inside a sub-problem that the
2026-09-26 decision had already ruled out.

## 1. The goal, and the two tracks

Goal: a **measured** high-performance neural-network kernel on the RISCV-CGRA
(bit-exact, reproducible with one command), plus the compiler machinery that gets
there.

There are exactly two tracks, and they are not alternatives:

* **Track A (spatial, this repo `tools/s2`)** — the mapper emits one
  configuration per kernel; performance comes from how well the DFG fits.
  **Status: closed for dense** (§3). It still provides the measurement
  infrastructure, the emitter, and the negative results.
* **Track B (spatiotemporal, `SPATIOTEMPORAL_RF_DESIGN.md` + the `fdra` hardware
  repo)** — restore the RF/II machinery that Tram was designed around, make **II**
  the objective, and run several contexts per configuration. This is where the
  remaining headroom is.

## 2. What is settled (do not re-derive)

| result | value | where |
|---|---|---|
| conv2d_3x3, bit-exact, reproducible | **18.1-19.8 ops/cycle** (195 cycles, 3528 ops) | `tools/conv/run.sh` |
| dense via per-sample DMA `nn4x2` | 1.57 / 2.24 / **2.91** ops/cycle at 64/128/256 samples | `docs/AG_FED_DENSE.md` |
| the law behind both | `ops/cycle = 2*I / (cycles per output word)`; dense I=4, conv **I=9 streams** | same |
| what conv really does | **nine `load_data` streams, one per tap** — no hardware window | round 53 |
| stream sharing | -27 % cycles on nn4x2 (889 -> 653), bit-exact | `docs/STATUS.md` |
| readback budget | ~1 KiB per launch; auto-clamped in the emitter | round 21 |
| strict-residue routing capacity | ~5 nodes; **fails at 12** (`nn3x2`) and at 17 (`dfg_dot9`) | rounds 44-55 |
| why it fails | **merge alignment**, not depth: every ADD needs both operands in one cycle, so a serial chain has as many merges as a tree | round 55 |
| emitter health | 0 mux conflicts in a full run; 0 silent double-booking | rounds 45-48 |

## 3. What is closed, and why that is a result

Spatial dense mapping at layer scale is **not achievable in this flow**, and the
evidence is now systematic rather than anecdotal:

* seven routing mitigations (early claim, hard-first order, four c0 bumps,
  delay-exact fallback, targeted rip-up of edges, of streams, read fan-out) — each
  moved the failing edge, none removed the failure;
* two placement strategies (leaf-producer moves, consumer relocation up to 10 hops
  with reachability filtering) — the repair fires and the node still cannot route;
* the failure is intrinsic: `hw latency` for a spatial mapping is dominated by
  configuration size and structure, not by mapping quality (2026-09-26 diagnosis).

**This is a publishable negative result**, and it is the reason Track B exists.

## 4. Track B plan (the actual work now)

Entry state (from `SPATIOTEMPORAL_RF_DESIGN.md`): rotating RF implemented and unit-tested
(5+2 tests), double-buffered config implemented, **RTL verified end-to-end**
(relu 8/8, 86 cycles), 15-DFG regression 9.067 vs SA 10.533 (15/15 wins).

**M1 entry state, verified 2026-11-08 (read-only; no build run).**

| fact | evidence | status |
|---|---|---|
| `cgra_rot_en = true` | `cgra-mg/src/main/scala/spec/TramSpec.scala:81` (comment: "enable rotating RF + config slot switching"); read through `dsa/template/CGRA.scala:142` | confirmed |
| dual-buffered config in RTL | `dsa/interconnect/Interconnect.scala:440` (`cfg_slot_wr` input), `:498-505` (`cfgA`/`cfgB` + enable), driven by `soc/CGRAController.scala:208` | confirmed |
| dual-slot config **in the compiler** | `cgra-compiler/src/mapper/io_scheduler.cpp`, HEAD `eb19a16` "io_scheduler: dual-slot config emission for spatio-temporal RTL + IOB mapping guard" | confirmed |
| rotating RF in the PE | `dsa/pe/PE.scala:158-191` (`rf.io.out(1)` "value produced k contexts ago"), config field `:332` | confirmed |
| unit tests | `cgra-mg/src/test/scala/dsa/RFTest.scala`; run with `sbt "project fdra; Test/runMain tram.dsa.RFRotatingTest"` and `... RFStaticTest` | present, unrun this round |
| hardware branch / HEAD | branch `spatio-temporal-compiler`, HEAD `eb19a16` | recorded |
| **relu RTL test** | no `relu` file anywhere under `fdra`; only referenced in `SPATIOTEMPORAL_RF_DESIGN.md` | **not in the tree — the 2026-09-26 "RELU PASS 8/8" is a record, not an artifact** |
| fdra worktree state | `git status` shows modified `app-compiler/*`, `benchmarks/test4/conv2d_3x3/cgra_call.txt` and others | **dirty: must be committed or stashed before any number can be called a baseline** |

Environment constraints (from `SPATIOTEMPORAL_RF_DESIGN.md` and the 2026-09-26 session, not
re-verified here): agent-side `make` trips safe-delete (>500 files) so builds must run
in the user's terminal; `TramSpec.loadSpec()` reads a stale `tram_spec.json` while `make`
uses `TramSpec.scala` — verification must use the latter; simulation needs
`+permissive +loadmem=<hex> +loadmem_addr=0x80000000 +permissive-off` with the hex
written as the whole objcopy raw image **byte-reversed**.

**M1 step 2 result (2026-11-08): the relu test was never committed.** `git log
--all --diff-filter=A -- '*relu*'`, `git log --all -S relu` and `git stash list` are all
empty; the branches are `spatio-temporal-compiler`, `spatio-temporal-rtl` and the
origin branches. So the 2026-09-26 run used an ad-hoc program that lived outside the
repo, and M1 must **put a test in the tree** rather than recover one.

**The ISA does support relu — in the uncommitted work.** `git diff` of
`cgra-mg/src/main/scala/op/Operations.scala` adds `MIN/MAX/CSHR/EQ/ULT` to the `OPC`
enumeration and changes `MIN/MAX` to compare with `asSInt` (signed), so
`relu(x) = MAX(x, 0)` is expressible. That fix exists **only in the working
tree**, which is the strongest argument for pinning it before any number is taken.

**Minimal RTL test plan (to be written into the tree next round):**

| item | value |
|---|---|
| kernel A (pipeline only) | `y = x + 1`: 1 ADD node, II irrelevant, exercises load -> PE -> store. With the constant stimulus x=1 the capture must be 2. |
| kernel B (relu) | `y = MAX(x, 0)` with signed compare; stimulus = ramp `x_j = j - 4` (so it goes negative), expected `0,0,0,0,0,1,2,3,...`. Needs the uncommitted `MAX`. |
| kernel C (cross-context) | `y_j = y_{j-1} + x_j` at II = 2, accumulator held in the rotating RF; with `x = 1` the capture must be `1,2,3,4,...`. **This is the only one of the three that exercises the spatio-temporal mechanism**, so it is the M1 gate for M3. |
| build | the official bare-metal flags from `software/tests` (`-DBAREMETAL=1 -nostdlib -static -T test.ld -march=rv64gc`), includes `riscv-tests/env` + `benchmarks/common` |
| simulate | `sims/verilator/simulator-chipyard-TramSoCConfig` with `+permissive +loadmem=<hex> +loadmem_addr=0x80000000 +permissive-off`; the hex is the whole objcopy raw image **byte-reversed** |
| pass | capture equals the expected sequence with 0 mismatches; note a chain of two config commands costs one extra cycle (measured 86 vs 85 in the 2026-09-26 relu run) |
| where | a new `benchmarks/test4/spatiotemporal/` directory **committed**, so this cannot be lost again |

**Worktree classification (236 dirty entries):**

* `cgra-mg/target/**` (≈120) and `benchmarks/test4/**` outputs (≈95) — **generated
  artifacts**, not work: they should be gitignored, and they are the reason a
  "clean tree" measurement is currently impossible.
* **19 real source files**: `app-compiler/llvm-pass/*` (CDFG pass), `cgra-compiler/*`
  (mapper, mapping, main, mapper_sa), `cgra-mg/src/main/scala/{dsa/interconnect/Interconnect.scala,
  dsa/template/CGRA.scala, op/Operations.scala}`, `cgra-mg/src/main/resources/operations.json`,
  `scripts/{run-verilator,setup-paths}.sh`. These are deliberate spatio-temporal
  work (the `MIN/MAX/CSHR/EQ/ULT` ops, signed compares, interconnect and template
  changes), **not leftovers** — but they are uncommitted, so HEAD `eb19a16` is *not*
  what the last verified run used.

**M1 step 3 result (2026-11-08): the installed binary IS the baseline, and it still computes correctly.** The simulator (`sims/verilator/simulator-chipyard-TramSoCConfig`, built
**2026-09-27 02:07**) post-dates every file under `cgra-mg/src/main/scala` (newest
01:07) and every compiled class, so it includes the spatio-temporal changes and the
uncommitted `MIN/MAX/CSHR/EQ/ULT` work. Running `tools/conv/run.sh` against it:

    CONV_PERF cycles=195
    conv2d_3x3: 42/42 bit-exact with the transposed tap order (C^T), 0 mismatch
    control: 0/42 with the row-major order (C)

So with `cgra_rot_en=true` and the dual-slot compiler, the datapath is still bit-exact
and the control still fails as it must. **The hardware side of M1 is done**: the
number to quote from now on is "conv2d_3x3, 42/42, 195 cycles, binary of 2026-09-27
02:07".

**M1 close-out checklist (run these in your own terminal; the agent cannot).**

```bash
# a. pin the baseline: ignore generated artifacts, commit the 19 real source files
cd /home/lsl/wafer-compiler-for-cgra/simulator/chipyard/generators/fdra
printf 'cgra-mg/target/\nsims/generated-src/\n' >> .gitignore
git status --porcelain | grep -v target/ | grep -v 'benchmarks/'
git add .gitignore app-compiler cgra-compiler cgra-mg/src cgra-mg/project \
        scripts sims/generated-src 2>/dev/null
git commit -m "spatio-temporal WIP: MIN/MAX/CSHR/EQ/ULT ops, signed compares, interconnect/template"

# b. the two RF unit tests. The header of RFTest.scala documents
#    "project fdra; ..." but build.sbt declares no such project (single-project
#    build), so try the documented form first and fall back to the bare one:
cd cgra-mg
sbt "project fdra; Test/runMain tram.dsa.RFRotatingTest" || \
  sbt "Test/runMain tram.dsa.RFRotatingTest"
sbt "project fdra; Test/runMain tram.dsa.RFStaticTest"  || \
  sbt "Test/runMain tram.dsa.RFStaticTest"
# expect 5 tests passed and 2 tests passed

# c. after any RTL change, prove the new binary was used before quoting numbers
ls -l --time-style=long-iso ../sims/verilator/simulator-chipyard-TramSoCConfig
/home/lsl/mapping/mcts-mapper/tools/conv/run.sh     # must still print 42/42, 195 cycles
```

If a step's output differs from the expectation above, stop and record it — a
baseline that cannot be reproduced is not a baseline.

M1 exit criteria therefore become concrete: (a) the worktree is pinned (commit or
stash), (b) both RF unit tests run and pass, (c) a relu-equivalent RTL test exists
**in the tree** and passes, (d) the 15-DFG regression numbers are reproduced with a
named command.

**M1 — freeze and re-verify the hardware baseline (1-2 steps).**
Confirm `cgra_rot_en` / slot state in `TramSpec.scala`, re-run the 15-DFG regression and
the relu RTL test, and record the exact build flags. Exit: numbers reproduce.

**M2 entry state, verified 2026-11-08 (read-only).**

| question | answer | evidence |
|---|---|---|
| Where is a mapping's cost computed today? | `MappingState.sum_reward`, accumulated **per route step** while building the mapping: `+1` per placement, `ROUTE_STEP_COST` per hop, `-registered_hop_penalty` per registered hop | `src/env/state.py:520, 704-715, 767-783` |
| How does MCTS compare candidates? | by that sum: `node.reward = state.get_reward() - cur_latency` | `src/agent/mcts.py:268-281` |
| Is there any schedule object? | **no**. The state holds a placement plus routes; nothing represents "iteration" or "period" | `src/env/state.py:511-520` |
| Does the hardware expose II? | `ii` / `loop_cycles` exist in the ISA and were annotated in 2026-09-26, and the compiler's `io_scheduler` already emits dual-slot configs | `cgra-mg/src/main/scala/soc/ISA.scala` (ii/loop_cycles block), `cgra-compiler/src/mapper/io_scheduler.cpp` |

**What "II as the objective" therefore means concretely.** The cost is per-route, so
II cannot enter it until the mapper produces something a period can be computed
from. The minimal, honest definition for this flow — streams are iterations, so a
DFG of N samples *is* a loop body executed N times:

    for every cycle c in the dependence graph (a path that returns to a node):
        II(c) = ceil( total_registered_delay(c) / distance(c) )      # Mardas
    II(mapping) = max over c of II(c), and 1 if there is no cycle

    cost(mapping) = II(mapping) * N + fill(mapping)   instead of  sum_reward

For the accumulator kernel `y_j = y_{j-1} + x_j` that gives II = 2 with one
registered hop in the loop, which is exactly the M3 target, and it is computable
**without** new hardware work.

First code step: `ii_of_mapping(placement, routes, dfg)` in this repo (pure Python,
no simulator), unit-tested on a hand-built 3-node accumulator (expect 2) and on
`dfg_dot9` (expect 1, no cycle). Only after that number exists does it make sense
to change `mcts.py`'s comparison — that is the "minimal change", in that order.

**M2 first code step done (2026-11-08): `src/agent/ii.py`.** `ii_of_mapping(dfg, weights)`
returns `(II, {cycle: II(cycle)}, truncated)`; the constant is `CYCLES_PER_CTX = 2`,
taken from the scheduler's own note that "only registered hops advance the
context (exe_ii = 2 cycles each)" (`tools/s2/mcts_bridge.py`, `path_slots`). Cycle
enumeration is capped at 20k partial paths and reports truncation.
Tests in `tests/test_ii.py`, all passing:

    accumulator II = 2      (y_j = y_{j-1} + x_j, one registered hop on the loop)
    dot9 II        = 1      (acyclic: a feed-forward kernel retires one sample/cycle)
    two-hop II     = 4
    weights        = {7: 1} (weights_from_routes counts registered hops)

**Open question this exposes:** every DFG in `tools/s2` today is acyclic, so
`ii_of_mapping` returns 1 for all of them — the format *can* hold a back edge, but
nothing in the flow produces one, and the mapper has never been asked to place a
graph with a recurrence. That is what M3 has to change first.

**M2 step 2 (2026-11-08): the DFG format accepts a recurrence; the mapper does not.**
`tools/s2/dfg_acc.json` (`y_j = y_{j-1} + x_j`, a **self-loop on the ADD**) parses
cleanly — the loader returns edge 1 as `(tail=1, head=1, operand=1)`. So the
format was never the obstacle. The mapper's DAG assumptions are, and they are
listed here so the next round does not rediscover them:

| place | what a self-loop does | verdict |
|---|---|---|
| `preds[e.head].append(e.tail)` (`mcts_bridge.py:888-890`) | `preds[1]` contains 1 | harmless by itself |
| `visit()` (`:947-951`) | adds `nid` to `visd` **before** recursing, so the self-edge returns immediately | **harmless** — no recursion limit hit |
| `ins = [e for e in EDGES if e.head == nid]` (`:965`) | the node's own output is one of its operands | **wrong schedule**: it is treated as an ordinary operand |
| `ctx_of[e.tail]` in the `want` computation (`:1001, 1020, 1042`) | `e.tail == nid`, whose `ctx_of` is only assigned *after* routing (`:1379`) | **crash** (`KeyError`) or a stale value |
| `base_c0 = max([ctx_of[e.tail] + hs[e.id] ...])` (`:1365`) | self-reference in the launch estimate | **crash** (`KeyError`) |
| `PEBLK_INV[placement[e.tail]]` in the routing loop (`:1042`) | routes from the node's own PE to itself | **fails or finds a nonsense path** |
| `settle`/`align_operands` equal-delay rule (`:1200-1330`) | demands the recurrence operand arrive in the *same* cycle | **wrong by construction**: it arrives one iteration earlier, i.e. `ctx - II` |

**Minimal fix (M3 step 1):** classify an edge with `tail == head` as a *recurrence*
edge: remove it from `ins` (so it is neither routed for alignment nor counted in
`base_c0`), give it a fixed delay of `II` (2 here) and the context
`ctx_of[nid] - II`, and let the emitter program the RF rotation for that operand
with `k = II / exe_ii = 1`. That is a ~30-line change confined to the node loop,
and it is the first time this flow would place a graph with a real recurrence.

**M3 step 1 result (2026-11-08): the recurrence-edge change is in, but it never runs —
the blocker is upstream.** The bridge now classifies `tail == head` edges into
`REC_EDGES` and drops them from `ins` (`mcts_bridge.py`, the node loop), but
`dfg_acc` fails long before that loop:

    stats: gave up after 12 attempts, failures {}
    MAPPING/EMISSION FAILED

`failures {}` is the tell: not one routing assertion ran, so the failure is in
**placement**, not in emission. `mapping.test_mapping()` (`src/mapper/mapping.py:374`)
returns `None` for this DFG — the MCTS asks the agent for a move, gets `-1`, and
breaks on the first iteration (`mapping.py:390-397`), leaving `latencies` empty.

So the first thing that cannot represent a loop is **the mapping environment's
action space**, not the bridge and not the DFG format. The next round has to answer
exactly one question — *why does the first action come back -1* — by checking
`MappingState.get_available_actions()` / `is_terminal()` on a 3-node cyclic DFG,
and whether the env builds its traversal from a topological order (which a cycle
does not have).

**M3 step 2 (2026-11-08): the action space is the cause, and the first fix is not
enough.** `tools/s2/probe_actions.py` makes the mechanism visible without a mapping run:

    dfg_acc    nodes=3  edges=3  terminal=False actions=32  ready=[0]
               after one move: terminal=True actions=0 ready=[0]
    dfg_dot9   nodes=36 edges=35 terminal=False actions=1440 ready=[0..17]
               after one move: terminal=False actions=1400 ready=[0..17]

So the ADD never becomes available: `MaskManager.init_mask` seeds placement with
nodes whose **incoming column is all zero** (`src/env/mask_manager.py:53-58`) and
`activate_ready_nodes` re-derives readiness the same way (`:126-135`), so a
self-edge keeps the ADD permanently unready and the mapping terminates after one
placement.

The minimal change — ignore the diagonal in both functions, since a recurrence is
satisfied by the node's own previous iteration — is now in, and **it did not change
the outcome**: the probe still shows `terminal=True actions=0` after one move, and
one `dfg_acc` mapping attempt still fails in 52 s with 0 recurrence prints. So the
self-edge exclusion is necessary but the readiness path that matters is elsewhere
(`MappingState.updateMeta` removes the placed node and extends with
`find_successors_nodes`, `src/env/state.py:284-295`, after `updateAdj` has already
rewritten the row to `-1`).

**Next round's single question:** for the ADD, print `dfg_adj[:, 1]`,
`dfg_features_vec[1, 6]` and `dfg_mask[1].sum()` before and after placing the
input, so the exact predicate that fails is named. No mapping run until then.

**M3 step 3 (2026-11-08): the action list was the missing piece, and the ADD is now
placeable.** The probe's four prints named it exactly:

    [initial] node 1 self-loop: adj[:,1]=[1,1,0] zeroed=[1,0,0] feat6=-1 dfg_mask=0   in_place_cur=False
    after one move:  terminal=True  actions=0   ready=[]        dfg_mask[1]=323
    [after]   node 1 self-loop: adj[:,1]=[-1,1,0] zeroed=[-1,0,0] dfg_mask=323 in_place_cur=False

So `activate_ready_nodes` **did** make the ADD ready (`dfg_mask = 323 > 0`) as soon as
the diagonal was ignored, but `place_cur_node` stayed empty: `get_actions()` iterates
`place_cur_node`, while `updateMeta` fills it from the successor row that `updateAdj`
has already rewritten to `-1`. One loop in `MappingState.updateAdj` now re-adds any
node whose `dfg_mask` became positive (`src/env/state.py`), and the probe shows:

    dfg_acc   after one move: terminal=False actions=128  ready=[1]
    dfg_dot9  after one move: terminal=False actions=1400 ready=[1..17]

The acyclic control is unchanged in count and now correctly drops the placed node
from `ready` (it used to keep it). **M3's next step is the first mapping attempt of
`dfg_acc`**, followed by a `dfg_quad` regression run.

**M2 — make II the objective (2-3 steps).**
The evaluator must score a schedule by `II` and `loop_cycles`, not by a single-pass
latency. Exit: a dense kernel reports II and cycles, and the trade-off is visible
in the numbers.

**M3 — modulo scheduling for one dense layer (3-4 steps).**
Take the 9-tap dot product (`tools/s2/dfg_dot9.json`, the kernel that Track A cannot
place) and schedule it with II = 2 across two contexts, using the rotating RF for
the accumulator. Exit: bit-exact value 45 in RTL, II reported.

**M4 — one real layer end-to-end (3-5 steps).**
A layer with I inputs and O outputs, fed by I streams (conv-style), from the same
data path the conv uses. Exit: bit-exact, ops/cycle reported next to the conv's
18.1 and the DMA-dense 2.9.

**M5 — write-up (2 steps).**
Two results and one method: the `2I` law with both kernels measured; the alignment
negative result (seven mitigations, two placement strategies); the spatiotemporal
restoration with II as the objective.

## 5. Rules for the loop from here

1. **Every round names its milestone** (M1..M5). A round that does not advance one
   is a documentation round and must say so.
2. **Two consecutive rounds on the same failure signature is a decision point**,
   not a third attempt: either a recorded design change or an explicit closure.
3. **No new mapper tweaks** for dense. Track A is a measurement harness now.
4. Every number quoted must be reproducible by one command in the repo.
