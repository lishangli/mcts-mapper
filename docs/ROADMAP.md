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

M1 exit criteria therefore become concrete: (a) the worktree is pinned (commit or
stash), (b) both RF unit tests run and pass, (c) a relu-equivalent RTL test exists
**in the tree** and passes, (d) the 15-DFG regression numbers are reproduced with a
named command.

**M1 — freeze and re-verify the hardware baseline (1-2 steps).**
Confirm `cgra_rot_en` / slot state in `TramSpec.scala`, re-run the 15-DFG regression and
the relu RTL test, and record the exact build flags. Exit: numbers reproduce.

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
