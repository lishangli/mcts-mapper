# S2 — spatio-temporal CGRA mapping/verification toolchain

MCTS placement → netlist-authoritative routing → dual-slot `cin` emission → RTL run
for the spatio-temporal (rotating-RF, dual-config-slot) Tram CGRA.

This directory is the canonical copy; it used to live outside version control
(`~/mapping/cgra-sw/s2`). Paths are resolved relative to this file so the tree can be
cloned anywhere; override with the environment variables listed below.

## Files

| File | Role |
|---|---|
| `mcts_bridge.py` | The bridge: MCTS placement → per-operand routing (`route_operand`) → schedule (ctx / slot / `c0`) → dual-slot config streams → `cgra_execute.c` + `main_s2.c` |
| `sched.py` | Netlist resources (`GIBS` / `WIRES` / `PES` / `PEBLK` / `REGS`), `Router` (reverse BFS on the `top.v` netlist), `path_delay()` (registered-otrack hops) |
| `gen_dag.py` | Layered/chain DFG generator + an **independent** evaluator, so a passing RTL run is compared against a value computed by different code |
| `regression.py` | End-to-end regression (bridge → riscv build → RTL → steady-state judge), 9 cases |
| `stress.py` | Serial driver for large generated DFGs; writes `stress_report.md` |
| `run_dfg.sh` | One DFG end to end: bridge, `riscv64-unknown-elf-gcc` build, Verilator run, value extraction |
| `dfg_*.json` | Test DFGs; `*_expect.json` carries the generator's expected output channels |

## Quick start

```bash
# one DFG end to end (bridge log, run log, extracted steady values)
./run_dfg.sh dfg_branch.json branch

# full regression (≈20 min, cases run serially because artefacts are rewritten in place)
python regression.py

# one case only
python regression.py --only branch

# generate a new DFG + expected values
python gen_dag.py --shape chain --layers 12 --outputs 2 --consts 3 --name chain12o2c3
```

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `S2_MAPPER_SRC` | `<repo>/src` | MCTS mapper package root (parser/mapper/agent) |
| `S2_NETLIST` | `<repo>/tools/netlist` | netlist tables extracted from `top.v` |
| `S2_OUT_DIR` | this directory | where `cgra_execute.c` / `main_s2.c` / logs / ELF go |
| `FDRA` | reference machine path | FDRA tree holding `software/tests` (RISC-V bare-metal) |
| `FDRA_SIM` | `$FDRA/../../sims/verilator` | Verilator build directory |
| `RISCV_TOOLCHAIN` | reference machine path | `riscv64-unknown-elf-gcc-8.3.0` prefix |
| `S2_OCC_STRICT` | `0` | `1` = exclusive mux ownership (pre-P0 semantics), for A/B runs |
| `S2_PACK` | `0` | `1` = fold independent ops onto one PE (time-domain folding) |
| `S2_PHASE_BUMP` | `4` | how many `c0` bump levels a node may try when parity fails |
| `S2_STATS` | `0` | per-attempt occupancy / failure counters |

## Status (2026-11-07)

* Regression `python regression.py` runs **12 cases**; in the last full batch run
  10 of them passed together, and after the routing changes all 12 were re-run
  individually and pass (see `docs/STATUS.md`). The cases include `dfg_big` (10 compute PEs,
  6 IOB streams), a 12-stage chain with two capture channels, two dense layers
  (`nn3x2`, `nn4x2`) and a **20-node layered DAG**.
* Capacity: the previous 12-node wall was an artefact of the mod-4 delivery
  constraint. Routes are emitted into both config banks and single-operator PEs
  are duplicated across slots, so the constraint is unnecessary in the normal
  flow; with `S2_NO_PHASE=1` (now the default, `S2_PACK` excepted) the 20-node
  `l4w5o3c3` maps on the first attempt and captures `[6, 6, 4]`. The phase model
  still applies to time-folded PEs and RF reads.
* Every run prints `S2_PERF exec=… store=… total=…`; `perf.py` tabulates them and the
  compute/execution/performance model lives in `docs/EXECUTION_MODEL.md`.
* `gen_nn.py` turns a dense layer into a DFG; `--inputs/--outputs` size it, and the
  independent evaluator writes the values the RTL must reproduce.
