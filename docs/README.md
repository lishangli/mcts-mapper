# Documentation index

| Document | What it answers |
|---|---|
| [STATUS.md](STATUS.md) | What is proven on the RTL, what is open, what was retracted, how to reproduce |
| [EXECUTION_MODEL.md](EXECUTION_MODEL.md) | Compute model (OPC semantics, ctx/slot multiplexing, rotating RF), execution model (route delay, alignment rule), performance model with measured constants |
| [NN_WORKLOADS.md](NN_WORKLOADS.md) | Verified neural-network kernels and their throughput (conv 18.1–19.8 ops/cycle, dense layers) |
| [RELATED_WORK.md](RELATED_WORK.md) | How CGRA-ME / Pillars / OpenCGRA / CCF / CGRA-Flow / HyCUBE map and simulate, and what this flow borrows |
| [MRRG_PLAN.md](MRRG_PLAN.md) | The time-expanded routing design, its implementation status, and the capacity boundary |
| [../SPATIOTEMPORAL_SCHEDULER_PLAN.md](../SPATIOTEMPORAL_SCHEDULER_PLAN.md) | Running design log of the spatio-temporal scheduler (S0–S2) |

Tooling:

| Path | Purpose |
|---|---|
| `tools/s2/` | MCTS placement → netlist routing → dual-slot cin → RTL; 12-case regression, perf sweep, DFG generators |
| `tools/netlist/` | Netlist tables extracted from `top.v` (GIB muxes, wires, PE operand muxes, registered ports) |
| `tools/conv/` | conv2d_3x3 timed driver and `run.sh` (build, run, verify, print throughput) |
