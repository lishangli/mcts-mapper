# Related work: how other open CGRA toolchains map and execute

Purpose: place the Tram/S2 flow in context, and borrow what is missing rather than
reinvent it. Sources are linked at the bottom.

## 1. Taxonomy we belong to

Open-CGRA survey (arXiv 2508.19090) splits dataflow accelerators by how much time
multiplexing they use:

* **Spatial** — every DFG node keeps a PE for the whole kernel; a DFG that does not
  fit is cut into subgraphs and executed one after another.
* **Spatio-temporal** — nodes are placed in *space and time*; PEs and interconnect
  are reconfigured every cycle, so the configuration memory has to hold II contexts.

Tram with `num_ctx = 4`, dual-slot PEs and the rotating RF is a small
spatio-temporal machine. Before the RTL change documented in
`SPATIOTEMPORAL_RF_DESIGN.md` it was effectively spatial, which is why mapping
quality could not move the measured latency.

## 2. What the mature frameworks do

| Framework | Mapping | Simulation | Limitation relevant to us |
|---|---|---|---|
| CGRA-ME | spatio-temporal, ILP/modulo scheduling, user-defined architecture | RTL only, no open simulator | no detailed memory model; simple kernels |
| Pillars | Scala architecture description, uses CGRA-ME frontend | cycle-accurate simulator | inherits CGRA-ME limits |
| OpenCGRA / CCF | supports control divergence and recurrences | user-written testbenches, no automatic data generation | mappers are not architecture-adaptive |
| CGRA-Flow | MLIR frontend, ILP-based mapping, VectorCGRA | integrated exploration + RTL | heavyweight (docker, GUI, 25 GB RAM) |
| Morpher / HyCUBE | modulo scheduling over an **MRRG**, Dijkstra routing, II = MII then increment | RTL + analytical models | design-specific |

The recurring technical ideas worth adopting:

1. **II as a first-class quantity.** `MII = max(ResMII, RecMII)`; schedule at MII
   and raise II until feasible. Our emitter has no II: it assigns one context per
   node inside a fixed 4-phase rotation, so the kernel runs once through the array
   and the whole pipeline is "II = 4 contexts" by construction.
2. **MRRG (modulo routing resource graph).** Replicate PEs, registers and wires
   for each time step of the II window, then place and route on that graph. Our
   equivalent is the ad-hoc `c0` / `ctx` assignment plus `route_parity`; it works
   for 12-node DFGs and fails on 20 nodes precisely because there is no global
   space-time resource graph to negotiate over.
3. **Oversubscription + rip-up.** Temporarily allow port/mux overuse, then resolve
   with congestion penalties (SPR-style), annealing, or a learned policy (LISA).
   Today we rerun MCTS from scratch up to 12 times, which the logs show is
   information-free (the same edges fail each round).
4. **Multi-hop links as schedulable resources.** HyCUBE shows that if combinational
   multi-hop paths are modelled as resources inside the MRRG, II drops (3 → 2 on
   their example). Our `path_delay` already measures registered hops; the missing
   half is *choosing* multi-hop paths as part of scheduling rather than as a
   post-hoc repair.

## 3. Where our flow is ahead of the pack

* **Netlist-authoritative routing.** Configurations are derived from the post-synthesis
  `top.v` (153 GIBs, 1668 mux outputs, per-port register flags), not from an
  idealized architecture description. Four independent anchor configs and every
  RTL regression match bit for bit, which is why silent mis-mappings (e.g. the
  three address spaces) could be found at all.
* **Value-level end-to-end checking.** Every emitted configuration is executed on
  RTL and judged on captured values, not on estimated latency.

## 4. Concrete borrow-list for the next iterations

| Borrow | Target in this repo |
|---|---|
| MRRG-style time-expanded resource graph | replace the per-edge `route_parity` repair with a graph over `(resource, ctx)` |
| II iteration (MII → raise) | add II to the emitter; the RTL already has `cgra_num_ctx = 4` and `exe_ii = 2` |
| Congestion-driven rip-up & reroute | replace the 12× remap with negotiated congestion |
| Port-group aware placement | MCTS reward term over operand pins per PE |

## 5. Sources

* Open CGRA ecosystem survey: https://arxiv.org/html/2508.19090
* CGRA-Flow: https://github.com/coredac/CGRA-Flow
* CGRA-ME: https://github.com/eddiehung/CGRA-ME
* OpenCGRA: https://github.com/pnnl/OpenCGRA
