"""Initiation interval of a mapping.

The spatio-temporal machine advances a *context* every `exe_ii` cycles (2, see
`path_slots` in tools/s2/mcts_bridge.py: "only registered hops advance the
context (exe_ii = 2 cycles each)"), and the rotating register file lets a value
produced in context c be read in context c+k. A dependence cycle that needs k
registered hops therefore costs k contexts, i.e. 2k cycles per iteration.

For each dependence cycle c:

    II(c) = CYCLES_PER_CTX * ceil(total_registered_delay(c) / distance(c))

and the mapping's interval is the maximum over cycles (1 when the graph is
acyclic, i.e. a pure feed-forward kernel retires one sample per cycle).

This is the Mardas bound with the machine's context model folded into the
constant. It exists so that a schedule can be *scored* by II at all: today the
mapper's objective is a per-route sum (`src/env/state.py`, `sum_reward`) which
cannot express a period.

Cycle enumeration is capped (`max_cycles`, default 20k partial paths) because the
number of simple cycles is exponential; a truncated enumeration returns the best
II found so far and says so via `truncated`.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Tuple

CYCLES_PER_CTX = 2

def _adjacency(dfg: dict) -> Dict[int, List[Tuple[int, int]]]:
    """node -> [(succ, edge_id)] for DFG edges."""
    adj: Dict[int, List[Tuple[int, int]]] = {}
    for e in dfg.get("edges", []):
        adj.setdefault(int(e["tail"]), []).append((int(e["head"]), int(e["_gvid"])))
    return adj

def ii_of_mapping(dfg: dict, weights: Dict[int, int],
                  cycles_per_ctx: int = CYCLES_PER_CTX,
                  max_cycles: int = 20000) -> Tuple[int, Dict[Tuple[int, ...], int], bool]:
    """(II, {cycle: II(cycle)}, truncated).

    `weights` maps DFG edge id -> registered hop count (0 for a purely
    combinational route). A cycle's `distance` is the number of iterations it
    spans, which is 1 for every edge this flow produces: a stream item is one
    iteration and no route carries a value across more than one iteration
    boundary. Pass an explicit per-edge distance through `weights` only if that
    ever changes.
    """
    adj = _adjacency(dfg)
    nodes = sorted({int(o["_gvid"]) for o in dfg.get("objects", [])})
    per_cycle: Dict[Tuple[int, ...], int] = {}
    truncated = False
    budget = [max_cycles]

    for start in nodes:
        # DFS for cycles whose smallest node is `start` (each cycle found once).
        # `on_path` prevents walking a node twice, which otherwise loops forever
        # on any 2-cycle and burns the whole budget without finding a cycle.
        stack: List[Tuple[int, List[int], List[int]]] = [(start, [], [start])]
        while stack:
            node, eids, on_path = stack.pop()
            if budget[0] <= 0:
                truncated = True
                break
            budget[0] -= 1
            for succ, eid in adj.get(node, ()):
                if succ < start:
                    continue           # would be found from a smaller start
                if succ == start and eids:
                    cyc = tuple(eids + [eid])
                    if len(set(cyc)) != len(cyc):
                        continue       # not a simple cycle
                    delay = sum(int(weights.get(e, 0)) for e in cyc)
                    per_cycle[cyc] = cycles_per_ctx * int(math.ceil(delay / 1.0))
                elif succ > start and succ not in on_path:
                    stack.append((succ, eids + [eid], on_path + [succ]))
        if truncated:
            break

    if not per_cycle:
        return 1, {}, truncated
    return max(per_cycle.values()), per_cycle, truncated

def weights_from_routes(routes: Dict[int, Iterable], is_reg: Callable[[Hashable, Hashable], bool]) -> Dict[int, int]:
    """Registered hop count per edge, from `tools/s2`-shaped paths.

    A path entry is (gib, dstport, idx); the hop is registered when the *wire it
    drives* is registered (`is_reg(gib, dstport)`), which is the same predicate
    the scheduler uses for delay.
    """
    out: Dict[int, int] = {}
    for eid, path in routes.items():
        n = 0
        for ent in path:
            if len(ent) >= 2 and is_reg(ent[0], ent[1]):
                n += 1
        out[int(eid)] = n
    return out
