"""Why does the first MCTS move come back -1 on a cyclic DFG?

Prints the initial placement action set for a DFG, and the same after one move, so
a cycle's effect on the environment's action space is visible without a mapping run.

usage:  python probe_actions.py <dfg.json> [<dfg2.json> ...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))

from parser import ADGIR, DFGParser, Operations
from mapper.mapping import Mapping
from env.state import MappingState


def probe(path):
    dfg = DFGParser(path).getDFG()
    adg = ADGIR(os.path.join(REPO, "example", "cgra_adg.json")).getADG()
    ops = Operations()
    ops.OpParser(os.path.join(REPO, "example", "operations.json"))
    m = Mapping(dfg, adg, ops, False)
    m.env.reset()
    st = MappingState(m.env)
    acts = st.get_actions()
    print("%-18s nodes=%-3d edges=%-3d terminal=%-5s actions=%d ready=%s"
          % (os.path.basename(path), len(m.env.dfg.getNodes()),
             len(m.env.dfg.getEdges()), st.is_terminal(), len(acts),
             sorted(m.env.place_cur_node)))
    nodes = m.env.dfg.getNodes()
    node_by_id = nodes if isinstance(nodes, dict) else {n.id: n for n in nodes}
    for a in acts[:5]:
        nid = m.env.get_dfg_node_id(a)
        print("    action %d -> dfg node %d (%s) on adg %d"
              % (a, nid, node_by_id[nid].getOpCode(), m.env.get_adg_node_id(a)))
    # --- why is a self-loop node never placeable? -------------------------
    import numpy as np
    rec = [n.id for n in m.env.dfg.getNodes()
           if any(e.tail == n.id and e.head == n.id for e in m.env.dfg.getEdges())]

    def diag(tag):
        for nid in rec:
            col = m.env.dfg_adj[:, nid]
            z = col.copy(); z[nid] = 0
            print("    [%s] node %d self-loop: adj[:,%d]=%s zeroed=%s feat6=%s "
                  "dfg_mask=%s mask=%s in_place_cur=%s"
                  % (tag, nid, nid, [int(x) for x in col],
                     [int(x) for x in z], m.env.dfg_features_vec[nid, 6],
                     float(m.env.dfg_mask[nid].sum()), float(st.mask[nid].sum()),
                     nid in m.env.place_cur_node))
    diag("initial")
    if acts:
        ok = m.env.update(acts[0], st.reward, False)
        print("    env.update(action=%d) -> %s, mapping_state=%s"
              % (acts[0], ok, st.mapping_state))
        st.take_action(action=acts[0], print_action=False)
        print("    after one move: terminal=%-5s actions=%d ready=%s mapping_state=%s"
              % (st.is_terminal(), len(st.get_actions()),
                 sorted(m.env.place_cur_node), st.mapping_state))
        diag("after")
    return st, m, acts


if __name__ == "__main__":
    for p in (sys.argv[1:] or [os.path.join(HERE, "dfg_acc.json")]):
        probe(p)
