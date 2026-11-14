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
    if acts:
        st.take_action(action=acts[0], print_action=False)
        print("    after one move: terminal=%-5s actions=%d ready=%s"
              % (st.is_terminal(), len(st.get_actions()),
                 sorted(m.env.place_cur_node)))
    return st, m, acts


if __name__ == "__main__":
    for p in (sys.argv[1:] or [os.path.join(HERE, "dfg_acc.json")]):
        probe(p)
