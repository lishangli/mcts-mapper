import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.ii import ii_of_mapping, weights_from_routes, CYCLES_PER_CTX  # noqa: E402

def dfg_of(nodes, edges):
    return {"objects": [{"_gvid": i, "opcode": "X"} for i in nodes],
            "edges": [{"_gvid": g, "tail": t, "head": h, "operand": 0}
                      for g, (t, h) in enumerate(edges)]}

def test_accumulator_one_reg_hop():
    # y_j = y_{j-1} + x_j : nodes 0(x) 1(y) 2(add); edges 0->2, 1->2, 2->1(back edge)
    dfg = dfg_of([0, 1, 2], [(0, 2), (1, 2), (2, 1)])
    w = {0: 0, 1: 1, 2: 0}      # the back edge 2->1 carries one registered hop
    ii, per, trunc = ii_of_mapping(dfg, w)
    assert ii == CYCLES_PER_CTX, (ii, per, trunc)
    assert not trunc
    return ii

def test_dot9_acyclic():
    d = json.load(open(os.path.join(os.path.dirname(__file__), "..", "tools", "s2", "dfg_dot9.json")))
    w = {int(e["_gvid"]): 0 for e in d["edges"]}
    ii, per, trunc = ii_of_mapping(d, w)
    assert ii == 1 and per == {}, (ii, per)
    return ii

def test_two_reg_hops():
    dfg = dfg_of([0, 1], [(0, 1), (1, 0)])
    w = {0: 1, 1: 1}            # two registered hops around the loop
    ii, per, trunc = ii_of_mapping(dfg, w)
    assert ii == 2 * CYCLES_PER_CTX, (ii, per)
    return ii

def test_weights_from_routes():
    routes = {7: [("g1", "otrackE_0", 3), ("g2", "itrackW_0", None)]}
    reg = {"otrackE_0": True, "itrackW_0": False}
    w = weights_from_routes(routes, lambda g, p: reg[p])
    assert w == {7: 1}, w
    return w

if __name__ == "__main__":
    print("accumulator II =", test_accumulator_one_reg_hop())
    print("dot9 II        =", test_dot9_acyclic())
    print("two-hop II     =", test_two_reg_hops())
    print("weights       =", test_weights_from_routes())
    print("ALL PASS")
