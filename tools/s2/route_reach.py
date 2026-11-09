"""Which registered-hop counts are reachable between two PEs, ignoring occupancy.

A route's cycle delay is its number of registered hops, and the context engine
needs a specific residue mod 4. If the residue is not reachable at all, no router
can help that placement and the fault is in placement, not routing.

usage:
  python route_reach.py <src_pe> <dst_pe> [max_hops]
"""
import json
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sched import Router, INST2BLK, is_reg          # noqa: E402

NET = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'netlist',
                   'netlist_wires.json')
IW = json.load(open(NET))
R = Router()


def reach(src, dst, max_hops=12):
    """{hops: {ports that reach src's output at that hop count}}"""
    src_pins = {tuple(w) for w in IW['pe_out'][str(src)]}
    out = {}
    for port in sorted(IW['pe_in'][str(dst)], key=int):
        w = IW['pe_in'][str(dst)][port]
        if not w:
            continue
        start = tuple(w)
        q = deque([(start, 0)])
        seen = {(start, 0)}
        while q:
            (g, p), d = q.popleft()
            if (g, p) in src_pins:
                out.setdefault(d, set()).add(int(port))
                continue
            if d >= max_hops:
                continue
            nd = d + (1 if is_reg(g, p) else 0)
            if nd > max_hops:
                continue
            for kind, gib, dstport, idx, s in (R.preds(g, p) or ()):
                # a track hop yields the source wire itself; a mux hop yields the
                # name of one of this GIB's input pins
                nxt = tuple(s) if isinstance(s, (list, tuple)) else (gib, s)
                if (nxt, nd) in seen:
                    continue
                seen.add((nxt, nd))
                q.append((nxt, nd))
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = int(sys.argv[1]), int(sys.argv[2])
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    res = reach(src, dst, mx)
    if not res:
        print('PE %d -> PE %d: NO route at all within %d hops' % (src, dst, mx))
        return 1
    print('PE %d -> PE %d: reachable registered-hop counts -> pins' % (src, dst))
    for d in sorted(res):
        print('  hops=%-3d residue=%d  pins=%s' % (d, d % 4, sorted(res[d])))
    print('residues: %s' % sorted({d % 4 for d in res}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
