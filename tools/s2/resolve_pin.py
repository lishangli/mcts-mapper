"""Resolve a mux source name to the block that owns it.

The mux dump prints `inst.port[idx=name]` where `inst` is a GIB **instance**.
Two namespaces look alike and cost a round each: GIB cfg blocks (what
`set_gib_bits` uses) and the instance ids in the netlist. Resolve by **wire
identity** instead: `pe_out` and `iob_out` are disjoint wire sets (510 vs 64,
overlap 0), so membership tells you the kind of producer.

usage:
  python resolve_pin.py 23 opinNW_0 [26 opinNW_0 ...]
  python resolve_pin.py --from-log <bridge.log>     # every hop of every path
"""
import json
import os
import re
import sys

NET = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'netlist',
                   'netlist_wires.json')


def load():
    IW = json.load(open(NET))
    pe, iob = {}, {}
    for blk, wires in IW['pe_out'].items():
        for w in wires:
            pe[tuple(w)] = blk
    for blk, wires in IW['iob_out'].items():
        for w in wires:
            iob[tuple(w)] = blk
    return pe, iob


def explain(pe, iob, inst, pin):
    w = (int(inst), pin)
    if w in pe:
        return 'PE %s' % pe[w]
    if w in iob:
        return 'IOB %s' % iob[w]
    return 'unknown (not a producer output)'


def main():
    pe, iob = load()
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == '--from-log':
        txt = open(args[1]).read()
        for line in txt.splitlines():
            if not line.startswith('path '):
                continue
            print(line.split(':')[0])
            for inst, port, ix, name in re.findall(
                    r'(\d+)\.(\w+)\[idx=([^=]+)=([^\]]+)\]', line):
                src = explain(pe, iob, inst, name)
                print('   %s.%s idx=%-4s %-12s -> %s' % (inst, port, ix, name, src))
        return 0
    for i in range(0, len(args) - 1, 2):
        print('%s.%s -> %s' % (args[i], args[i + 1],
                               explain(pe, iob, args[i], args[i + 1])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
