"""Summarise why a mapping attempt failed, from a bridge log.

The bridge prints one 'emission failed, retrying: ...' line per attempt and,
with S2_STATS=1, an occupancy line per attempt. This turns those into a table:
which edges die, with which failure kind, and whether the placement was the
cause (no free pin), the routing (congestion) or the netlist (unroutable).

usage: python failure_report.py <tag_bridge.log> [--csv out.csv]
"""
import json
import re
import sys
from collections import Counter

RE_FAIL = re.compile(r'emission failed, retrying: (.*)')
RE_EDGE = re.compile(r'(no (?:parity )?route) (\d+)->(\d+) op(\d+)')
RE_NEED = re.compile(r'\{.*\}')


def parse(path):
    attempts = []
    for line in open(path, errors='replace'):
        m = RE_FAIL.search(line)
        if not m:
            continue
        msg = m.group(1).strip()
        rec = {'msg': msg, 'kind': 'other', 'edge': None, 'why': None}
        em = RE_EDGE.search(msg)
        if em:
            rec['kind'] = em.group(1)
            rec['edge'] = '%s->%s op%s' % (em.group(2), em.group(3), em.group(4))
        need = RE_NEED.search(msg)
        if need:
            try:
                rec['why'] = json.loads(need.group(0).replace("'", '"'))
            except ValueError:
                rec['why'] = need.group(0)
            if isinstance(rec['why'], dict) and rec['why'].get('kind'):
                rec['kind'] += ' / ' + rec['why']['kind']
        attempts.append(rec)
    return attempts


def main():
    path = sys.argv[1]
    attempts = parse(path)
    if not attempts:
        print('no failed attempts in %s' % path)
        return 0
    kinds = Counter(a['kind'] for a in attempts)
    edges = Counter(a['edge'] for a in attempts if a['edge'])
    print('%d failed attempts' % len(attempts))
    for k, n in kinds.most_common():
        print('  %-34s %d' % (k, n))
    if edges:
        print('repeated dead edges (attempts):')
        for e, n in edges.most_common(8):
            print('  %-16s %d' % (e, n))
    pins = [a['why'] for a in attempts
            if isinstance(a['why'], dict) and a['why'].get('free_pins') == []]
    if pins:
        print('%d attempts died with no spare pin on the destination PE' % len(pins))
    return 0


if __name__ == '__main__':
    sys.exit(main())
