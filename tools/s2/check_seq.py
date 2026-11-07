"""Compare a captured sequence against the DFG's per-sample reference.

Constant-input kernels cannot catch a timing change: if every sample carries the
same value, shifting when a value arrives changes nothing. This checker is for
runs made with `S2_INPUT_RAMP=1`, where stream i carries `v_i + j` at sample j,
so the expected output sequence is unique and a misaligned mapping shows up.

usage: python check_seq.py <dfg.json> <run.log>
"""
import json
import re
import sys

INPUT_VALUES = {'%const1': 3, '%const2': 5, 'CONST1': 3, 'CONST2': 5}
INPUT_VALUES.update({'%%x%d' % i: i + 1 for i in range(8)})   # matches the bridge
OPS = {
    'ADD': lambda a, b: a + b,
    'SUB': lambda a, b: a - b,
    'MUL': lambda a, b: a * b,
    'PASS': lambda a, b: a,
}


def reference(dfg_path, samples):
    dfg = json.load(open(dfg_path))
    objs = {o['_gvid']: o for o in dfg['objects']}
    ins = {o['_gvid']: [] for o in dfg['objects']}
    for e in dfg['edges']:
        ins[e['head']].append((e['tail'], e['operand']))

    def value(nid, j):
        o = objs[nid]
        op = o['opcode'].upper()
        if op == 'INPUT':
            return INPUT_VALUES.get(o.get('ref_name'), 0) + j
        if op == 'CONST':
            return int(o['value'], 0)
        args = sorted(ins[nid], key=lambda t: t[1])
        a = value(args[0][0], j)
        b = value(args[1][0], j) if len(args) > 1 else 0
        return OPS[op](a, b) & 0xFFFFFFFF

    outs = [o['_gvid'] for o in dfg['objects'] if o['opcode'].upper() == 'OUTPUT']
    return outs, [[value(ins[o][0][0], j) for j in range(samples)] for o in outs]


def captured(path):
    t = open(path).read()
    arr = {}
    for k, i, v in re.findall(r'O(\d+)\[(\d+)\] = (-?\d+)', t):
        arr.setdefault(int(k), {})[int(i)] = int(v)
    return [ [arr[k][i] for i in sorted(arr[k])] for k in sorted(arr) ]


def best_offset(exp, got):
    """smallest mismatch count over a shift of the reference against the capture.

    The capture starts when the AG window starts, which is not sample 0 of the
    reference, so the offset is searched on both sides.
    """
    best = (10 ** 9, None)
    for d in range(0, min(len(exp) - 8, 512)):
        n = min(len(exp) - d, len(got))
        m = sum(1 for j in range(n) if got[j] != exp[d + j])
        if m < best[0]:
            best = (m, d)
    return best


def main():
    dfg, log = sys.argv[1], sys.argv[2]
    caps = captured(log)
    refs = None
    any_bad = False
    for k, got in enumerate(caps):
        nz = [i for i, v in enumerate(got) if v != 0]
        if not nz:
            print('channel %d: all zero' % k)
            continue
        first = nz[0]
        if refs is None:
            refs = reference(dfg, max(1, len(got) - first))
        exp = refs[k] if k < len(refs) else None
        if exp is None:
            continue
        tail = got[first:]
        bad, off = best_offset(exp, tail)
        ok = bad == 0 and len(exp) > 8
        any_bad |= not ok
        print('channel %d: %d samples, best offset %s, %d mismatches -> %s'
              % (k, len(tail), off, bad, 'OK' if ok else 'MISMATCH'))
    return 1 if any_bad else 0


if __name__ == '__main__':
    sys.exit(main())
