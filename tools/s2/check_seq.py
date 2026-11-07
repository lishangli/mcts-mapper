"""Align a ramp capture against the DFG reference and report the residual.

Constant streams cannot detect a timing error: if every sample carries the same
value, shifting *when* a value arrives changes nothing. With `S2_INPUT_RAMP=1`
stream i carries `base_i + j` at sample j, so the expected output sequence is
unique and any per-operand skew shows up as a **constant residual**:

    capture[i] = ref(i - L) + r

`L` is the pipeline latency (harmless: where the window starts) and `r` is the part
a pure shift cannot explain. `r != 0` means the operands of some node arrive from
different sample indices, i.e. the mapping is wrong for real data even though it
computes correctly on constant inputs.

usage: python check_seq.py <dfg.json> <run.log> [--min-run=N]
"""
import json
import re
import sys

OPS = {
    'ADD': lambda a, b: a + b,
    'SUB': lambda a, b: a - b,
    'MUL': lambda a, b: a * b,
    'PASS': lambda a, b: a,
}


def load_bases(dfg_path, log_path):
    """per-input ramp bases, from the generated driver (authoritative).

    The driver sits next to the run log as main_s2.c (perf.py) or <name>_main.c.
    Reading it matters: guessing bases 1,2,3 silently shifts the reference and
    turns a real +2 skew into a plausible-looking latency.
    """
    import os
    d = os.path.dirname(os.path.abspath(log_path))
    cands = [os.path.join(d, 'main_s2.c'),
             log_path.replace('_run.log', '_main.c')]
    txt = ''
    for c in cands:
        if os.path.exists(c):
            txt = open(c).read()
            break
    # stream -> (base, ramped?). S2_RAMP_ONE ramps a single stream, so the
    # pattern has to be read per array rather than assumed for all of them.
    bases = {}
    for i, v in re.findall(r'c(\d+)\[i\] = (\d+) \+ i', txt):
        bases[int(i)] = (int(v), True)
    for i, v in re.findall(r'c(\d+)\[i\] = (\d+);', txt):
        bases.setdefault(int(i), (int(v), False))
    if bases:
        return bases
    dfg = json.load(open(dfg_path))
    return {k: (1, True) for k, _ in enumerate(
        o for o in dfg['objects'] if o['opcode'].upper() == 'INPUT')}


def reference(dfg_path, bases, samples):
    dfg = json.load(open(dfg_path))
    objs = {o['_gvid']: o for o in dfg['objects']}
    ins = {o['_gvid']: [] for o in dfg['objects']}
    for e in dfg['edges']:
        ins[e['head']].append((e['tail'], e['operand']))
    insorted = sorted(o['_gvid'] for o in dfg['objects']
                      if o['opcode'].upper() == 'INPUT')
    base_of = {n: bases.get(k, (1, True)) for k, n in enumerate(insorted)}

    def value(nid, j):
        o = objs[nid]
        op = o['opcode'].upper()
        if op == 'INPUT':
            b, ramped = base_of.get(nid, (1, True))
            return b + (j if ramped else 0)
        if op == 'CONST':
            return int(o['value'], 0)
        args = sorted(ins[nid], key=lambda t: t[1])
        a = value(args[0][0], j)
        b = value(args[1][0], j) if len(args) > 1 else 0
        return OPS[op](a, b) & 0xFFFFFFFF

    outs = [o['_gvid'] for o in dfg['objects'] if o['opcode'].upper() == 'OUTPUT']
    return [[value(ins[o][0][0], j) for j in range(samples)] for o in outs]


def captured(path):
    t = open(path).read()
    arr = {}
    for k, i, v in re.findall(r'O(\d+)\[(\d+)\] = (-?\d+)', t):
        arr.setdefault(int(k), {})[int(i)] = int(v)
    return [[arr[k][i] for i in sorted(arr[k])] for k in sorted(arr)]


def align(ref, got):
    """(longest constant-residual run, latency, residual, start).

    On a linear ramp every latency gives a constant residual, so the run length
    cannot choose between them; the smallest |residual| is the best pure-shift
    explanation, and a non-zero one is what a per-operand skew looks like.
    """
    scored = []
    for L in range(0, max(1, min(len(ref), len(got))) + 1):
        run, r0, start = 0, None, None
        bestrun = (0, None, None)
        for i in range(L, min(len(got), L + len(ref))):
            r = got[i] - ref[i - L]
            if r0 is None or r == r0:
                if r0 is None:
                    r0, start = r, i
                run += 1
                if run > bestrun[0]:
                    bestrun = (run, r0, start)
            else:
                r0, start, run = r, i, 1
        scored.append((bestrun[0], L, bestrun[1], bestrun[2]))
    if not scored:
        return (0, None, None, None)
    top = max(s[0] for s in scored)
    cands = [s for s in scored if s[0] == top]
    cands.sort(key=lambda s: abs(s[2]))
    return cands[0]


def main():
    argv = sys.argv[1:]
    min_run = 24
    for a in argv:
        if a.startswith('--min-run='):
            min_run = int(a.split('=')[1])
    args = [a for a in argv if not a.startswith('--')]
    if len(args) < 2:
        print(__doc__)
        return 2
    dfg, log = args[0], args[1]
    caps = captured(log)
    bases = load_bases(dfg, log)
    refs = reference(dfg, bases, 512)
    bad = 0
    for k, got in enumerate(caps):
        if k >= len(refs):
            continue
        run, L, r, start = align(refs[k], got)
        ok = run >= min_run and r == 0
        bad += 0 if ok else 1
        print('channel %d: run=%d latency=%d residual=%d (from idx %s) -> %s'
              % (k, run, L, r, start, 'OK' if ok else 'SKEW'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
