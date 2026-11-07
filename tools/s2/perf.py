"""Measure one DFG kernel end to end and compare with its expected values.

Runs tools/s2/run_dfg.sh for each case, then reports the hardware cost the RTL
printed (`S2_PERF exec=<cycles> store=<cycles>`, emitted by the generated
cgra_execute.c) next to the steady-state capture value and the DFG's expected
value. This is the measurement side of the execution model documented in
docs/EXECUTION_MODEL.md.

usage: python perf.py [--only SUBSTR] [--repeat N] [--json OUT]
"""
import json
import os
import re
import subprocess
import sys
import time

S2 = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('S2_OUT_DIR') or S2      # where run_dfg.sh writes artefacts
sys.path.insert(0, S2)
from regression import CASES, steady, bridge_hint   # noqa: E402  (same list)

PERF_RE = re.compile(
    r'S2_PERF load=(\d+) exec=(\d+) store=(\d+) total=(\d+) caps=(\d+)'
    r'(?: it=(\d+))?')


def opt(args, name, default=None):
    if name not in args:
        return default
    i = args.index(name)
    if i + 1 >= len(args):
        raise SystemExit('%s needs a value' % name)
    return args[i + 1]


def measure(tag):
    """Cycle counters from a finished run log, or None.

    load  = load_cfg + input DMA
    exec  = the execute() instruction (returns before the array drains)
    store = capture DMA issue, 512 B per channel
    total = load + exec + store + fence, i.e. the kernel's wall-clock cost
    caps  = number of capture channels, the store's byte count is 512 * caps
    """
    path = os.path.join(OUT, tag + '_run.log')
    if not os.path.exists(path):
        return None
    m = PERF_RE.search(open(path, errors='replace').read())
    if not m:
        return None
    keys = ('load', 'exec', 'store', 'total', 'caps', 'it')
    return {k: int(v) if v is not None else 1
            for k, v in zip(keys, m.groups())}


def case_from_path(path):
    """(--dfg) expected values come from the companion *_expect.json, if any."""
    path = os.path.abspath(path)
    exp_path = path[:-5] + '_expect.json' if path.endswith('.json') else None
    expect = []
    if exp_path and os.path.exists(exp_path):
        expect = json.load(open(exp_path))['expect']
    return os.path.basename(path), expect


def main():
    args = sys.argv[1:]
    only = opt(args, '--only')
    dfg_arg = opt(args, '--dfg')
    rep = int(opt(args, '--repeat', 1))
    out = opt(args, '--json', os.path.join(OUT, 'perf_latest.json'))
    if dfg_arg:
        cases = [case_from_path(dfg_arg)]
    else:
        cases = [c for c in CASES if only is None or only in c[0]]
    if not cases:
        print('no case matches --only %r' % only)
        return 2
    rows, ok, t0 = [], 0, time.time()
    for dfg, expect in cases:
        for k in range(rep):
            tag = 'pf_%s_%d' % (os.path.splitext(dfg)[0], k)
            # A placement can emit a program that never returns (measured: one
            # in ~6 dfg_quad runs with operand alignment on). Without a timeout
            # that is a silent hang, so bound it and remap instead of waiting.
            tries = int(os.environ.get('S2_SIM_RETRY', '3'))
            tmo = int(os.environ.get('S2_SIM_TIMEOUT', '300'))
            p = None
            for t_ in range(tries):
                try:
                    p = subprocess.run([os.path.join(S2, 'run_dfg.sh'), dfg, tag],
                                       capture_output=True, text=True, cwd=S2,
                                       timeout=tmo)
                    break
                except subprocess.TimeoutExpired:
                    print('%-28s   -> simulator produced no result in %ds, remapping (%d/%d)'
                          % (dfg, tmo, t_ + 1, tries))
                    for f in (os.path.join(OUT, tag + '_run.log'),):
                        if os.path.exists(f):
                            os.remove(f)
                    p = None
            if p is None:
                p = subprocess.CompletedProcess([], 1, '', 'timeout')
            perf = measure(tag) or {}
            log = os.path.join(OUT, tag + '_run.log')
            # a mapping failure aborts run_dfg.sh before the simulator runs, so
            # there is no run log: report the bridge's reason instead of raising
            st = steady(log) if os.path.exists(log) else {}
            got = [st.get(i, (None, 0))[0] for i in range(len(expect))]
            good = p.returncode == 0 and got == expect
            ok += good
            rows.append(dict(perf, case=dfg, got=got, expect=expect, pass_=good))
            print('%-28s load=%-4s exec=%-4s store=%-4s total=%-5s caps=%-2s '
                  'got=%-14s expect=%-14s %s'
                  % (dfg, perf.get('load'), perf.get('exec'),
                     perf.get('store'), perf.get('total'), perf.get('caps'),
                     got, expect, 'PASS' if good else 'FAIL'))
            if not good and not os.path.exists(os.path.join(OUT, tag + '_run.log')):
                print('%-28s   -> %s' % ('', bridge_hint(tag)))
    print('\n%d/%d PASS  (%.0fs)' % (ok, len(rows), time.time() - t0))
    json.dump({'rows': rows, 'elapsed_s': round(time.time() - t0, 1)},
              open(out, 'w'), indent=1)
    return 0 if ok == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
