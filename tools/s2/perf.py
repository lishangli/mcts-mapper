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
import signal
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
            # Whole-case wall-clock cap: S2_MAP_TIMEOUT * S2_SIM_RETRY can add up
            # to hours, which is how a single round gets eaten by one DFG.
            case_t0 = time.time()
            case_tmo = int(os.environ.get('S2_CASE_TIMEOUT', '1800'))
            for t_ in range(tries):
                if time.time() - case_t0 > case_tmo:
                    print('%-28s   -> case budget %ds exhausted, giving up'
                          % (dfg, case_tmo))
                    break
                # start_new_session + killpg: subprocess.run(timeout=) kills only
                # the shell, so the python child (mcts_bridge, or verilator) keeps
                # running. Two such orphans were burning a core each for over two
                # hours and every later run looked "hung" because of them.
                proc = subprocess.Popen(
                    [os.path.join(S2, 'run_dfg.sh'), dfg, tag],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    cwd=S2, start_new_session=True)
                # Two separate budgets. Mapping legitimately takes minutes on
                # larger graphs, so the simulator budget must start only once the
                # ELF exists; otherwise the guard kills healthy mappings (that is
                # what happened to dfg_fan8).
                map_tmo = int(os.environ.get('S2_MAP_TIMEOUT', '900'))
                elf = os.path.join(OUT, tag + '.riscv')
                t0_, t_elf = time.time(), None
                while proc.poll() is None:
                    now = time.time()
                    if t_elf is None and os.path.exists(elf):
                        t_elf = now
                    if t_elf is None and now - t0_ > map_tmo:
                        break
                    if t_elf is not None and now - t_elf > tmo:
                        break
                    time.sleep(2)
                over = 'mapping' if t_elf is None else 'simulator'
                try:
                    out_, err_ = proc.communicate(timeout=30)
                    p = subprocess.CompletedProcess(proc.args, proc.returncode, out_, err_)
                    if proc.returncode == 0:
                        break
                except subprocess.TimeoutExpired:
                    pass
                if True:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except OSError:
                        proc.kill()
                    proc.communicate()
                    print('%-28s   -> %s produced no result in time, remapping (%d/%d)'
                          % (dfg, over, t_ + 1, tries))
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
