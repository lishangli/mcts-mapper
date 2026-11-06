"""S2 stress driver: generate layered DAGs, run them end to end, tabulate capacity.

Each entry is one generated DAG -> run_dfg.sh (bridge -> riscv build -> RTL sim)
-> parse the artefacts -> compare against the generator's own evaluator.

Runs are strictly serial: run_dfg.sh rewrites main_s2.c / cgra_execute.c and the
tag-prefixed files in place, so two runs would clobber each other.

`run_dfg.sh` pipes the bridge through `tee`, so its exit code says nothing about
routing success; a config only counts as PASS when the captured steady values
match the evaluator AND their trailing run is long enough.

usage:
  python stress.py                     # built-in sweep (about 15-20 min)
  python stress.py --only l4w5o3c3      # single config by tag prefix
  python stress.py --report none        # skip stress_report.md
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_dag
from regression import steady, bridge_hint, REMAP_BUDGET   # noqa: E402

CIN_RE = re.compile(r'cin: (\d+) rows \(A=(\d+) B=(\d+)\), iob_ens=0x([0-9a-fA-F]+)'
                    r', captures=\[([0-9, ]*)\]')
STATS_RE = re.compile(r'stats: attempt (\d+) ok, occ cfg=(\d+) rd=(\d+) peak=(\d+)')
FAIL_RE = re.compile(r'emission failed, retrying: (.*)')

# (tag, layers, width, outputs, consts, shape). IOB banks = captures + const
# streams, so the sweep walks both terms of that sum: outputs add captures one
# by one, consts add load streams, while shape/width/depth grow routing pressure
# independently. Chain keeps ids adjacent (placement-friendly); layered is the
# wide cross-layer case that stresses the mod-4 phase constraint.
SWEEP = [
    ('chain12o2c3', 12, 1, 2, 3, 'chain'),   # 12 stages, 2 captures
    ('chain16o3c3', 16, 1, 3, 3, 'chain'),   # 16 stages, 3 captures
    ('l3w4o1c3', 3, 4, 1, 3, 'layered'),     # 12 compute mesh, 1 output: reference
    ('l3w5o2c0', 3, 5, 2, 0, 'layered'),     # no inline const: leaves from streams
    ('l3w5o2c6', 3, 5, 2, 6, 'layered'),     # more const leaves -> IOB pressure
]

# Documented as unreachable within the 12x remap budget (see stress_report.md):
#   ('chain20o3c3', 20, 1, 3, 3, 'chain')    -> 10/12 attempts fail, no parity route
#   ('l4w5o3c3', 4, 5, 3, 3, 'layered')      -> 10/12 attempts fail, no parity route
# Run one of them with: python gen_dag.py --shape chain --layers 20 --width 1 \
#   --outputs 3 --consts 3 --name chain20o3c3 && ./run_dfg.sh dfg_dag_chain20o3c3.json chain20o3c3


def make_dfg(tag, layers, width, outputs, consts, seed=0, shape='layered'):
    """Write dfg_dag_<tag>.json (+ _expect.json) and return its metadata."""
    dfg = gen_dag.build_dag(layers, width, outputs, consts, seed, shape=shape)
    dfg['name'] = tag
    expect = gen_dag.evaluate(dfg)
    meta = gen_dag.summarize(dfg, expect, layers=layers, width=width,
                             outputs=outputs, consts=consts, seed=seed,
                             shape=shape)
    meta['dfg'] = 'dfg_dag_%s.json' % tag
    with open(os.path.join(HERE, meta['dfg']), 'w') as f:
        json.dump(dfg, f, indent=1)
    with open(os.path.join(HERE, 'dfg_dag_%s_expect.json' % tag), 'w') as f:
        json.dump(meta, f, indent=1)
    return meta


def _read(path):
    if not os.path.exists(path):
        return ''
    with open(path, errors='replace') as f:
        return f.read()


def mux_capacity():
    """Total GIB mux outputs in the netlist: the real routing capacity, as
    opposed to the 32 IOB banks which only bound streams + captures."""
    try:
        import sched as s2sched
        if not hasattr(s2sched, 'GIBS'):
            return None, None
        g = s2sched.GIBS
        return sum(len(x['mux']) for x in g.values()), len(g)
    except Exception:
        return None, None


def read_steady(tag):
    """steady() of the run log; a failed bridge means the log never appears"""
    path = os.path.join(HERE, tag + '_run.log')
    return steady(path) if os.path.exists(path) else {}


def collect(tag, meta, timeout):
    """Run one config and return a result row."""
    t0 = time.time()
    cmd = [os.path.join(HERE, 'run_dfg.sh'), meta['dfg'], tag]
    try:
        p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                           timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = 'timeout'
    secs = time.time() - t0

    blog = _read(os.path.join(HERE, tag + '_bridge.log'))
    rlog = _read(os.path.join(HERE, tag + '_run.log'))

    cin = cap = streams = iob_bits = None
    m = CIN_RE.search(blog)
    if m:
        cin, cap, iob_bits = int(m.group(1)), m.group(5), int(m.group(4), 16)
        cap = len([x for x in cap.split(',') if x.strip()])
    streams = len(re.findall(r'^const val=', blog, re.M))
    remap = len(FAIL_RE.findall(blog))
    reasons = FAIL_RE.findall(blog)
    st = STATS_RE.search(blog)
    occ = 'cfg=%s rd=%s peak=%s' % (st.group(2), st.group(3), st.group(4)) if st else '-'
    occ_peak = int(st.group(4)) if st else None

    got = read_steady(tag)
    values = [got.get(i, (None, 0))[0] for i in range(len(meta['expect']))]
    runs = [got.get(i, (None, 0))[1] for i in range(len(meta['expect']))]
    ok = len(runs) == len(meta['expect']) and all(
        v == e and r >= 8 for v, r, e in zip(values, runs, meta['expect']))

    bits = bin(iob_bits).count('1') if iob_bits is not None else None
    banks_ok = (bits is None or cap is None or bits == cap + streams)

    row = {
        'tag': tag, 'obj': meta['objects'], 'compute': meta['compute_nodes'],
        'edges': meta['edges'], 'cin': cin, 'iob': bits, 'streams': streams,
        'caps': cap, 'remap': remap, 'occ': occ, 'occ_peak': occ_peak,
        'expect': meta['expect'],
        'got': values, 'runs': runs, 'secs': secs, 'rc': rc,
        'ok': ok and banks_ok, 'banks_ok': banks_ok,
        'why': (reasons[-1] if reasons else '') if not ok else '',
        'hint': '' if ok else bridge_hint(tag),
    }
    return row


def fmt(row):
    def show(x):
        return '-' if x is None else x
    return ('%-12s obj=%-3d cmp=%-3d cin=%-4s iob=%-3s strm=%-2s caps=%-2s '
            'remap=%-2d %-22s expect=%-16s got=%-16s %6.1fs %s'
            % (row['tag'], row['obj'], row['compute'], show(row['cin']),
               show(row['iob']), row['streams'], show(row['caps']), row['remap'],
               row['occ'], row['expect'], row['got'], row['secs'],
               'PASS' if row['ok'] else 'FAIL'))


def write_report(rows, path, capacity=None, gibs=None):
    head = ('| tag | objects | compute | edges | cin rows | IOB banks | streams | '
            'captures | remap | occ peak | expect | got | time | result |\n'
            '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')
    body = ''
    for r in rows:
        body += ('| %s | %d | %d | %d | %s | %s | %d | %s | %d | %s | %s | %s | %.1fs | %s |\n'
                 % (r['tag'], r['obj'], r['compute'], r['edges'], r['cin'], r['iob'],
                    r['streams'], r['caps'], r['remap'], r['occ_peak'], r['expect'],
                    r['got'], r['secs'], 'PASS' if r['ok'] else '**FAIL**'))
    fails = [r for r in rows if not r['ok']]
    if fails:
        notes = '\n## 失败配置\n\n'
        for r in fails:
            notes += ('- `%s`: remap=%d, 原因 `%s`%s\n'
                      % (r['tag'], r['remap'], r['why'] or r['hint'],
                         '' if r['banks_ok'] else '（iob_ens 位数与 captures+streams 不符）'))
    else:
        notes = '\n## 失败配置\n\n本次扫描范围内无失败配置。\n'

    peaks = [r['occ_peak'] for r in rows if r['occ_peak']]
    usage = ('- 布线容量：%d 个 GIB mux 输出（%d GIB），最高实测 occ peak=%d（%.0f%%）\n'
             % (capacity, gibs, max(peaks), 100.0 * max(peaks) / capacity)
             if capacity and peaks else '')
    with open(path, 'w') as f:
        f.write('# S2 大规模 DFG 压力扫描\n\n')
        f.write('生成器 `gen_dag.py`（独立求值器给期望值），驱动 `stress.py`，'
                '判定复用 `regression.steady()` + `bridge_hint()`。\n\n')
        f.write('## 资源上限\n\n')
        if capacity:
            f.write('- IOB bank：32 个，约束 `popcount(iob_ens) == captures + 常量流数`\n')
            f.write('- 布线容量：%d 个 GIB mux 输出（%d 个 GIB）\n' % (capacity, gibs))
        if usage:
            f.write(usage)
        f.write('\n## 指标\n\n')
        f.write(head + body)
        f.write(notes)
        f.write('\n## 复现\n\n```bash\ncd /home/lsl/mapping/cgra-sw/s2\n'
                'python stress.py                 # 全部配置\n'
                'python stress.py --only chain    # 单个配置\n```\n')
    return path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    only = argv[argv.index('--only') + 1] if '--only' in argv else None
    report = (argv[argv.index('--report') + 1] if '--report' in argv
              else os.path.join(HERE, 'stress_report.md'))
    timeout = (int(argv[argv.index('--timeout') + 1]) if '--timeout' in argv else 1200)

    cfgs = [c for c in SWEEP if only is None or only in c[0]]
    if not cfgs:
        print('no config matches --only %r; known: %s'
              % (only, ', '.join(c[0] for c in SWEEP)))
        return 2

    capacity, gibs = mux_capacity()
    if capacity:
        print('capacity: %d GIB mux outputs over %d GIBs, 32 IOB banks'
              % (capacity, gibs), flush=True)

    rows = []
    for tag, layers, width, outputs, consts, shape in cfgs:
        meta = make_dfg(tag, layers, width, outputs, consts, shape=shape)
        print('%s: %d objects (%d compute), %d edges, expect %s'
              % (tag, meta['objects'], meta['compute_nodes'], meta['edges'],
                 meta['expect']), flush=True)
        row = collect(tag, meta, timeout)
        rows.append(row)
        print(fmt(row), flush=True)
        if not row['ok']:
            print('    why: %s' % (row['why'] or row['hint']), flush=True)
        if not row['banks_ok']:
            print('    WARNING: popcount(iob_ens)=%s != captures(%s)+streams(%s)'
                  % (row['iob'], row['caps'], row['streams']), flush=True)

    print('\n%d/%d PASS' % (sum(1 for r in rows if r['ok']), len(rows)))
    print('elapsed %.1fs' % sum(r['secs'] for r in rows))
    if report.lower() != 'none':
        print('report: %s' % write_report(rows, report, capacity, gibs))
    return 0 if all(r['ok'] for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
