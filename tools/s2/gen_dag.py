"""S2 stress-data generator: layered DAG in the DFG JSON schema + an independent
evaluator that says what each output channel must settle to.

The evaluator deliberately shares no code with mcts_bridge.py: if both the
generator and the bridge were wrong in the same way, the comparison would pass
and prove nothing.

Layer 0 holds the leaves (two INPUT streams whose values the bridge fixes at
3 and 5, plus C CONST nodes carrying their own value). Layers 1..L hold W
compute nodes each; a node's operands come from strictly earlier layers, or
from an earlier node of the same layer, so `_gvid` order stays a topological
order and every output taps the last layer. OUTPUT nodes get increasing _gvid,
which is the order the bridge emits O0..On in (caps is built from
`sorted(nid for nid in node_by_id if op == 'OUTPUT')`).

usage:
  python gen_dag.py --layers 4 --width 5 --outputs 3 --consts 3 --name l4w5o3
  python gen_dag.py --layers 3 --width 4 --outputs 2 --consts 2 --seed 7
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IO_FIELDS = {'pattern': '4, 91', 'size': '256', 'offset': '0, 0'}

# must mirror mcts_bridge.CONST_VALUES for INPUT nodes; CONST nodes carry their
# own value instead, so a stress DAG never needs the bridge table extended
INPUT_VALUES = {'%const1': 3, '%const2': 5, 'CONST1': 3, 'CONST2': 5}


def build_dag(layers=4, width=5, outputs=3, consts=3, seed=0,
              ops=('ADD', 'SUB', 'PASS'), maxfanout=2, shape='layered'):
    """Return a DFG dict: objects/edges with contiguous _gvid, operand in {0,1}.

    ops excludes MUL by default: a layered multiply explodes past 32 bits after
    a few levels and the sweep would then measure arithmetic overflow instead of
    routing capacity. Pass --ops ADD,SUB,MUL,PASS to opt in.
    """
    if layers < 1 or width < 1:
        raise ValueError('layers/width must be >= 1')
    tap = layers * width if shape == 'chain' else width
    if outputs < 1 or outputs > tap:
        raise ValueError('outputs must be in 1..%d for shape=%s' % (tap, shape))
    if consts < 0:
        raise ValueError('consts must be >= 0')
    if not ops:
        raise ValueError('ops must not be empty')

    rnd = random.Random(seed)
    objs, edges = [], []

    def add_node(**kw):
        kw['_gvid'] = len(objs)
        objs.append(kw)
        return kw['_gvid']

    def add_edge(tail, head, operand):
        edges.append({'_gvid': len(edges), 'tail': tail, 'head': head,
                      'operand': operand})

    # ---- layer 0: leaves
    leaves = [
        add_node(opcode='INPUT', name='CONST1', ref_name='%const1', **IO_FIELDS),
        add_node(opcode='INPUT', name='CONST2', ref_name='%const2', **IO_FIELDS),
    ]
    for i in range(consts):
        v = 1 + (i % 9)
        leaves.append(add_node(opcode='CONST', name='%%c%d_%d' % (v, i), value=str(v)))

    # ---- compute layers
    hist = [list(leaves)]
    prev = leaves
    fanout = {}

    def pick(cands, avoid=()):
        """random source among the least-wired candidates.

        A uniform random pick concentrates edges on a few hot leaves (a CONST
        reached from 8 PEs needs 8 routes from one PE output), which measures
        fanout congestion rather than graph scale. Real DFGs are SSA-ish, so
        prefer a source that is still under maxfanout and only fall back to a
        wired one when everything is saturated.
        """
        fresh = [n for n in cands if n not in avoid and fanout.get(n, 0) < maxfanout]
        pool_ = fresh or [n for n in cands if n not in avoid] or list(cands)
        n = pool_[rnd.randrange(len(pool_))]
        fanout[n] = fanout.get(n, 0) + 1
        return n

    if shape == 'chain':
        # single-file pipeline: each stage takes the previous stage as operand 0
        # and the stage before that as operand 1, so every node fans out exactly
        # twice and the leaves are only touched at the head. Ids stay adjacent,
        # which isolates "how many stages" from "how wide/far".
        cur = []
        for i in range(layers * width):
            h = add_node(opcode=ops[i % len(ops)], name='c%d' % i)
            add_edge(cur[i - 1] if i else leaves[1], h, 0)
            add_edge(cur[i - 2] if i >= 2 else leaves[(i + 2) % len(leaves)], h, 1)
            cur.append(h)
        hist, prev = [cur], cur
    else:
        for layer in range(1, layers + 1):
            cur = []
            older = [n for grp in hist for n in grp]  # everything before this layer
            for j in range(width):
                op = ops[(layer - 1 + j) % len(ops)]
                # 1 in 4 nodes reaches into an older layer (long cross-layer
                # edge); the rest take the previous layer, which keeps most
                # routes one hop away - long random edges everywhere make the
                # mod-4 phase constraint unsatisfiable long before the
                # interconnect runs out of wires.
                cand = older if (older and j % 4 == 1) else prev
                if j and j % 3 == 0:
                    s0 = cur[j - 1]                  # same-layer edge, keep it short
                    fanout[s0] = fanout.get(s0, 0) + 1
                else:
                    s0 = pick(cand)
                s1 = pick(cand, avoid=(s0,))
                h = add_node(opcode=op, name='n%d_%d' % (layer, j))
                add_edge(s0, h, 0)
                add_edge(s1, h, 1)
                cur.append(h)
            hist.append(cur)
            prev = cur

    # ---- outputs tap the last layer
    for k in range(outputs):
        out = add_node(opcode='OUTPUT', name='OUT%d' % k,
                       ref_name='%%out%d' % k, **IO_FIELDS)
        add_edge(prev[width - outputs + k], out, 0)

    return {'name': 'dag_l%dw%do%dc%d_s%d' % (layers, width, outputs, consts, seed),
            'directed': True, 'objects': objs, 'edges': edges}


def evaluate(dfg):
    """Expected steady value of every output, in OUTPUT-_gvid order."""
    val, src = {}, {}
    for e in dfg['edges']:
        src.setdefault(e['head'], {})[e['operand']] = e['tail']
    order = sorted(dfg['objects'], key=lambda o: o['_gvid'])
    for o in order:
        gid, op = o['_gvid'], o['opcode']
        if op == 'INPUT':
            val[gid] = INPUT_VALUES.get(o.get('ref_name'),
                                        INPUT_VALUES.get(o.get('name'), 0))
        elif op == 'CONST':
            val[gid] = int(str(o.get('value', '0')), 0)
        elif op == 'OUTPUT':
            val[gid] = val.get(src.get(gid, {}).get(0), 0)   # capture passes through
        else:
            a = val.get(src.get(gid, {}).get(0), 0)
            b = val.get(src.get(gid, {}).get(1), 0)
            if op == 'ADD':
                val[gid] = a + b
            elif op == 'SUB':
                val[gid] = a - b
            elif op == 'MUL':
                val[gid] = a * b
            elif op == 'PASS':
                val[gid] = a          # the hardware PASSes operand 0 only
            else:
                raise ValueError('unsupported opcode %r' % op)
    return [val[o['_gvid']] for o in order if o['opcode'] == 'OUTPUT']


def summarize(dfg, expect, **params):
    ops = [o['opcode'] for o in dfg['objects']]
    meta = {
        'name': dfg['name'],
        'params': params,
        'objects': len(dfg['objects']),
        'edges': len(dfg['edges']),
        'compute_nodes': sum(1 for o in ops if o not in ('INPUT', 'CONST', 'OUTPUT')),
        'inputs': sum(1 for o in ops if o == 'INPUT'),
        'consts': sum(1 for o in ops if o == 'CONST'),
        'outputs': sum(1 for o in ops if o == 'OUTPUT'),
        'expect': list(expect),
    }
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--width', type=int, default=5)
    ap.add_argument('--outputs', type=int, default=3)
    ap.add_argument('--consts', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ops', default='ADD,SUB,PASS')
    ap.add_argument('--maxfanout', type=int, default=2,
                    help='soft cap on how many consumers a node feeds')
    ap.add_argument('--shape', choices=('layered', 'chain'), default='layered',
                    help='layered: L x W mesh; chain: single-file pipeline')
    ap.add_argument('--name', default=None, help='file tag; defaults to the DAG name')
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args(argv)

    ops = tuple(x.strip().upper() for x in args.ops.split(',') if x.strip())
    dfg = build_dag(args.layers, args.width, args.outputs, args.consts,
                    args.seed, ops, args.maxfanout, args.shape)
    tag = args.name or dfg['name']
    expect = evaluate(dfg)
    meta = summarize(dfg, expect, layers=args.layers, width=args.width,
                     outputs=args.outputs, consts=args.consts, seed=args.seed,
                     ops=','.join(ops), maxfanout=args.maxfanout, shape=args.shape)
    meta['dfg'] = os.path.basename('dfg_dag_%s.json' % tag)

    dfg_path = os.path.join(args.out, 'dfg_dag_%s.json' % tag)
    exp_path = os.path.join(args.out, 'dfg_dag_%s_expect.json' % tag)
    with open(dfg_path, 'w') as f:
        json.dump(dfg, f, indent=1)
    with open(exp_path, 'w') as f:
        json.dump(meta, f, indent=1)

    print('%s: %d objects (%d compute), %d edges -> expect %s'
          % (tag, meta['objects'], meta['compute_nodes'], meta['edges'], expect))
    if any(abs(v) >= 2 ** 31 for v in expect):
        print('WARNING: expected value exceeds int32, the RTL printout will wrap',
              file=sys.stderr)
    print('wrote %s' % dfg_path)
    print('wrote %s' % exp_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
