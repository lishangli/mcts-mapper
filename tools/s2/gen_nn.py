"""Generate a neural-network layer (dot product + bias) as a DFG.

A layer of `outputs` neurons over `inputs` activation streams is

    y_o = sum_i w[o][i] * x_i + b[o]

which maps to `inputs` MUL nodes + (`inputs`-1) ADD nodes + 1 bias ADD per
neuron. Activations come in as INPUT streams (IOB constant streams in the
current bridge), weights and biases are CONST nodes that the bridge inlines
into the PE that maps them, so a layer needs no DRAM traffic for the weights.

The expected values are computed here, independently of mcts_bridge.py, and
written to <name>_expect.json for tools/s2/perf.py.

usage:
  python gen_nn.py --inputs 3 --outputs 2 --name nn3x2
  python gen_nn.py --inputs 4 --outputs 1 --weights 2,1,3,1 --bias 1
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IO_FIELDS = {'pattern': '4, 91', 'size': '256', 'offset': '0, 0'}
# x_i streams carry i+1; mcts_bridge.CONST_VALUES has the same table
INPUT_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]


def build(inputs=3, outputs=2, seed=0, name='nn'):
    objs, edges = [], []

    def node(**kw):
        kw['_gvid'] = len(objs)
        objs.append(kw)
        return kw['_gvid']

    def edge(tail, head, operand):
        edges.append({'_gvid': len(edges), 'tail': tail, 'head': head,
                      'operand': operand})

    x = [node(opcode='INPUT', name='x%d' % i, ref_name='%%x%d' % i, **IO_FIELDS)
         for i in range(inputs)]

    rnd = random.Random(seed)
    w = [[rnd.randint(1, 3) for _ in range(inputs)] for _ in range(outputs)]
    b = [rnd.randint(-2, 2) for _ in range(outputs)]

    outs = []
    for o in range(outputs):
        prods = []
        for i in range(inputs):
            wn = node(opcode='CONST', name='%%w%d_%d' % (o, i), value=str(w[o][i]))
            p = node(opcode='MUL', name='m%d_%d' % (o, i))
            edge(x[i], p, 0)
            edge(wn, p, 1)
            prods.append(p)
        acc = prods[0]
        for p in prods[1:]:
            a = node(opcode='ADD', name='s%d_%d' % (o, len(prods)))
            edge(acc, a, 0)
            edge(p, a, 1)
            acc = a
        bn = node(opcode='CONST', name='%%b%d' % o, value=str(b[o]))
        y = node(opcode='ADD', name='y%d' % o)
        edge(acc, y, 0)
        edge(bn, y, 1)
        out = node(opcode='OUTPUT', name='OUT%d' % o, ref_name='%%out%d' % o,
                   **IO_FIELDS)
        edge(y, out, 0)
        outs.append(out)

    expect = [sum(w[o][i] * INPUT_VALUES[i] for i in range(inputs)) + b[o]
              for o in range(outputs)]
    dfg = {'name': name, 'directed': True, 'objects': objs, 'edges': edges}
    meta = {'name': name,
            'params': {'inputs': inputs, 'outputs': outputs, 'seed': seed},
            'weights': w, 'bias': b, 'input_values': INPUT_VALUES[:inputs],
            'compute_nodes': outputs * (2 * inputs),
            'expect': expect,
            'dfg': name + '.json'}
    return dfg, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=int, default=3)
    ap.add_argument('--outputs', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--name', default=None)
    args = ap.parse_args()
    if not 1 <= args.inputs <= len(INPUT_VALUES):
        raise SystemExit('inputs must be in 1..%d' % len(INPUT_VALUES))
    name = args.name or 'nn%dx%d' % (args.inputs, args.outputs)
    dfg, meta = build(args.inputs, args.outputs, args.seed, name)
    json.dump(dfg, open(os.path.join(HERE, name + '.json'), 'w'), indent=1)
    json.dump(meta, open(os.path.join(HERE, name + '_expect.json'), 'w'), indent=1)
    print('%s: %d compute nodes, y = %s' % (name, meta['compute_nodes'], meta['expect']))


if __name__ == '__main__':
    sys.exit(main())
