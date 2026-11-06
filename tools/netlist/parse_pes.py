"""Parse GPE modules: operand mux tables (Muxn) -> netlist_pes.json"""
import re, json, os
V = ('/home/lsl/wafer-compiler-for-cgra/simulator/chipyard/sims/verilator/'
     'generated-src/chipyard.TestHarness.TramSoCConfig/'
     'chipyard.TestHarness.TramSoCConfig.top.v')
OUT = os.path.dirname(os.path.abspath(__file__))
txt = open(V, errors='replace').read()
result = {}
for mm in re.finditer(r'module (GPE(?:_\d+)*)\(', txt):
    name = mm.group(1)
    j = txt.index('endmodule', mm.end())
    body = txt[mm.end():j]
    cm = re.search(r"cfgMatch = \d+'h([0-9a-fA-F]+) == io_cfg_addr\[11:3\]", body)
    if not cm:
        continue
    blk = int(cm.group(1), 16)
    dst_of, cfg_of, ins_of = {}, {}, {}
    for line in body.split('\n'):
        lm = re.match(r'\s*assign (\w+) = (Muxn\w*_io_out)\b', line)
        if lm:
            dst_of[lm.group(2)] = lm.group(1)
            continue
        cm2 = re.match(r"\s*assign (Muxn\w*)_io_config = (?:isAccOp \? 3'h\d : )?cfgOut\[(\d+):(\d+)\]", line)
        if cm2:
            cfg_of[cm2.group(1)] = (int(cm2.group(2)), int(cm2.group(3)))
            continue
        im = re.match(r'\s*assign (Muxn\w*)_io_in_(\d+) = ([\w\[\]:]+)', line)
        if im:
            ins_of.setdefault(im.group(1), {})[int(im.group(2))] = im.group(3)
            continue
    muxes = {}
    for sig, dst in dst_of.items():
        mn = sig.split('_io_out')[0]
        d = ins_of.get(mn, {})
        n = (max(d) + 1) if d else 0
        muxes[dst] = {'cfg': cfg_of.get(mn), 'srcs': [d.get(i) for i in range(n)]}
    inst = int(name.split('_')[1]) if '_' in name else 0
    result[inst] = {'blk': blk, 'module': name, 'mux': muxes}
json.dump(result, open(OUT + '/netlist_pes.json', 'w'), indent=1)
print('parsed', len(result), 'GPEs')
p = result.get(120, {})
for dst, info in p.get('mux', {}).items():
    print('pes_120', dst, 'cfg', info['cfg'], 'srcs', info['srcs'])
