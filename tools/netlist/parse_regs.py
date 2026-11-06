"""Extract clocked vs combinational GIB output ports from top.v -> netlist_regs.json
Delay model: a route hop costs 1 cycle iff its destination port is a registered
otrack (Interconnect.scala 520); combinational otracks (522) and all ipin muxes
cost 0. Rebuild: python3 parse_regs.py
Output: {blk: {'inst': gib_inst, 'reg': [port, ...]}}"""
import re, json, os
V = ('/home/lsl/wafer-compiler-for-cgra/simulator/chipyard/sims/verilator/'
     'generated-src/chipyard.TestHarness.TramSoCConfig/'
     'chipyard.TestHarness.TramSoCConfig.top.v')
OUT = os.path.dirname(os.path.abspath(__file__))
txt = open(V, errors='replace').read()
res = {}
for mm in re.finditer(r'module (GIB\w*)\(', txt):
    name = mm.group(1)
    body = txt[mm.end():txt.index('endmodule', mm.end())]
    cm = re.search(r"cfgMatch = \d+'h([0-9a-fA-F]+) == io_cfg_addr", body)
    if not cm:
        continue
    blk = int(cm.group(1), 16)
    outs = re.findall(r'output\s*(?:\[[^\]]+\])?\s+(io_\w+)', body)
    regd = set(re.findall(r'reg\s*(?:\[[^\]]+\])?\s+(io_\w+)_REG\b', body))
    res[blk] = {'inst': int(name.split('_')[1]) if '_' in name else 0,
                'reg': sorted(p[3:] for p in outs if p in regd)}
json.dump(res, open(OUT + '/netlist_regs.json', 'w'), indent=1)
n_reg = sum(len(v['reg']) for v in res.values())
n_out = sum(1 for v in res.values())
print('parsed %d GIBs, %d registered ports' % (n_out, n_reg))
