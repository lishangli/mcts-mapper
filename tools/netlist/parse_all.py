"""Parse cgra-compiler-generated Verilog netlist (top.v) -> authoritative
GIB/PE mux graph. Rebuild: python3 parse_all.py
Outputs: netlist_gibs.json (mux tables keyed by cfg id),
         netlist_wires.json (instance port wires keyed by array num).
Prereq: inst_map.json (rebuilt by inst_map.py)."""
import re, json, os

V = ('/home/lsl/wafer-compiler-for-cgra/simulator/chipyard/sims/verilator/'
     'generated-src/chipyard.TestHarness.TramSoCConfig/'
     'chipyard.TestHarness.TramSoCConfig.top.v')
OUT = os.path.dirname(os.path.abspath(__file__))
txt = open(V, errors='replace').read()
result = {}
for mm in re.finditer(r'module (GIB\w*)\(', txt):
    name = mm.group(1)
    j = txt.index('endmodule', mm.end())
    body = txt[mm.end():j]
    cm = re.search(r"cfgMatch = \d+'h([0-9a-fA-F]+) == io_cfg_addr", body)
    if not cm:
        continue
    blk = int(cm.group(1), 16)
    dst_of, cfg_of, ins_of, direct = {}, {}, {}, {}
    for line in body.split('\n'):
        lm = re.match(r'\s*assign (io_\w+) = (mux_?\d*_io_out)\b', line)
        if lm:
            dst_of[lm.group(2)] = lm.group(1)[3:]
            continue
        cm2 = re.match(r'\s*assign (mux_?\d*)_io_config = cfgOut\[(\d+):(\d+)\]', line)
        if cm2:
            cfg_of[cm2.group(1)] = (int(cm2.group(2)), int(cm2.group(3)))
            continue
        im = re.match(r'\s*assign (mux_?\d*)_io_in_(\d+) = (io_\w+)\b', line)
        if im:
            ins_of.setdefault(im.group(1), {})[int(im.group(2))] = im.group(3)[3:]
            continue
        rm = re.match(r'\s*(?:assign )?(\w*_REG) (?:<=|=) (mux_?\d*_io_out)\b', line)
        if rm:
            dstport = rm.group(1)[:-4]
            if dstport.startswith('io_'):
                dstport = dstport[3:]
            dst_of[rm.group(2)] = dstport
            continue
        rm2 = re.match(r'\s*assign (\w*_REG) = (io_\w+)\b', line)
        if rm2:
            direct[rm2.group(1)[:-4]] = rm2.group(2)[3:]
            continue
        dm = re.match(r'\s*assign (io_(?:ipin|otrack)\w+) = (io_(?:opin|itrack|otrack)\w+)\b', line)
        if dm:
            direct[dm.group(1)[3:]] = dm.group(2)[3:]
    muxes = {}
    for sig, dst in dst_of.items():
        mn = sig.split('_io_out')[0]
        d = ins_of.get(mn, {})
        n = (max(d) + 1) if d else 0
        muxes[dst] = {'cfg': cfg_of.get(mn), 'srcs': [d.get(i) for i in range(n)]}
    inst = int(name.split('_')[1]) if '_' in name else 0
    result[blk] = {'module': name, 'inst': inst, 'mux': muxes, 'direct': direct}

json.dump(result, open(OUT + '/netlist_gibs.json', 'w'), indent=1)
nm = sum(len(r['mux']) for r in result.values())
print('parsed %d GIBs, %d muxes total' % (len(result), nm))
