"""Bridge: MCTS placement -> S2 netlist routing/cfg emission (INPUT-stream).

Routing occupancy (`occ`) is an Occ table, not a bare set: a mux output is
identified by the input index it selects, so two paths may share it when they
agree on that index (same wire, same value, same phase) and a wire consumed as
a source (direct/track hop) stays exclusive. Any leftover double-config still
trips set_gib_bits' gib_owner check, which retriggers the 12x remap loop.

Env switches:
  S2_OCC_STRICT=1  restore the old exclusive semantics (one owner per mux
                   output) for A/B comparison; behaviour equals pre-P0.
  S2_STATS=1       per-attempt occupancy/failure counters on stdout (default
                   off so run_dfg.sh greps and regression.py stay clean).
  S2_PACK=1        fold independent ops onto one PE (see the S2_PACK section).
  S2_PHASE_BUMP=N  how many c0 bump levels a node may try before its parity
                   convergence counts as failed (default 4 = levels 0/2/1/3,
                   i.e. hold the slot first, then flip it; 1 = level 0 only,
                   which reproduces the pre-bump hard-fail behaviour).
"""
import sys, os, json
D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(D, '..', '..'))
SRC = os.environ.get('S2_MAPPER_SRC') or os.path.join(REPO, 'src')
NET = os.environ.get('S2_NETLIST') or os.path.join(REPO, 'tools', 'netlist')
sys.path.insert(0, SRC)
os.chdir(SRC)
from parser import ADGIR, DFGParser, Operations
from mapper import Mapping
from config import NetworkConfig
from models import AgentNetwork
from agent import MCTSPlayer
import env.state as _es
_es.MappingState.start_anime = lambda self, vis: None
_es.MappingState.end_anime = lambda self, vis: None
_es.MappingState.draw = lambda self, vis: None
DFG_JSON = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, 'chain_dfg.json')
MODEL = (sys.argv[2] if len(sys.argv) > 2
         else os.environ.get('S2_MODEL')
         or os.path.join(REPO, 'models', 'best-agent.pt'))
OUT_DIR = os.environ.get('S2_OUT_DIR') or D
# Input stream values by ref_name. gen_nn.py feeds x0..x7 with 1..8 so a
# dot-product layer has distinct activations; %const1/2 keep the old DAG cases.
CONST_VALUES = {'%const1': 3, '%const2': 5, 'CONST1': 3, 'CONST2': 5}
CONST_VALUES.update({'%%x%d' % i: i + 1 for i in range(8)})
PEBLK = {int(k): v for k, v in json.load(open(
    os.path.join(NET, 'pe_blk_map.json'))).items()}
PEBLK_INV = {v: k for k, v in PEBLK.items()}
PE_BLKS = set(PEBLK_INV)   # ADG PE node id == cfg block number
dfgp = DFGParser(DFG_JSON)
dfg = dfgp.getDFG()
ADG_JSON = os.path.join(REPO, 'example', 'cgra_adg.json')
adgir = ADGIR(ADG_JSON)
adg = adgir.getADG()
ops = Operations()
ops.OpParser(os.path.join(REPO, 'example', 'operations.json'))
nodes = dfg.getNodes()
EDGES = dfg.getEdges()
nodes_l = nodes if isinstance(nodes, list) else list(nodes.values())
node_by_id = {n.id: n for n in nodes_l}
op_of = {nid: n.getOpCode().upper() for nid, n in node_by_id.items()}
net_cfg = NetworkConfig()
route_size = len(adg.getEdges())
actions_len = adg.getNodeNums() * (adg.getMaxNodeId() + 1)
agent_net = AgentNetwork(net_cfg.node_feature_dim, net_cfg.edge_feature_dim,
                         net_cfg.hidden_dim, actions_len, route_size,
                         net_cfg.policy_dim, use_gpu=net_cfg.use_gpu,
                         model_file=MODEL)
agent = MCTSPlayer(agent_net, 10)
sys.path.insert(0, D)
from sched import Router, INST2BLK, path_delay, is_reg
from collections import deque
r = Router()
IW = json.load(open(os.path.join(NET, 'netlist_wires.json')))

# Relaxed (default) lets paths share a mux output that selects the same input
# index; S2_OCC_STRICT=1 restores the pre-P0 exclusive semantics for A/B.
# Measured on dfg_big: relaxed emitted on attempt 0 in 5/5 runs, pre-P0 in 5/8
# (8 runs, 3 needed a remap); both modes pass the 8-case regression.
STRICT_OCC = os.environ.get('S2_OCC_STRICT', '0') == '1'

class Occ:
    """Routing occupancy, keyed by (gib, port).

    A mux output is identified by the input index it selects, so two paths may
    share it once they agree on that index: same wire, same value, same phase.
    A port used as a *source* (direct/track hop) stays exclusive, because what
    it carries is whatever some other path configured there.

    Refcounted so a path can release its own claim, reroute for parity, then
    re-claim without stealing a config that another path shares.
    """
    __slots__ = ('strict', 'cfg', 'rd', 'peak')

    def __init__(self, strict=False):
        self.strict = strict
        self.cfg = {}   # (gib, port) -> {idx: refcount}  mux output -> selected input
        self.rd = {}    # (gib, port) -> refcount         wire consumed as a source
        self.peak = 0

    def __contains__(self, key):
        """set-compatible `in`: anything owned at all (sched.Router.route uses it)"""
        return key in self.cfg or key in self.rd

    def __len__(self):
        return len(self.cfg) + len(self.rd)

    def write_ok(self, key, idx):
        """may this path drive mux output `key` from input index `idx`?"""
        if self.strict:
            return key not in self          # legacy: exclusive, even for the same idx
        if key in self.rd:
            return False                    # somebody reads this wire: don't touch it
        idxs = self.cfg.get(key)
        return idxs is None or idx in idxs

    def read_ok(self, key):
        """may this path take wire `key` as its source?"""
        return key not in self.cfg and key not in self.rd

    def claim(self, path):
        """mark everything a path touches; the marks are what release() reverts"""
        marks = []
        for ent in path:
            key = (ent[0], ent[1])
            idx = ent[2] if len(ent) > 2 else None
            if idx is None:
                self.rd[key] = self.rd.get(key, 0) + 1
            else:
                d = self.cfg.setdefault(key, {})
                d[idx] = d.get(idx, 0) + 1
            marks.append((key, idx))
        if len(self) > self.peak:
            self.peak = len(self)
        return marks

    def release(self, marks):
        for key, idx in marks:
            if idx is None:
                n = self.rd.get(key, 0) - 1
                if n > 0: self.rd[key] = n
                else: self.rd.pop(key, None)
                continue
            d = self.cfg.get(key)
            if not d or idx not in d: continue
            if d[idx] > 1: d[idx] -= 1
            else:
                del d[idx]
                if not d: self.cfg.pop(key, None)

    def stats(self):
        return 'cfg=%d rd=%d peak=%d' % (len(self.cfg), len(self.rd), self.peak)

def as_occ(occ):
    """normalise the optional occ argument (bare tuple/set/None -> empty table)"""
    return occ if isinstance(occ, Occ) else Occ(strict=STRICT_OCC)

def restore_base(ins, base, base_pins, hs, route_of, claimed, port_of,
                 port_used, di, occ):
    """Roll one node's in-edge routes back to the snapshot taken right after its
    unconstrained routing, so every phase-bump level starts from the same state.

    Order is load bearing: release what the failed level still owns *before* the
    base paths are re-claimed.  `occ` is refcounted, so releasing marks twice (or
    releasing marks that a failed level already dropped) would underflow onto a
    config another path shares - a silent cross-talk, not an error.
    """
    for e in ins:
        if e.id not in base: continue
        if claimed.get(e.id):
            occ.release(claimed[e.id])
        claimed[e.id] = []          # nothing of this level survives the rollback
    cur = {p for p in (port_of.get(e.id) for e in ins if e.id in base) if p is not None}
    pins = port_used.setdefault(di, set())
    pins.difference_update(cur)     # only this node's own pins: the PE may host another op
    pins.update(base_pins)
    for e in ins:
        if e.id not in base: continue
        b_port, b_path, b_hs = base[e.id]
        port_of[e.id] = b_port
        route_of[e.id] = b_path
        hs[e.id] = b_hs
        claimed[e.id] = occ.claim(b_path)

# S2_STATS=1 prints per-attempt occupancy/failure counters (off by default so
# run_dfg.sh greps and regression.py's 'emission failed' counting stay clean)
STATS_ON = os.environ.get('S2_STATS', '0') == '1'
FAIL_REASONS = {}

# S2_PHASE_BUMP: the node's earliest launch cycle c0 fixes which phase class the
# in-edges must hit (want = (c0 - ctx[tail]) % 4). Bumping c0 shifts `want`, so a
# parity class that is unreachable under the current occupancy often becomes
# reachable at c0+2. Level order is (0, 2, 1, 3): +2 first because it keeps the
# slot (slot = ctx % 2, and two nodes on one PE must sit in opposite slots),
# then +1/+3 which flip it. Semantically safe: PEs are stateless per cycle, so
# launching later only moves the value wave, never changes it.
PHASE_BUMP = max(1, int(os.environ.get('S2_PHASE_BUMP', '4')))
PHASE_LEVELS = (0, 2, 1, 3)[:min(4, PHASE_BUMP)]
PHASE_BUMPS = {}     # bump level -> nodes that needed it (all levels tried are counted)
BUMP_NODES = []      # 'Op@+2' for every node that only converged after a bump

def fail_kind(msg):
    """bucket an assertion message for the retry statistics"""
    for k in ('no parity route', 'no route', 'no const IOB', 'no capture IOB', 'conflict'):
        if k in msg:
            return k
    return msg[:40]

def route_to_terms(start, terms, occ=()):
    """BFS upstream from a dst pin to any of `terms`; returns 3-tuples.

    Occupancy is decided per step, not per node: a mux step may reuse a mux
    output that already selects the same input index, while a direct/track step
    reads the current wire and therefore needs it untouched.
    """
    occ = as_occ(occ)
    q = deque([(start, [])]); seen = {start}
    while q:
        (g, p), pa = q.popleft()
        if (g, p) in terms and pa: return pa
        if pa and occ.strict and (g, p) in occ: continue    # legacy exclusive prune
        for kind, gib, dstport, idx, src in r.preds(g, p):
            if kind == 'mux':
                if not occ.write_ok((gib, dstport), idx): continue
                node = (gib, src)
            elif kind == 'direct':
                if pa and not occ.read_ok((g, p)): continue
                node = (gib, src)
            else:                                    # track: also reads (g, p)
                if pa and not occ.read_ok((g, p)): continue
                node = src
            if node not in seen:
                seen.add(node); q.append((node, pa + [(gib, dstport, idx)]))
    return None

def route_parity(src_inst, dst_inst, port, want, occ=()):
    """route whose CYCLE delay == want (mod 4), reverse BFS from the dst input pin.
    Only registered otrack hops cost a cycle (77/153 GIBs are combinational), so
    hop count and delay differ and the ctx constraint must use the latter.
    Nodes are visited once: a port resolves to exactly ONE mux source, so a walk
    that revisits a port would need two different mux configs for the same wire
    (the BFS used to allow that via per-parity visited states and emitted
    self-contradicting cin rows)."""
    occ = as_occ(occ)
    tgt = IW['pe_in'][str(dst_inst)].get(str(port))
    assert tgt is not None, 'no pe_in %d' % dst_inst
    terms = set(tuple(p) for p in IW['pe_out'][str(src_inst)])
    start = tuple(tgt)
    q = deque([(start, 0, [])]); seen = {start}
    while q:
        (g, p), par, pa = q.popleft()
        if (g, p) in terms and pa and par % 4 == want:
            return pa
        if pa and occ.strict and (g, p) in occ: continue
        npar = (par + (1 if is_reg(g, p) else 0)) % 4
        for kind, gib, dstport, idx, src in r.preds(g, p):
            if kind == 'mux':
                if not occ.write_ok((gib, dstport), idx): continue
            elif pa and not occ.read_ok((g, p)):
                continue
            node = src if kind == 'track' else (gib, src)
            if node in seen: continue
            seen.add(node)
            q.append((node, npar, pa + [(gib, dstport, idx)]))
    return None

def iob_stream_route(dst_inst, port, avoid=(), occ=()):
    """route IOB(out) -> PE in: BFS upstream from PE in pin to any unused IOB out pin"""
    occ = as_occ(occ)
    tgt = IW['pe_in'][str(dst_inst)].get(str(port))
    if tgt is None: return None
    terms = {}
    for n in range(32):
        for o in IW['iob_out'].get(str(n), []):
            terms[tuple(o)] = n
    q = deque([(tuple(tgt), [])]); seen = {tuple(tgt)}
    while q:
        (g, p), pa = q.popleft()
        if (g, p) in terms and pa and terms[(g, p)] not in avoid:
            return terms[(g, p)], pa
        if pa and occ.strict and (g, p) in occ: continue
        for kind, gib, dstport, idx, src in r.preds(g, p):
            if kind == 'mux':
                if not occ.write_ok((gib, dstport), idx): continue
            elif pa and not occ.read_ok((g, p)):
                continue
            node = (gib, src) if kind != 'track' else src
            if node not in seen:
                seen.add(node)
                q.append((node, pa + [(gib, dstport, idx)]))
    return None

def pick_iob(src_pe, avoid=(), occ=()):
    """src_pe is a netlist inst id; ADG node ids (== cfg blks) carry coordinates"""
    adg = json.load(open(ADG_JSON))
    bypos = {}
    for x in adg['instances']:
        bypos[x['id']] = (x.get('x'), x.get('y'))
    sx, sy = bypos.get(PEBLK.get(src_pe, src_pe), (999, 999))
    def iob_dist(n):
        iid = n + 1 if n < 16 else n + 291   # IOB cfg blk == ADG node id
        x, y = bypos.get(iid, (999, 999))
        return abs(x - sx) + abs(y - sy)
    for n in sorted(range(32), key=iob_dist):
        if n in avoid: continue
        cins = {int(k): v for k, v in IW['iob_in'].get(str(n), {}).items()}
        if not cins: continue
        cpi = 0 if 0 in cins else sorted(cins)[0]
        terms = set(tuple(x) for x in IW['pe_out'][str(src_pe)])
        pa = route_to_terms(tuple(cins[cpi]), terms, occ)
        if pa: return n, cpi, pa
    return None, None, None

def op_mux_slot(inst, operand, kind, p):
    """select index of port p in the REQUESTED operand mux's srcs (both mux
    variants: disjoint groups or shared ports with independent selects)"""
    import sched
    name = ('io_in_%d' % p) if kind == 'in' else 'cfgOut[31:0]'
    info = sched.PES[str(inst)]['mux']['rdu_io_in_%d' % operand]
    assert name in info['srcs'], '%s not in op%d srcs %s' % (name, operand, info['srcs'])
    return info['srcs'].index(name)

def group_ports(inst, operand):
    import sched
    return sorted(int(x[6:]) for x in
                  sched.PES[str(inst)]['mux']['rdu_io_in_%d' % operand]['srcs']
                  if x.startswith('io_in_'))

LAST_ROUTE_FAIL = {}   # set when route_operand runs out of candidates

def route_operand(src_inst, dst_inst, operand, prefer=None, occ=(), used_ports=()):
    """route to a port in the operand's mux group (per-PE heterogeneous).
    used_ports: pins already carrying a signal for this PE - a physical pin can
    only carry one value, so two operands on the same pin silently alias.

    sched.Router.route only knows `key in occ` (exclusive), so the relaxed mode
    drives the same search through route_to_terms, whose step predicates can
    share a mux output configured to the same input index."""
    occ = as_occ(occ)
    grp = [p for p in group_ports(dst_inst, operand) if p not in used_ports]
    cands = ([prefer] if prefer in grp else []) + [c for c in grp if c != prefer]
    terms = None
    for cand in cands:
        if occ.strict:
            pth = r.route(src_inst, dst_inst, cand, occ=occ)
        else:
            tgt = IW['pe_in'][str(dst_inst)].get(str(cand))
            if tgt is None: continue
            if terms is None:
                terms = set(tuple(x) for x in IW['pe_out'][str(src_inst)])
            pth = route_to_terms(tuple(tgt), terms, occ)
        if pth: return cand, pth
    # Diagnostics: distinguish "this PE has no spare pin" from "the netlist path
    # exists but occupancy blocks it" from "no path at all". Only the middle case
    # is congestion, and only that one benefits from placement changes.
    free_pins = [p for p in group_ports(dst_inst, operand) if p not in used_ports]
    free_path = None
    if free_pins:
        tgt = IW['pe_in'][str(dst_inst)].get(str(free_pins[0]))
        if tgt is not None:
            free_path = route_to_terms(tuple(tgt),
                                       set(tuple(x) for x in IW['pe_out'][str(src_inst)]),
                                       ())
    LAST_ROUTE_FAIL.clear()
    LAST_ROUTE_FAIL.update({
        'src': src_inst, 'dst': dst_inst, 'operand': operand,
        'group': group_ports(dst_inst, operand), 'used': sorted(used_ports),
        'free_pins': free_pins,
        'kind': ('no free pin' if not free_pins
                 else 'congestion' if free_path else 'structurally unroutable'),
        'free_hops': len(free_path) if free_path else None,
    })
    return None, None

OPC = {'PASS': 0, 'ADD': 1, 'SUB': 2, 'MUL': 3}
POS = {'BaseAddr': 0, 'Stride0': 14, 'Cycles0': 27, 'II': 89, 'IsStore': 96, 'Muxn': 102}
WID = {'BaseAddr': 14, 'Stride0': 13, 'Cycles0': 12, 'II': 1, 'IsStore': 1, 'Muxn': 2}
def iob_cfg(fields):
    cfg = 0
    for k, v in fields.items():
        cfg |= (v & ((1 << WID[k]) - 1)) << POS[k]
    return [(cfg >> (32 * i)) & 0xFFFFFFFF for i in range(4)]
def iob_rows(n, is_store, mux):
    iw = iob_cfg({'BaseAddr': (n % 16) * 1024, 'Stride0': 1, 'Cycles0': 64,
                  'II': 1, 'IsStore': is_store, 'Muxn': mux})
    blk_io = n + 1 if n < 16 else n + 291
    return [((blk_io << 3) | i, iw[i]) for i in range(4)]

# ==== mapping + emission with retries ====
placement = None
EMIT_OK = False
for attempt in range(12):
    mapping = Mapping(dfg, adg, ops, False)
    mapping.use_true_reward()
    lat = mapping.test_mapping(agent, is_shown=False, temp=1e-2)
    if lat is None:
        if STATS_ON:
            FAIL_REASONS['mapping failed'] = FAIL_REASONS.get('mapping failed', 0) + 1
        continue
    cfg = mapping.state.build_config_dict()
    placement = {int(nid) - 1: attr['adgNode']
                 for nid, attr in cfg['dfg_node_attrs'].items()}
    def on_iob(adg_id):
        return adg_id is not None and (adg_id < 16 or adg_id >= 307)

    def is_stream(nid):
        """node realised as an IOB constant stream instead of a PE"""
        op = op_of[nid]
        return op == 'INPUT' or (op == 'CONST' and on_iob(placement.get(nid)))

    def node_value(nid):
        """CONST carries its value in .info (str), INPUT in CONST_VALUES by name"""
        n = node_by_id[nid]
        info = getattr(n, 'info', None)
        if isinstance(info, (int, float)):
            return int(info)
        if isinstance(info, str):
            try: return int(info, 0)
            except ValueError: pass
        if isinstance(info, dict):
            ref = info.get('ref_name')
            if ref in CONST_VALUES: return CONST_VALUES[ref]
        return CONST_VALUES.get(n.getOpName(), 0)

    if not all(placement[nid] in PEBLK_INV for nid, n in node_by_id.items()
               if n.getOpCode().upper() not in ('OUTPUT', 'INPUT') and not is_stream(nid)):
        continue
    port_of = {}
    for e in EDGES:
        eattr = cfg['dfg_edge_attrs'].get(str(e.id)) or cfg['dfg_edge_attrs'].get(e.id)
        if eattr is None: continue
        dst_adg = placement[e.head]
        links = eattr['edgeLinks'].get(dst_adg)
        if links and 'srcPort' in links:
            port_of[e.id] = links['srcPort']
    print('placement:', placement, 'port_of:', port_of)
    preds = {nid: [] for nid in node_by_id}
    for e in EDGES:
        preds[e.head].append(e.tail)
    ctx_of = {}
    try:
        iob_streams = []
        for e in EDGES:
            if is_stream(e.tail):
                iob_streams.append((e.tail, node_value(e.tail), e.head, e.operand))
        print('iob streams:', iob_streams)
        slot_gpe = {0: {}, 1: {}}
        slot_gib = {0: {}, 1: {}}
        gib_owner = {}   # (slot, blk, dstport) -> (label, idx) that drove this mux
        def set_gib_bits(slot, gib_inst, dstport, idx, label='?'):
            assert gib_inst in INST2BLK, 'inst %d is not a GIB' % gib_inst
            blk = INST2BLK[gib_inst]
            assert blk not in PE_BLKS, 'GIB %d -> blk %d collides with PE blk' % (gib_inst, blk)
            import sched
            info = sched.GIBS[str(blk)]['mux'][dstport]
            hi, lo = info['cfg']
            key = (slot, blk, dstport)
            prev = gib_owner.get(key)
            assert prev is None or prev[1] == idx, \
                'conflict %d %s (idx %d from %s vs idx %d from %s)' \
                % (blk, dstport, prev[1], prev[0], idx, label)
            gib_owner[key] = (label, idx)
            w = slot_gib[slot].setdefault(blk, 0)
            mask = ((1 << (hi - lo + 1)) - 1) << lo
            slot_gib[slot][blk] = (w & ~mask) | (idx << lo)
        def emit_route(path, slot, label='?'):
            for ent in path:            # 4-tuples from Router.route, 3 from route_parity
                gib, dstport, idx = ent[0], ent[1], ent[2]
                if idx is not None:
                    set_gib_bits(slot, gib, dstport, idx, label)
        # topo-ordered parity-aware ctx assignment (S2 scheduler core)
        order = []; visd = set()
        def visit(nid):
            if nid in visd: return
            visd.add(nid)
            for p_ in preds[nid]: visit(p_)
            order.append(nid)
        for nid in node_by_id: visit(nid)
        route_of = {}
        # mux configs shareable by input index, wires read as a source exclusive
        occ = Occ(strict=STRICT_OCC)
        port_used = {}   # pe inst -> input pins already carrying a signal
        for nid in order:
            op = op_of[nid]
            if op in ('INPUT', 'CONST'):
                ctx_of[nid] = 0; continue
            ins = [e for e in EDGES if e.head == nid]
            if op == 'OUTPUT':
                ctx_of[nid] = ctx_of[ins[0].tail]; continue
            hs = {}
            claimed = {}   # edge id -> mux outputs it currently owns in `occ`
            opnd_of = {}
            di = PEBLK_INV[placement[nid]]
            for e in ins:
                if is_stream(e.tail):
                    continue  # IOB stream: constant present every cycle
                opnd = e.operand if e.operand in (0, 1) else 0
                opnd_of[e.id] = opnd
                in_p, path = route_operand(PEBLK_INV[placement[e.tail]], di, opnd,
                                           prefer=port_of.get(e.id), occ=occ,
                                           used_ports=port_used.get(di, ()))
                assert path, 'no route %d->%d op%d %s' % (
                    e.tail, e.head, opnd, LAST_ROUTE_FAIL or '{}')
                port_used.setdefault(di, set()).add(in_p)
                hs[e.id] = path_delay(path); route_of[e.id] = path
                port_of[e.id] = in_p
                # claim immediately: sibling edges of this same node must not be
                # allowed to reuse a mux this edge already drives
                claimed[e.id] = occ.claim(path)
            base_c0 = max([ctx_of[e.tail] + hs[e.id] for e in ins if e.id in hs] + [1])
            # snapshot of the unconstrained routing: every bump level restarts here
            base = {e.id: (port_of[e.id], route_of[e.id], hs[e.id])
                    for e in ins if e.id in hs}
            base_pins = {port_of[e.id] for e in ins if e.id in hs}

            def settle(c0):
                """parity convergence for ONE bump level -> (ok, c0, dead edge).

                A dead end no longer aborts the whole attempt: the caller retries the
                node at a later c0 (see PHASE_LEVELS), which shifts `want` and hence
                the phase class the router has to hit.  `claimed[dead]` is left empty
                (its marks were released before the failed search), so the rollback in
                restore_base() can never double-release them.
                """
                for _it in range(4):
                    for e in ins:
                        if e.id not in hs: continue
                        want = (c0 - ctx_of[e.tail]) % 4
                        if hs[e.id] % 4 != want:
                            # release own claim so the search may reuse its own wiring
                            occ.release(claimed[e.id]); claimed[e.id] = []
                            p2 = route_parity(PEBLK_INV[placement[e.tail]], di,
                                              port_of[e.id], want, occ=occ)
                            if p2 is None:
                                # pin dead-ends under the parity+occ constraints: move
                                # this operand to another free pin of the same mux group
                                for cand in group_ports(di, opnd_of[e.id]):
                                    if cand == port_of[e.id]: continue
                                    if cand in port_used.get(di, ()): continue
                                    q = route_parity(PEBLK_INV[placement[e.tail]], di,
                                                     cand, want, occ=occ)
                                    if not q: continue
                                    port_used[di].discard(port_of[e.id])
                                    port_used[di].add(cand)
                                    port_of[e.id] = cand
                                    p2 = q
                                    break
                            if not p2:
                                return False, c0, e
                            hs[e.id] = path_delay(p2); route_of[e.id] = p2
                            claimed[e.id] = occ.claim(p2)
                    c0n = max([ctx_of[e.tail] + hs[e.id] for e in ins if e.id in hs] + [c0])
                    if c0n == c0: break
                    c0 = c0n
                return True, c0, None

            dead = None
            for bump in PHASE_LEVELS:
                if bump:
                    restore_base(ins, base, base_pins, hs, route_of, claimed,
                                 port_of, port_used, di, occ)
                    c0 = base_c0 + bump     # later launch -> different `want`
                else:
                    c0 = base_c0
                ok, c0, dead = settle(c0)
                if ok: break
            assert ok, 'no parity route %d->%d (bumped %d)' % (
                dead.tail, dead.head, PHASE_LEVELS[-1])
            PHASE_BUMPS[bump] = PHASE_BUMPS.get(bump, 0) + 1
            if bump:
                BUMP_NODES.append('%s@+%d' % (node_by_id[nid].getOpName(), bump))
            # only ctx mod 4 is meaningful; keep it normalised so long routes stop
            # ratcheting the absolute counter (and the parity loop with it)
            ctx_of[nid] = c0 % 4
            print('sched %s(ctx=%d slot=%d%s):' % (node_by_id[nid].getOpName(), ctx_of[nid],
                                                   ctx_of[nid] % 2,
                                                   ' bumped+%d' % bump if bump else ''),
                  [(e.tail, hs.get(e.id), 'want=%d' % ((c0 - ctx_of[e.tail]) % 2))
                   for e in ins])
        # routes must be alive in BOTH slots (value wave propagates via regs)
        for eid, path in route_of.items():
            e = next(x for x in EDGES if x.id == eid)
            for slot in (0, 1): emit_route(path, slot, 'edge%d(%d->%d)' % (eid, e.tail, e.head))
        # INPUT streams: IOB out -> consumer's in (one IOB per stream: each bank
        # holds exactly one constant vector, sharing an IOB would clobber values)
        din_meta = []
        used_iob = set()
        for si, (cnid, val, consumer, operand) in enumerate(iob_streams):
            e0 = next(e for e in EDGES if e.tail == cnid and e.head == consumer)
            opnd = operand if operand in (0, 1) else 0
            dst_inst = PEBLK_INV[placement[consumer]]
            got = None
            for cand in group_ports(dst_inst, opnd):
                if cand in port_used.get(dst_inst, ()): continue
                if IW['pe_in'][str(dst_inst)].get(str(cand)) is None: continue
                got = iob_stream_route(dst_inst, cand, used_iob, occ)
                if got: in_p = cand; break
            assert got is not None, 'no const IOB'
            port_used.setdefault(dst_inst, set()).add(in_p)
            port_of[e0.id] = in_p
            n, pa = got
            used_iob.add(n)
            occ.claim(pa)
            for gib, dstport, idx in pa:
                if idx is not None:
                    for slot in (0, 1):
                        set_gib_bits(slot, gib, dstport, idx,
                                     'stream%d(iob%d->%d)' % (si, n, consumer))
            din_meta.append((si, n, val))
            print('const val=%d -> iobs_%d' % (val, n))
        # S2_PACK=1: fold independent compute ops onto the first op's PE, one slot
        # each (time multiplexing). Only legal when the folded op's operands all
        # come from IOB streams (no pre-computed inter-PE route to invalidate).
        if int(os.environ.get('S2_PACK', '0')):
            computes = [nid for nid in order
                        if op_of[nid] not in ('INPUT', 'OUTPUT', 'CONST')
                        and not is_stream(nid)]
            if len(computes) > 1:
                host = computes[0]
                for k, nid in enumerate(computes[1:]):
                    ins = [e for e in EDGES if e.head == nid]
                    placement[nid] = placement[host]
                    d = 0
                    for e in ins:
                        if is_stream(e.tail):
                            continue      # IOB stream: every cycle, no phase issue
                        assert e.tail == host, \
                            'pack %s: only self-loop or stream operands supported' \
                            % node_by_id[nid].getOpName()
                        opnd = e.operand if e.operand in (0, 1) else 0
                        hi_ = PEBLK_INV[placement[host]]
                        in_p, pth = route_operand(hi_, hi_, opnd, occ=occ,
                                                  used_ports=port_used.get(hi_, ()))
                        assert pth, 'pack %s: no self-loop route' % node_by_id[nid].getOpName()
                        port_used.setdefault(hi_, set()).add(in_p)
                        port_of[e.id] = in_p
                        d = max(d, path_delay(pth))
                        route_of[e.id] = pth
                        occ.claim(pth)
                        for slot in (0, 1):
                            emit_route(pth, slot, 'pack%d(%d->%d)' % (e.id, e.tail, e.head))
                    ctx_of[nid] = ctx_of[host] + (d if d else 1 + 2 * k)
                    assert ctx_of[nid] % 2 != ctx_of[host] % 2, \
                        'pack %s: self-loop delay %d keeps slot %d (need opposite slot)' \
                        % (node_by_id[nid].getOpName(), d, ctx_of[host] % 2)
                    print('packed %s onto PE blk %d slot %d (host %s, delay %d)'
                          % (node_by_id[nid].getOpName(), placement[nid],
                             ctx_of[nid] % 2, node_by_id[host].getOpName(), d))
        pe_slot_occ = {}   # (pe blk, slot) -> op name, guards silent GPE cfg overwrite
        for nid, n in node_by_id.items():
            op = op_of[nid]
            if op in ('OUTPUT', 'INPUT') or is_stream(nid):
                continue
            pe = placement[nid]
            slot = ctx_of[nid] % 2
            w0 = 0
            if op == 'CONST':
                # materialise as PASS(cfgOut) on the mapper-chosen PE: constant is
                # the cfg word itself, so no operand routing is needed
                opc = OPC['PASS']
                w0 = node_value(nid) & 0xFFFFFFFF
                m0 = op_mux_slot(PEBLK_INV[pe], 0, 'const', None)
                m1 = 0
            else:
                opc = OPC[op]
                m0 = m1 = 0
                for e in EDGES:
                    if e.head != nid: continue
                    operand = e.operand if e.operand in (0, 1) else 0
                    in_p = port_of.get(e.id)
                    assert in_p is not None, 'no port for %s' % e.id
                    slot_v = op_mux_slot(PEBLK_INV[pe], operand, 'in', in_p)
                    if operand == 0: m0 = slot_v
                    else: m1 = slot_v
            w1 = opc | (m0 << 13) | (m1 << 16)
            # ADG PE node id == cfg block number (PEBLK values); INST2BLK is GIB-only
            assert pe in PE_BLKS, 'PE node %d not a cfg block' % pe
            assert (pe, slot) not in pe_slot_occ, (
                'PE blk %d slot %d taken by %s, collides with %s' % (
                    pe, slot, pe_slot_occ.get((pe, slot)), node_by_id[nid].getOpName()))
            slot_gpe[slot][pe] = (w0, w1)
            pe_slot_occ[(pe, slot)] = node_by_id[nid].getOpName()
        # a PE hosting a single op runs it in BOTH slots: an empty slot decodes as
        # PASS(0) and injects zeros that downstream PEs sample half the time
        for pe in {k[0] for k in pe_slot_occ}:
            occ_slots = [s for s in (0, 1) if (pe, s) in pe_slot_occ]
            if len(occ_slots) == 1:
                src_slot = occ_slots[0]
                if pe not in slot_gpe[1 - src_slot]:
                    slot_gpe[1 - src_slot][pe] = slot_gpe[src_slot][pe]
        # OUTPUT capture: one IOB channel per OUTPUT node (multi-output DFGs)
        caps = []   # (iob_sel, pi_c, out name)
        for onid in sorted(nid for nid in node_by_id if op_of[nid] == 'OUTPUT'):
            src_pe = PEBLK_INV[placement[[e.tail for e in EDGES if e.head == onid][0]]]
            iob_sel = None
            mapper_iob = placement.get(onid)
            if mapper_iob is not None and (mapper_iob < 16 or mapper_iob >= 307):
                cand_sel = mapper_iob - 1 if mapper_iob < 16 else mapper_iob - 291
                if cand_sel not in used_iob:   # never share an IOB with a load stream
                    iob_sel = cand_sel
            cap_path = None
            pi_c = 0
            if iob_sel is not None:
                cins = {int(k): v for k, v in IW['iob_in'].get(str(iob_sel), {}).items()}
                if cins:
                    pi_c = 0 if 0 in cins else sorted(cins)[0]
                    terms = set(tuple(x) for x in IW['pe_out'][str(src_pe)])
                    cap_path = route_to_terms(tuple(cins[pi_c]), terms, occ)
            if cap_path is None:
                iob_sel, pi_c, cap_path = pick_iob(src_pe, used_iob, occ)
                assert iob_sel is not None, 'no capture IOB'
            used_iob.add(iob_sel)
            occ.claim(cap_path)
            print('capture %s -> iobs_%d' % (node_by_id[onid].getOpName(), iob_sel))
            for gib, dstport, idx in cap_path:
                if idx is not None:
                    for slot in (0, 1):
                        set_gib_bits(slot, gib, dstport, idx,
                                     'capture(%d->iob%d)' % (onid, iob_sel))
            caps.append((iob_sel, pi_c))
        # emit cin
        rowsA, rowsB = [], []
        for slot, rows in ((0, rowsA), (1, rowsB)):
            for blk, (w0, w1) in sorted(slot_gpe[slot].items()):
                rows.append(((blk << 3) | 0, w0)); rows.append(((blk << 3) | 1, w1))
            for blk, w0 in sorted(slot_gib[slot].items()):
                rows.append(((blk << 3) | 0, w0))
        for si, n, val in din_meta:
            for r_ in iob_rows(n, 0, 0):
                rowsA.append(r_); rowsB.append(r_)
        for iob_sel, pi_c in caps:
            for r_ in iob_rows(iob_sel, 1, pi_c):
                rowsA.append(r_); rowsB.append(r_)
        while (len(rowsA) * 6) % 16 != 0:
            rowsA.append(rowsA[-1])   # idempotent pad (never target a real PE blk)
        base_b = (len(rowsA) * 6) // 16
        allrows = rowsA + rowsB
        lines = ['\t\t{0x%04x, 0x%04x, 0x%04x},' % (d & 0xFFFF, d >> 16, a)
                 for a, d in allrows]
        iob_ens = 0
        for s, _ in caps: iob_ens |= (1 << s)
        for si, n, val in din_meta: iob_ens |= (1 << n)
        ld = ''.join('\tload_data(din_addr[%d], 0x%x, 256, 0, 0, 0);\n' % (si, n * 4096)
                     for si, n, val in din_meta)
        st = ''.join('\tstore(dout_addr[%d], 0x%x, 512, 0, 0);\n' % (i, s * 4096)
                     for i, (s, _) in enumerate(caps))
        # s2_rdcycle brackets execute()/store() so the run log carries the
        # hardware cost of one kernel, the ground truth for the timing model
        # (docs/EXECUTION_MODEL.md).
        exe = ('#include "cgra_execute.h"\n'
               '#include <stdio.h>\n'
               'static inline unsigned long s2_rdcycle(void) { unsigned long c;\n'
               '  __asm__ volatile("rdcycle %%0" : "=r"(c)); return c; }\n'
               'void cgra_execute0(void** din_addr, void** dout_addr)\n{\n'
               '\tstatic unsigned short cin[%d][3] __attribute__((aligned(64))) = {\n'
               '%s\n\t};\n\n'
               '\tload_cfg((void*)cin, 0x20000, %d, 0, 0);\n' % (len(allrows), '\n'.join(lines), len(allrows) * 6)
               + ld +
               '\tconfig(0x0, %d, 0, 0);\n'
               '\tconfig(0x%x, %d, 0, 0);\n'
               '\tunsigned long t0 = s2_rdcycle();\n'
               '\texecute(0x%x, 0, 0);\n'
               '\tunsigned long t1 = s2_rdcycle();\n' % (len(rowsA), base_b, len(rowsB), iob_ens)
               + st +
               '\tunsigned long t2 = s2_rdcycle();\n'
               '\tfence(1);\n'
               '\tunsigned long t3 = s2_rdcycle();\n'
               '\tprintf("S2_PERF exec=%lu store=%lu total=%lu\\n",\n'
               '\t       t1 - t0, t2 - t1, t3 - t0);\n'
               '}\n')
        open(OUT_DIR + '/cgra_execute.c', 'w').write(exe)
        # generate matching main (one const array per stream). 64B alignment keeps
        # the CPU fill loop from emitting misaligned vector stores, which make the
        # TL monitor fire "PutPartial contains invalid mask" on buffer_1
        mn = '#include <stdio.h>\n#include <stdint.h>\n'
        mn += 'extern void cgra_execute0(void** din_addr, void** dout_addr);\n'
        for i in range(len(caps)):
            mn += 'int O%d[128] __attribute__((aligned(64)));\n' % i
        for si, n, val in din_meta:
            mn += 'int c%d[64] __attribute__((aligned(64)));\n' % si
        mn += 'int main(void) {\n'
        for si, n, val in din_meta:
            mn += '  for (int i = 0; i < 64; i++) c%d[i] = %d;\n' % (si, val)
        mn += '  void* din[%d]; void* dout[%d];\n' % (max(len(din_meta), 1), len(caps))
        for si, n, val in din_meta:
            mn += '  din[%d] = (void*)c%d;\n' % (si, si)
        for i in range(len(caps)):
            mn += '  dout[%d] = (void*)O%d;\n' % (i, i)
        mn += '  cgra_execute0(din, dout);\n  printf("k1 out:\\n");\n'
        for i in range(len(caps)):
            mn += ('  for (int k = 0; k < 128; k++) printf("O%d[%%d] = %%d\\n", k, O%d[k]);\n'
                   % (i, i))
        mn += '  return 0;\n}\n'
        open(OUT_DIR + '/main_s2.c', 'w').write(mn)
        json.dump({'placement': placement, 'port_of': port_of,
                   'captures': caps, 'base_b': base_b,
                   'rowsA': len(rowsA), 'rowsB': len(rowsB),
                   'din_meta': din_meta},
                  open(OUT_DIR + '/bridge_meta.json', 'w'), indent=1)
        print('cin: %d rows (A=%d B=%d), iob_ens=0x%x, captures=%s' %
              (len(allrows), len(rowsA), len(rowsB), iob_ens, [s for s, _ in caps]))
        if STATS_ON:
            print('stats: attempt %d ok, occ %s, strict=%s, failures %s'
                  % (attempt, occ.stats(), STRICT_OCC, dict(FAIL_REASONS)))
            print('stats: phase bumps %s (levels %s) nodes [%s]'
                  % (dict(sorted(PHASE_BUMPS.items())), list(PHASE_LEVELS),
                     ' '.join(BUMP_NODES)))
        EMIT_OK = True
        break
    except AssertionError as e:
        if STATS_ON:
            k = fail_kind(str(e))
            FAIL_REASONS[k] = FAIL_REASONS.get(k, 0) + 1
            o = globals().get('occ')
            print('stats: attempt %d failed (%s), occ %s'
                  % (attempt, k, o.stats() if isinstance(o, Occ) else 'n/a'))
        print('emission failed, retrying:', e)
        continue
if placement is None or not EMIT_OK:
    if STATS_ON:
        print('stats: gave up after %d attempts, failures %s, bumps %s'
              % (attempt + 1, dict(FAIL_REASONS), dict(sorted(PHASE_BUMPS.items()))))
    print('MAPPING/EMISSION FAILED')
    sys.exit(1)
