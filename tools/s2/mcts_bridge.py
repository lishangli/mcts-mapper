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
import sys, os, json, random, copy
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
# S2_RIPUP=0 disables the intra-node rip-up retry (second operand order), which
# is the A/B control for "does routing order matter for this DFG".
RIPUP = os.environ.get('S2_RIPUP', '1') == '1'
# S2_RIPUP_CROSS=1 lets a failing edge release previously routed edges and
# re-route them (PathFinder-lite). Victims must keep their exact delay, so no
# already-assigned context can be invalidated.
CROSS_RIPUP = os.environ.get('S2_RIPUP_CROSS', '1') == '1'
# S2_SLOT_OCC=1 makes routing occupancy time-expanded (docs/MRRG_PLAN.md):
# a GIB mux is claimed per config bank, so two paths can share a wire when
# their values arrive in different contexts. Default off until RTL-validated.
SLOT_OCC = os.environ.get('S2_SLOT_OCC', '0') == '1'
# S2_NO_PHASE=1 drops the mod-4 delivery constraint (any path delay accepted).
# Routes are emitted into BOTH config banks and a single-op PE is duplicated
# into both slots, so the value should be present every cycle; this switch
# tests that claim against RTL instead of assuming it.
# Default ON for the normal flow: the emitted route lives in both banks, so the
# mod-4 check only adds failures. Time-folded operators (S2_PACK) keep it,
# because there each PE hosts two different ops and the output is only valid in
# the context its slot is active.
NO_PHASE = os.environ.get(
    'S2_NO_PHASE',
    '0' if os.environ.get('S2_PACK', '0') == '1' else '1') == '1'
# S2_ALIGN_OPNDS=1 makes every operand of a node take the SAME path delay.
# A PE samples all input pins in one cycle, so equal delay is the only way the
# node multiplies the same sample twice; the mod-4 residue is not enough (the
# ramp testbench measured a one-sample skew without this).
ALIGN_OPNDS = os.environ.get('S2_ALIGN_OPNDS', '1') == '1'

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
        return (self._any_cfg(key) or (key[0], key[1], 0) in self.rd or
                (key[0], key[1], 1) in self.rd or (key[0], key[1], None) in self.rd)

    def __len__(self):
        return len(self.cfg) + len(self.rd)

    def _any_cfg(self, key):
        """is this wire configured at all (in any slot)?"""
        return ((key[0], key[1], 0) in self.cfg or
                (key[0], key[1], 1) in self.cfg or
                (key[0], key[1], None) in self.cfg)

    def write_ok(self, key, idx, slot=None):
        """may this path drive mux output `key` from input index `idx`?

        `slot` = config bank (ctx parity) the value arrives in. Two paths may own
        the same wire when their slots differ, because a GIB has cfgA/cfgB and
        only one of them is active in a given context (CGRA.scala ctx_ptr(0)).
        slot=None means "unknown": such a claim must stay exclusive.
        """
        rd_used = ((key[0], key[1], 0) in self.rd or
                   (key[0], key[1], 1) in self.rd or
                   (key[0], key[1], None) in self.rd)
        if self.strict:
            return not self._any_cfg(key) and not rd_used
        if rd_used:
            return False                    # somebody reads this wire: don't touch it
        cands = (((key[0], key[1], 0), (key[0], key[1], 1), (key[0], key[1], None))
                 if slot is None else
                 ((key[0], key[1], slot), (key[0], key[1], None)))
        for k in cands:
            idxs = self.cfg.get(k)
            if idxs is not None and idx not in idxs:
                return False
        return True

    def read_ok(self, key, slot=None):
        """may this path take wire `key` as its source?

        Same slot argument as write_ok: a wire can serve a reader in the other
        config bank, provided its driver is programmed there too.
        """
        if slot is None:
            return (not self._any_cfg(key) and
                    (key[0], key[1], 0) not in self.rd and
                    (key[0], key[1], 1) not in self.rd and
                    (key[0], key[1], None) not in self.rd)
        if (key[0], key[1], None) in self.rd or key in self.rd:
            return False
        if (key[0], key[1], slot) in self.rd:
            return False
        for s in (slot, None):
            if (key[0], key[1], s) in self.cfg:
                return False
        return True

    def claim(self, path, slots=None):
        """mark everything a path touches; the marks are what release() reverts.

        `slots` is a per-hop config bank list (see path_slots) or None when the
        schedule is not known yet, which claims both banks conservatively.
        """
        marks = []
        for i, ent in enumerate(path):
            key = (ent[0], ent[1])
            idx = ent[2] if len(ent) > 2 else None
            slot = None if slots is None else slots[i]
            if idx is None:
                k3 = (key[0], key[1], slot)
                self.rd[k3] = self.rd.get(k3, 0) + 1
                marks.append((key, idx, slot))
            else:
                k3 = (key[0], key[1], slot)
                d = self.cfg.setdefault(k3, {})
                d[idx] = d.get(idx, 0) + 1
                marks.append((key, idx, slot))
        if len(self) > self.peak:
            self.peak = len(self)
        return marks

    def release(self, marks):
        for m in marks:
            key, idx = m[0], m[1]
            slot = m[2] if len(m) > 2 else None
            if idx is None:
                k3 = (key[0], key[1], slot)
                n = self.rd.get(k3, 0) - 1
                if n > 0: self.rd[k3] = n
                else: self.rd.pop(k3, None)
                continue
            k3 = (key[0], key[1], slot)
            d = self.cfg.get(k3)
            if not d or idx not in d: continue
            if d[idx] > 1: d[idx] -= 1
            else:
                del d[idx]
                if not d: self.cfg.pop(k3, None)

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

ADG_POS = None

def pe_xy(pe_inst):
    """(x, y) of a PE instance / cfg block, cached"""
    global ADG_POS
    if ADG_POS is None:
        ADG_POS = {x['id']: (x.get('x'), x.get('y'))
                   for x in json.load(open(ADG_JSON))['instances']}
    return ADG_POS.get(PEBLK.get(pe_inst, pe_inst), (999, 999))

def path_slots(path, c0):
    """config bank (ctx parity) each hop of `path` is active in.

    `path` is stored destination-first. The value arrives at the consumer in
    context `c0`; a node p registered hops before the consumer sees context
    `c0 - p`, and the GIB bank that must select its source is `(c0 - p) & 1`.
    Only registered hops advance the context (exe_ii = 2 cycles each).
    """
    slots, par = [], 0
    for ent in path:
        slots.append((c0 - par) & 1)
        if is_reg(ent[0], ent[1]):
            par = (par + 1) % 4
    return slots

def route_parity(src_inst, dst_inst, port, want, occ=(), c0=None):
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
    # State = (wire, delay mod 4). Visiting a wire once per *parity* is what gives
    # the router its delay freedom: with a single visit per wire, a pair whose
    # only free paths all share one residue has no route at all, which is what
    # stopped the 20-node DFGs. Two visits cost nothing on the wire, because a
    # simple path configures it once; the returned path is checked to be simple,
    # so a walk can never ask for two mux configs on the same wire (the old
    # self-contradicting cin bug).
    q = deque([(start, 0, [])]); seen = {(start, 0)}
    MAX_HOPS = 24
    while q:
        (g, p), par, pa = q.popleft()
        if (g, p) in terms and pa and par % 4 == want:
            keys = [(ent[0], ent[1]) for ent in pa]
            if len(keys) == len(set(keys)):
                return pa
            continue
        if len(pa) >= MAX_HOPS:
            continue
        if pa and occ.strict and (g, p) in occ: continue
        npar = (par + (1 if is_reg(g, p) else 0)) % 4
        slot = None if c0 is None else (c0 - par) & 1
        for kind, gib, dstport, idx, src in r.preds(g, p):
            if kind == 'mux':
                if not occ.write_ok((gib, dstport), idx, slot): continue
            elif pa and not occ.read_ok((g, p), slot):
                continue
            node = src if kind == 'track' else (gib, src)
            st = (node, npar)
            if st in seen: continue
            seen.add(st)
            q.append((node, npar, pa + [(gib, dstport, idx)]))
    return None

def route_exact(src_inst, dst_inst, port, target, occ=(), c0=None):
    """route whose CYCLE delay equals target exactly.

    route_parity fixes only the residue mod 4, which is enough for the context
    rotation but not for a node's operands: both operands of a PE enter in the
    same cycle, so their path delays must be equal, not congruent. Routing the
    shorter one at the longer one's delay is what removes the sample skew the
    ramp testbench measures.
    """
    occ = as_occ(occ)
    tgt = IW['pe_in'][str(dst_inst)].get(str(port))
    if tgt is None: return None
    terms = set(tuple(p) for p in IW['pe_out'][str(src_inst)])
    start = tuple(tgt)
    q = deque([(start, 0, [])]); seen = {(start, 0)}
    while q:
        (g, p), d, pa = q.popleft()
        if (g, p) in terms and pa and d == target:
            keys = [(ent[0], ent[1]) for ent in pa]
            if len(keys) == len(set(keys)):
                return pa
            continue
        if len(pa) >= 24 or d > target: continue
        if pa and occ.strict and (g, p) in occ: continue
        nd = d + (1 if is_reg(g, p) else 0)
        if nd > target: continue
        slot = None if c0 is None else (c0 - d) & 1
        for kind, gib, dstport, idx, src in r.preds(g, p):
            if kind == 'mux':
                if not occ.write_ok((gib, dstport), idx, slot): continue
            elif pa and not occ.read_ok((g, p), slot):
                continue
            node = src if kind == 'track' else (gib, src)
            st = (node, nd)
            if st in seen: continue
            seen.add(st)
            q.append((node, nd, pa + [(gib, dstport, idx)]))
    return None

def iob_stream_route(dst_inst, port, avoid=(), occ=(), delay=None):
    """route IOB(out) -> PE in: BFS upstream from PE in pin to any unused IOB out pin.

    delay: required registered-hop delay, or None for any. A stream pin is
    sampled in the same cycle as the node's other operands, so when the node
    also takes a routed edge the stream must take the same number of hops --
    otherwise the multiply reads sample j from the stream and j-1 from the edge
    (dfg_quad measured exactly that).
    """
    occ = as_occ(occ)
    tgt = IW['pe_in'][str(dst_inst)].get(str(port))
    if tgt is None: return None
    terms = {}
    for n in range(32):
        for o in IW['iob_out'].get(str(n), []):
            terms[tuple(o)] = n
    q = deque([(tuple(tgt), 0, [])]); seen = {(tuple(tgt), 0)}
    while q:
        (g, p), d, pa = q.popleft()
        if ((g, p) in terms and pa and terms[(g, p)] not in avoid
                and (delay is None or d == delay)):
            return terms[(g, p)], pa
        if pa and occ.strict and (g, p) in occ: continue
        for kind, gib, dstport, idx, src in r.preds(g, p):
            if kind == 'mux':
                if not occ.write_ok((gib, dstport), idx): continue
            elif pa and not occ.read_ok((g, p)):
                continue
            nd = d + (1 if is_reg(g, p) else 0)
            node = (gib, src) if kind != 'track' else src
            if (node, nd) not in seen:
                seen.add((node, nd))
                q.append((node, nd, pa + [(gib, dstport, idx)]))
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
OPND_ASSIGN = {}       # edge id -> operand mux actually wired (may be swapped)

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
# Capture window and store size are parameters, not constants, so the timing
# model can be fitted from controlled runs (docs/EXECUTION_MODEL.md §3).
CAP_CYCLES = int(os.environ.get('S2_CAP_CYCLES', '64'))
STORE_BYTES = int(os.environ.get('S2_STORE_BYTES', str(CAP_CYCLES * 4)))
# Number of execute() rounds on the SAME configuration. K > 1 amortises the
# launch cost (config + input DMA) and is the only way this flow can measure a
# per-iteration throughput before real II scheduling exists.
REPEAT = max(1, int(os.environ.get('S2_REPEAT', '1')))

def iob_rows(n, is_store, mux):
    iw = iob_cfg({'BaseAddr': (n % 16) * 1024, 'Stride0': 1,
                  'Cycles0': CAP_CYCLES, 'II': 1, 'IsStore': is_store,
                  'Muxn': mux})
    blk_io = n + 1 if n < 16 else n + 291
    return [((blk_io << 3) | i, iw[i]) for i in range(4)]

# ==== mapping + emission with retries ====
placement = None
EMIT_OK = False
for attempt in range(int(os.environ.get('S2_ATTEMPTS', '12'))):
    mapping = Mapping(dfg, adg, ops, False)
    mapping.use_true_reward()
    # Retry diversity: a near-greedy policy (temp 1e-2) reproduces almost the
    # same placement every attempt, so the 12-attempt budget used to re-fail on
    # the same edge. Escalate the sampling temperature instead.
    TEMP = (1e-2, 1e-2, 5e-2, 5e-2, 0.15, 0.15, 0.35, 0.35, 0.6, 0.6, 1.0, 1.0)
    lat = mapping.test_mapping(agent, is_shown=False, temp=TEMP[min(attempt, len(TEMP) - 1)])
    if lat is None:
        if STATS_ON:
            FAIL_REASONS['mapping failed'] = FAIL_REASONS.get('mapping failed', 0) + 1
        continue
    cfg = mapping.state.build_config_dict()
    placement = {int(nid) - 1: attr['adgNode']
                 for nid, attr in cfg['dfg_node_attrs'].items()}

    # Locality pass: the mapper optimises latency, not wirelength, and long
    # routes are what congest the interconnect (a 24-node DAG dead-ends on a
    # 9-hop edge whose destination pins are all free). Anneal the placement for
    # total Manhattan edge length with the PE set fixed.
    def _locality(pl, iters=int(os.environ.get('S2_LOCALITY_ITERS', '12000'))):
        pos = {}
        for x in json.load(open(ADG_JSON))['instances']:
            pos[x['id']] = (x.get('x'), x.get('y'))
        # only compute nodes that the mapper put on a PE may swap: moving an
        # INPUT/CONST off its IOB would change it from a stream to a PE node
        pe_nodes = [n for n in pl
                    if pl[n] in PEBLK_INV and op_of.get(n) not in ('INPUT', 'CONST')]
        if len(pe_nodes) < 3:
            return pl
        adj = [(e.tail, e.head) for e in EDGES
               if e.tail in pl and e.head in pl
               and pl[e.tail] in pos and pl[e.head] in pos]

        def cost(p):
            return sum(abs(pos[p[a]][0] - pos[p[b]][0]) +
                       abs(pos[p[a]][1] - pos[p[b]][1]) for a, b in adj)
        cur = cost(pl)
        rnd = random.Random(len(pe_nodes) * 7919)
        for i in range(iters):
            a, b = rnd.sample(pe_nodes, 2)
            pl[a], pl[b] = pl[b], pl[a]
            new = cost(pl)
            if new <= cur:
                cur = new
            else:
                pl[a], pl[b] = pl[b], pl[a]
        return pl

    # off by default: it reduces total wirelength but does not yet fix the
    # 24-node last-hop congestion, and the mapper's own placement passes the
    # 20-node case, so this stays an experiment until it is measured better
    if os.environ.get('S2_LOCALITY', '0') == '1':
        placement = _locality(dict(placement))

    # Move each stream's IOB to the one nearest its consumer. The mapper picks
    # the IOB position blindly, and a long IOB route is what turns into
    # 'no const IOB' once the compute graph has claimed the neighbourhood.
    if os.environ.get('S2_IOB_NEAR', '1') == '1':
        _pos = {x['id']: (x.get('x'), x.get('y'))
                for x in json.load(open(ADG_JSON))['instances']}
        _used = {v for v in placement.values() if v is not None}
        _iobs = list(range(1, 17)) + list(range(307, 323))
        for _nid, _adg in list(placement.items()):
            if _adg is None or not (_adg < 16 or _adg >= 307):
                continue
            _cons = [e.head for e in EDGES if e.tail == _nid]
            if not _cons:
                continue
            _cx, _cy = _pos.get(placement[_cons[0]], (999, 999))
            _best, _bd = None, 1 << 30
            for _iob in _iobs:
                if _iob in _used and _iob != _adg:
                    continue
                _x, _y = _pos.get(_iob, (999, 999))
                _d = abs(_x - _cx) + abs(_y - _cy)
                if _d < _bd:
                    _best, _bd = _iob, _d
            if _best is not None and _best != _adg:
                _used.discard(_adg)
                _used.add(_best)
                placement[_nid] = _best
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
        # S2_TOPO changes the DFS neighbour order, which yields a different but
        # still topological schedule: congestion depends on which long edge
        # claims the shared wires first, so trying a second order per placement
        # is cheap (mapping only, no new placement).
        _topo = int(os.environ.get('S2_TOPO', '0'))
        def _neigh(nid):
            ps = preds[nid]
            if _topo == 1:
                return list(reversed(ps))
            if _topo == 2:
                return sorted(ps)
            return ps

        def visit(nid):
            if nid in visd: return
            visd.add(nid)
            for p_ in _neigh(nid): visit(p_)
            order.append(nid)
        for nid in node_by_id: visit(nid)
        route_of = {}
        ROUTED = []      # committed edges, newest last: the rip-up pool
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
            # Operands of one node compete for the same pins and the same last-hop
            # wires, so the order they are routed in decides whether both fit. If
            # the first order dead-ends, release this node's own claims and retry
            # hardest-first (longest source-to-destination distance): the cheap half
            # of a rip-up, since nothing outside this node is touched.
            def dist_of(e):
                a = PEBLK.get(PEBLK_INV[placement[e.tail]], placement[e.tail])
                b = placement[nid]
                return abs(a - b) if isinstance(a, int) and isinstance(b, int) else 0

            def route_operands(seq_, c0e=None):
                """route in the given order; returns the failing edge or None."""
                for e in seq_:
                    if is_stream(e.tail):
                        continue  # IOB stream: constant present every cycle
                    opnd = e.operand if e.operand in (0, 1) else 0
                    opnd_of[e.id] = opnd
                    OPND_ASSIGN[e.id] = opnd
                    src_pe = PEBLK_INV[placement[e.tail]]
                    slots = None
                    path = in_p = None
                    if SLOT_OCC:
                        # parity-first routing: pick the wanted config bank from
                        # an estimated launch context, so the claim is slot-exact
                        # and this edge can share wires with earlier ones
                        want = ((c0_est if c0e is None else c0e)
                                - ctx_of[e.tail]) % 4
                        for cand in group_ports(di, opnd):
                            if cand in port_used.get(di, ()):
                                continue
                            p2 = route_parity(src_pe, di, cand, want,
                                              occ=occ,
                                              c0=(c0_est if c0e is None else c0e))
                            if p2:
                                in_p, path = cand, p2
                                slots = path_slots(p2, c0_est if c0e is None else c0e)
                                break
                    if path is None:
                        # no parity path (or the feature is off): fall back to the
                        # slot-agnostic route, claimed conservatively
                        in_p, path = route_operand(src_pe, di, opnd,
                                                   prefer=port_of.get(e.id),
                                                   occ=occ,
                                                   used_ports=port_used.get(di, ()))
                        slots = None
                    if not path:
                        return e
                    port_used.setdefault(di, set()).add(in_p)
                    hs[e.id] = path_delay(path); route_of[e.id] = path
                    port_of[e.id] = in_p
                    # claim immediately: sibling edges of this same node must not
                    # be allowed to reuse a mux this edge already drives
                    claimed[e.id] = occ.claim(path, slots)
                return None

            seq = [e for e in ins if not is_stream(e.tail)]
            # provisional launch context: every operand arrives after at least one
            # hop; the settle() step refines it once the real delays are known
            c0_est = max([ctx_of[e.tail] + 1 for e in seq] + [1])
            pins_before = set(port_used.get(di, ()))

            def rewind():
                """undo this node's routing so another operand order can be tried"""
                for e in seq:
                    occ.release(claimed.pop(e.id, []))
                    hs.pop(e.id, None); route_of.pop(e.id, None)
                    port_of.pop(e.id, None)
                if pins_before: port_used[di] = set(pins_before)
                else: port_used.pop(di, None)

            def align_operands(seq_):
                """make every operand of this node arrive in the same cycle.

                A PE samples all of its input pins at once, so the operands must
                have equal path delay. The router otherwise picks each delay
                freely, and the node silently multiplies sample j by sample j-1
                (measured on dfg_quad: skew -1). Tries the current maximum delay,
                then one more cycle.
                """
                if not ALIGN_OPNDS or len(seq_) < 2:
                    return None
                snap = (copy.deepcopy(occ.cfg), copy.deepcopy(occ.rd))
                marks = {e.id: claimed.get(e.id, []) for e in seq_}
                keep = {e.id: (hs[e.id], route_of[e.id]) for e in seq_}
                base_t = max(hs[e.id] for e in seq_)
                for t in (base_t, base_t + 1):
                    for e in seq_:
                        occ.release(claimed.get(e.id, [])); claimed[e.id] = []
                    ok = True
                    for e in seq_:
                        p = route_exact(PEBLK_INV[placement[e.tail]], di,
                                        port_of[e.id], t, occ=occ, c0=c0_use)
                        if p is None:
                            ok = False
                            break
                        claimed[e.id] = occ.claim(
                            p, path_slots(p, c0_use) if SLOT_OCC else None)
                        hs[e.id] = path_delay(p); route_of[e.id] = p
                    if ok:
                        return None
                    occ.cfg = copy.deepcopy(snap[0])
                    occ.rd = copy.deepcopy(snap[1])
                    for e in seq_:
                        claimed[e.id] = marks[e.id]
                        hs[e.id], route_of[e.id] = keep[e.id]
                return seq_[0]

            # A (src, dst) pair only accepts paths whose registered-hop delay hits
            # one residue mod 4. When that palette is empty the whole node must be
            # launched in another context; try the four offsets before giving up.
            fail = None
            c0_use = c0_est
            for c0_use in ([c0_est + s for s in (0, 1, 2, 3)] if SLOT_OCC
                           else [c0_est]):
                fail = route_operands(seq, c0_use)
                if fail is None:
                    break
                rewind()
            if fail is not None and RIPUP and len(seq) > 1:
                rewind()
                fail = route_operands(sorted(seq, key=dist_of, reverse=True), c0_use)
            if (fail is not None and RIPUP and len(seq) == 2
                    and op_of[nid] in ('ADD', 'MUL')):
                # ADD/MUL are commutative: swapping the operands moves one edge
                # from its operand mux group to the other, which is often the
                # only free set of input pins. Semantics are unchanged.
                rewind()
                for e in seq:
                    opnd_of[e.id] = 1 - (e.operand if e.operand in (0, 1) else 0)
                    OPND_ASSIGN[e.id] = opnd_of[e.id]
                print('swap operands %s (%s)' % (nid, op_of[nid]))
                fail = route_operands(seq)
            def try_cross_ripup(e):
                """release recent edges that block `e`, re-route them unchanged.

                Only edges whose delay can be reproduced exactly are accepted, so
                every context assigned so far remains valid (no silent timing
                change). Rolls back completely when an attempt fails.
                """
                # Candidates: the most recent k edges, plus the edges whose
                # destination is spatially near this one — the block is almost
                # always in the destination's last-hop neighbourhood, which the
                # "recent k" heuristic only hits by luck.
                sets = [ROUTED[-k:] for k in (1, 2, 3, 4, 6, 8, 12, 16, 20)
                        if len(ROUTED) >= k]
                dx, dy = pe_xy(di)
                near = [v for v in ROUTED
                        if v['dst'] != di and
                        abs(pe_xy(v['dst'])[0] - dx) + abs(pe_xy(v['dst'])[1] - dy) <= 2]
                if near:
                    sets.append(near[-6:])
                for victims in sets:
                    snap = (copy.deepcopy(occ.cfg), copy.deepcopy(occ.rd))
                    marks = [v['claimed'] for v in victims]
                    for v in victims:
                        occ.release(v['claimed'])
                        v['claimed'] = []
                    rewind()
                    ok = route_operands(seq, c0_use) is None
                    if ok:
                        good = True
                        for v in victims:
                            p = route_parity(v['src'], v['dst'], v['port'],
                                             v['hs'] % 4, occ=occ, c0=v['c0'])
                            # The delay must be reproduced exactly: NO_PHASE drops
                            # the context constraint, but two operands of one node
                            # still have to arrive in the same cycle, so changing a
                            # victim's delay silently misaligns them (measured:
                            # nn2x6 mapped to [0,1,2] instead of [8,4,7,5,4,7]).
                            if p is None or path_delay(p) != v['hs']:
                                good = False
                                break
                            occ.release(v['claimed'])
                            v['claimed'] = occ.claim(
                                p, path_slots(p, v['c0']) if SLOT_OCC else None)
                            route_of[v['eid']] = p
                        if good:
                            print('ripup %d edge(s) for %d->%d'
                                  % (len(victims), e.tail, e.head))
                            return None
                    occ.cfg, occ.rd = snap
                    for v, m in zip(victims, marks):
                        if v['claimed'] != m:
                            occ.release(v['claimed'])
                        v['claimed'] = m
                    rewind()
                return e

            if fail is None:
                fail = align_operands(seq)
                if fail is not None:
                    rewind()

            if fail is not None and CROSS_RIPUP:
                fail = try_cross_ripup(fail)
            if fail is not None:
                assert False, 'no route %d->%d op%d %s' % (
                    fail.tail, fail.head, fail.operand, LAST_ROUTE_FAIL or '{}')
            base_c0 = max([ctx_of[e.tail] + hs[e.id] for e in ins if e.id in hs] + [1])
            # snapshot of the unconstrained routing: every bump level restarts here
            base = {e.id: (port_of[e.id], route_of[e.id], hs[e.id])
                    for e in ins if e.id in hs}
            base_pins = {port_of[e.id] for e in ins if e.id in hs}

            if SLOT_OCC:
                # second pass: the first pass had no context, so its claims are
                # slot-agnostic. Now that base_c0 is known, re-route every
                # operand with its parity and re-claim per config bank; only
                # then can a later node legally share the other bank.
                for e in ins:
                    if e.id not in hs or is_stream(e.tail):
                        continue
                    want = (base_c0 - ctx_of[e.tail]) % 4
                    occ.release(claimed.pop(e.id, []))
                    p2 = route_parity(PEBLK_INV[placement[e.tail]], di,
                                      port_of[e.id], want, occ=occ, c0=base_c0)
                    if p2 is None:
                        claimed[e.id] = occ.claim(route_of[e.id])
                        continue
                    hs[e.id] = path_delay(p2); route_of[e.id] = p2
                    claimed[e.id] = occ.claim(p2, path_slots(p2, base_c0))
                base = {e.id: (port_of[e.id], route_of[e.id], hs[e.id])
                        for e in ins if e.id in hs}

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
                        if hs[e.id] % 4 != want and not NO_PHASE:
                            # release own claim so the search may reuse its own wiring
                            occ.release(claimed[e.id]); claimed[e.id] = []
                            p2 = route_parity(PEBLK_INV[placement[e.tail]], di,
                                              port_of[e.id], want, occ=occ, c0=c0)
                            if p2 is None:
                                # pin dead-ends under the parity+occ constraints: move
                                # this operand to another free pin of the same mux group
                                for cand in group_ports(di, opnd_of[e.id]):
                                    if cand == port_of[e.id]: continue
                                    if cand in port_used.get(di, ()): continue
                                    q = route_parity(PEBLK_INV[placement[e.tail]], di,
                                                     cand, want, occ=occ, c0=c0)
                                    if not q: continue
                                    port_used[di].discard(port_of[e.id])
                                    port_used[di].add(cand)
                                    port_of[e.id] = cand
                                    p2 = q
                                    break
                            if not p2:
                                return False, c0, e
                            hs[e.id] = path_delay(p2); route_of[e.id] = p2
                            claimed[e.id] = occ.claim(
                                p2, path_slots(p2, c0) if SLOT_OCC else None)
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
            for e in ins:
                if e.id in hs:
                    ROUTED.append({'eid': e.id,
                                   'src': PEBLK_INV[placement[e.tail]],
                                   'dst': di, 'port': port_of[e.id],
                                   'c0': c0, 'hs': hs[e.id],
                                   'claimed': claimed[e.id]})
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
        sdly = {}   # consumer -> stream path delay, so sibling streams agree
        for si, (cnid, val, consumer, operand) in enumerate(iob_streams):
            e0 = next(e for e in EDGES if e.tail == cnid and e.head == consumer)
            opnd = operand if operand in (0, 1) else 0
            dst_inst = PEBLK_INV[placement[consumer]]
            # A stream routed after every compute edge often finds no free path
            # ('no const IOB'). Apply the same rip-up as for compute edges:
            # release recent edges, route the stream, then re-route each victim
            # with its exact delay so no context assignment changes.
            def _try_stream():
                # The stream pin is sampled together with the node's other
                # operands, so it must take the same number of registered hops.
                # Without that the node mixes sample j (stream) with sample j-k
                # (routed edge), which is the skew dfg_quad measures.
                others = [hs[e.id] for e in EDGES
                          if e.head == consumer and e.id in hs]
                tgt = max(others) if others else sdly.get(consumer)
                for cand in group_ports(dst_inst, opnd):
                    if cand in port_used.get(dst_inst, ()): continue
                    if IW['pe_in'][str(dst_inst)].get(str(cand)) is None: continue
                    g = iob_stream_route(dst_inst, cand, used_iob, occ,
                                         delay=tgt if ALIGN_OPNDS else None)
                    if g:
                        return cand, g
                return None, None

            got = None
            cand_sets = [[]] + [ROUTED[-k:] for k in (1, 2, 3, 4, 6, 8)
                                if len(ROUTED) >= k]
            for victims in cand_sets:
                snap = (copy.deepcopy(occ.cfg), copy.deepcopy(occ.rd)) if victims else None
                marks = [v['claimed'] for v in victims]
                for v in victims:
                    occ.release(v['claimed']); v['claimed'] = []
                in_p, got = _try_stream()
                if got:
                    ok = True
                    for v in victims:
                        p = route_parity(v['src'], v['dst'], v['port'],
                                         v['hs'] % 4, occ=occ, c0=v['c0'])
                        if p is None or path_delay(p) != v['hs']:
                            ok = False
                            break
                        occ.release(v['claimed'])
                        v['claimed'] = occ.claim(
                            p, path_slots(p, v['c0']) if SLOT_OCC else None)
                        route_of[v['eid']] = p
                    if ok:
                        if victims:
                            print('stream rip-up: released %d edge(s)' % len(victims))
                        break
                if snap:
                    occ.cfg, occ.rd = snap
                for v, m in zip(victims, marks):
                    if v['claimed'] != m:
                        occ.release(v['claimed'])
                    v['claimed'] = m
                got = None
            assert got is not None, 'no const IOB'
            port_used.setdefault(dst_inst, set()).add(in_p)
            port_of[e0.id] = in_p
            n, pa = got
            sdly.setdefault(consumer, path_delay(pa))
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
                    operand = OPND_ASSIGN.get(
                        e.id, e.operand if e.operand in (0, 1) else 0)
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
        # S2_MAX_CAPS limits how many output channels are read back. The kernel
        # still computes all of them; this isolates the per-channel DMA cost of
        # the measurement, which is ~240 cycles per channel (§3.7).
        MAX_CAPS = int(os.environ.get('S2_MAX_CAPS', '99'))
        for onid in sorted(nid for nid in node_by_id if op_of[nid] == 'OUTPUT'):
            if len(caps) >= MAX_CAPS:
                break
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
        # the input DMA must cover the whole window: with Cycles0 > 64 the old
        # fixed 256 B loaded only half the samples and the kernel read stale
        # bank contents (this is what made CAP=128 produce wrong values)
        # The DMA moves at most 256 B per call (one burst), so a window longer
        # than 64 words must be loaded in several bursts; a single 512 B request
        # silently left the upper half of the bank stale, which is what made
        # CAP=128 compute a missing operand.
        ld = ''.join(
            ''.join('\tload_data((void*)((char*)din_addr[%d] + %d), 0x%x, %d, 0, 0, 0);\n'
                    % (si, off, n * 4096 + off, min(256, CAP_CYCLES * 4 - off))
                    for off in range(0, CAP_CYCLES * 4, 256))
            for si, n, val in din_meta)
        st = ''.join('\tstore(dout_addr[%d], 0x%x, %d, 0, 0);\n'
                     % (i, s * 4096, STORE_BYTES)
                     for i, (s, _) in enumerate(caps))
        # s2_rdcycle brackets each phase so the run log carries the hardware cost
        # of one kernel (docs/EXECUTION_MODEL.md): load = config+input DMA,
        # exec = the execute() instruction, store = capture DMA issue, total =
        # through the fence. caps = capture channels, i.e. 512 B of store each.
        exe = ('#include "cgra_execute.h"\n'
               '#include <stdio.h>\n'
               'static inline unsigned long s2_rdcycle(void) { unsigned long c;\n'
               '  __asm__ volatile("rdcycle %%0" : "=r"(c)); return c; }\n'
               'void cgra_execute0(void** din_addr, void** dout_addr)\n{\n'
               '\tstatic unsigned short cin[%d][3] __attribute__((aligned(64))) = {\n'
               '%s\n\t};\n\n'
               '\tunsigned long t_load0 = s2_rdcycle();\n'
               '\tload_cfg((void*)cin, 0x20000, %d, 0, 0);\n' % (len(allrows), '\n'.join(lines), len(allrows) * 6)
               + ld +
               '\tunsigned long t_load1 = s2_rdcycle();\n'
               '\tconfig(0x0, %d, 0, 0);\n'
               '\tconfig(0x%x, %d, 0, 0);\n' % (len(rowsA), base_b, len(rowsB))
               + '\tunsigned long t_exec0 = s2_rdcycle();\n'
               '\tfor (int r_ = 0; r_ < %d; r_++) { execute(0x%x, 0, 0);'
               ' fence(1); }\n' % (REPEAT, iob_ens)
               + '\tunsigned long t_exec1 = s2_rdcycle();\n'
               + st +
               '\tunsigned long t_store1 = s2_rdcycle();\n'
               '\tfence(1);\n'
               '\tunsigned long t_all = s2_rdcycle();\n'
               '\tprintf("S2_PERF load=%%lu exec=%%lu store=%%lu total=%%lu caps=%d'
               ' it=%d\\n",\n'
               '\t       t_load1 - t_load0, t_exec1 - t_exec0, t_store1 - t_exec1,\n'
               '\t       t_all - t_load0);\n' % (len(caps), REPEAT)
               + '}\n')
        open(OUT_DIR + '/cgra_execute.c', 'w').write(exe)
        # generate matching main (one const array per stream). 64B alignment keeps
        # the CPU fill loop from emitting misaligned vector stores, which make the
        # TL monitor fire "PutPartial contains invalid mask" on buffer_1
        mn = '#include <stdio.h>\n#include <stdint.h>\n'
        mn += 'extern void cgra_execute0(void** din_addr, void** dout_addr);\n'
        for i in range(len(caps)):
            mn += ('int O%d[%d] __attribute__((aligned(64)));\n'
                   % (i, max(128, CAP_CYCLES)))
        for si, n, val in din_meta:
            mn += ('int c%d[%d] __attribute__((aligned(64)));\n'
                   % (si, max(64, CAP_CYCLES)))
        mn += 'int main(void) {\n'
        # S2_RAMP_ONE=k ramps a single stream and leaves the others constant:
        # the output then changes by w_k per sample, so the residual of that
        # channel says which operand carries the sample skew and by how much.
        _ramp_one = int(os.environ.get('S2_RAMP_ONE', '-1'))
        for si, n, val in din_meta:
            _ramp = (os.environ.get('S2_INPUT_RAMP', '0') == '1'
                     and (_ramp_one < 0 or si == _ramp_one))
            if _ramp:
                # per-sample-varying stimulus: a change that shifts *when* a
                # value arrives can still pass with constant inputs, so timing
                # changes (re-scheduling, delay insertion) need this to be
                # validated at all
                mn += ('  for (int i = 0; i < %d; i++) c%d[i] = %d + i;\n'
                       % (max(64, CAP_CYCLES), si, val))
            else:
                mn += ('  for (int i = 0; i < %d; i++) c%d[i] = %d;\n'
                       % (max(64, CAP_CYCLES), si, val))
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
