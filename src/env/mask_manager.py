"""ADG/DFG mask management for placement and routing constraints."""

import logging
from typing import Any, Dict, List

import numpy as np

from enums import IO_OPS, NODE_TYPE_GIB, NODE_TYPE_GPE, NODE_TYPE_IOB

logger = logging.getLogger(__name__)


class MaskManager:
    """Manages placement (mask, dfg_mask) and routing (route_mask) masks.

    Responsible for initialization, updates during actions, and edge
    precomputation used by routing.
    """

    def __init__(self, adg: Any, dfg: Any):
        n_nodes = max(adg.getNodeNums(), len(dfg.getNodes()))
        max_id = adg.getMaxNodeId()

        self.mask = np.ones((n_nodes, max_id + 1))
        self.dfg_mask = np.zeros((n_nodes, max_id + 1))
        self.route_mask = np.zeros(len(adg.getEdges()))

        self._adg = adg
        self._dfg = dfg
        self.adg_actions = max_id + 1
        self._edge_count = len(adg.getEdges())

        # Precompute edge mappings
        self._src_to_edges: Dict = {}
        self._dst_to_src: Dict = {}
        edges = adg.getEdges()
        for eid, edge in edges.items():
            src, dst = edge.getSrcId(), edge.getDstId()
            self._src_to_edges.setdefault(src, []).append((eid, dst))
            self._dst_to_src.setdefault(dst, []).append((eid, src))

    # ========================================================================
    # Initialization
    # ========================================================================

    def init_mask(self, dfg_features_vec: np.ndarray, dfg_adj: np.ndarray,
                  place_cur_node: List[int]) -> List[int]:
        """Initialize placement masks. Returns updated place_cur_node."""
        self.dfg_mask.fill(0)
        place_cur_node.clear()

        # Nodes with no incoming edges and feature[6] == -1 can be placed.
        # A self-edge (tail == head) is a *recurrence*: its producer is the node
        # itself one iteration earlier, so it must NOT count as an unplaced
        # predecessor -- otherwise a loop can never be placed at all (measured:
        # dfg_acc's ADD stays unready forever and the first mapping attempt ends
        # with actions=0).
        adj = dfg_adj.copy()
        np.fill_diagonal(adj, 0)
        zero_incoming = np.where(
            (np.all(adj == 0, axis=0)) &
            (dfg_features_vec[:, 6] == -1)
        )[0]
        self.dfg_mask[zero_incoming, :] = 1
        place_cur_node.extend(zero_incoming.tolist())

        # GIB nodes cannot host DFG nodes
        gib_ids = [nid for nid, n in self._adg.getNodes().items()
                   if n.getType() == NODE_TYPE_GIB]
        self.mask[:, gib_ids] = 0

        # IO nodes can only map to IOB, not GPE
        io_ids = [n.id for n in self._dfg.getNodes()
                  if n.getOpCode() in IO_OPS]
        gpe_ids = [n.getId() for n in self._adg.getNodes().values()
                   if n.getType() == NODE_TYPE_GPE]
        for dfg_id in io_ids:
            self.mask[dfg_id, gpe_ids] = 0

        # Computation nodes cannot map to IOB
        comp_ids = [n.id for n in self._dfg.getNodes()
                    if n.getOpCode() not in IO_OPS]
        iob_ids = [n.getId() for n in self._adg.getNodes().values()
                   if n.getType() == NODE_TYPE_IOB]
        for dfg_id in comp_ids:
            self.mask[dfg_id, iob_ids] = 0

        # P2-2: FU operation-compatibility mask (CGRA capability constraint).
        # A DFG node may only map to an FU that can execute its opcode; ops
        # supported by no FU (e.g. CONST, which the SA compiler merges into
        # immediates) keep the existing mask to avoid regressions. The ADG's
        # mask is already 1:1 (columns cleared after placement), so IOB port
        # capacity is implicitly enforced.
        op_caps: Dict[str, set] = {}
        for nid, node in self._adg.getNodes().items():
            if node.getType() in (NODE_TYPE_GPE, NODE_TYPE_IOB):
                for op in node.getOperations():
                    op_caps.setdefault(op, set()).add(nid)
        for dfg_node in self._dfg.getNodes():
            op = dfg_node.getOpCode()
            caps = op_caps.get(op) or op_caps.get(op.upper())
            if caps:
                allow = np.zeros(self.adg_actions)
                allow[list(caps)] = 1
                self.mask[dfg_node.id] *= allow

        return place_cur_node

    def init_route_mask(self) -> None:
        self.route_mask.fill(0)

    # ========================================================================
    # Reset
    # ========================================================================

    def reset(self) -> None:
        n_nodes = max(self._adg.getNodeNums(), len(self._dfg.getNodes()))
        max_id = self._adg.getMaxNodeId()
        self.mask = np.ones((n_nodes, max_id + 1))
        self.dfg_mask = np.zeros((n_nodes, max_id + 1))
        self.route_mask = np.zeros(self._edge_count)

    # ========================================================================
    # Placement mask updates
    # ========================================================================

    def clear_placement_rows_cols(self, dfg_id: int, adg_id: int) -> None:
        """Clear the mask row for dfg_id and column for adg_id."""
        self.mask[dfg_id, :] = 0
        self.mask[:, adg_id] = 0
        self.dfg_mask[dfg_id, :] = 0

    def activate_ready_nodes(self, dfg_adj: np.ndarray,
                             dfg_features_vec: np.ndarray) -> None:
        """Activate mask for DFG nodes that have all predecessors placed.

        Self-edges are ignored for the same reason as in init_mask: a recurrence
        is satisfied by the node's own previous iteration.
        """
        adj = dfg_adj.copy()
        np.fill_diagonal(adj, 0)
        ready = np.where(
            (np.all(adj <= 0, axis=0)) &
            (dfg_features_vec[:, 6] == -1)
        )[0]
        self.dfg_mask[ready, :] = 1

    # ========================================================================
    # Route mask updates
    # ========================================================================

    def update_route_masks(self, in_node: int, out_node: int,
                           adg_adj: np.ndarray) -> None:
        """Update route masks when routing from in_node to out_node."""
        # Outgoing edges from out_node (enable)
        if out_node in self._src_to_edges:
            out_edges = self._src_to_edges[out_node]
            valid = [eid for eid, _ in out_edges if self.route_mask[eid] >= 0]
            self.route_mask[valid] = 1
            for eid, dst in out_edges:
                if eid in valid:
                    adg_adj[out_node, dst] = 1

        # Outgoing edges from in_node (disable)
        if in_node in self._src_to_edges:
            in_edges = self._src_to_edges[in_node]
            valid = [eid for eid, _ in in_edges if self.route_mask[eid] >= 0]
            self.route_mask[valid] = 0
            for eid, dst in in_edges:
                if eid in valid:
                    adg_adj[in_node, dst] = 0

        # Reverse edges to in_node (disable)
        if in_node in self._dst_to_src:
            rev_edges = self._dst_to_src[in_node]
            valid = [eid for eid, _ in rev_edges if self.route_mask[eid] >= 0]
            self.route_mask[valid] = 0
            for eid, src in rev_edges:
                if eid in valid:
                    adg_adj[src, in_node] = 0

    def activate_route_edges_for_sources(self, source_node_ids: List[int],
                                         edges: Dict[int, Any]) -> None:
        """Activate routing edges originating from the given source ADG nodes."""
        src_ids = np.array([e.getSrcId() for e in edges.values()])
        edge_ids = np.array(list(edges.keys()))
        mask = np.isin(src_ids, source_node_ids)
        self.route_mask[edge_ids[mask]] = 1

    def mark_route_edge_used(self, action: int) -> None:
        """Mark a route edge as used (disable it)."""
        self.route_mask[action] = -1

    # ========================================================================
    # Forward edge lookups
    # ========================================================================

    def get_edges_from(self, node_id: int) -> List[tuple]:
        """Return list of (edge_id, dst_id) from given node."""
        return self._src_to_edges.get(node_id, [])

    def get_edges_to(self, node_id: int) -> List[tuple]:
        """Return list of (edge_id, src_id) to given node."""
        return self._dst_to_src.get(node_id, [])
