"""CGRA mapping environment with state representation and action validation."""

import copy
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    from line_profiler import profile
except ImportError:
    profile = lambda f: f  # no-op

from config import MCTS_CFG, TRAIN_CFG
from env.mask_manager import MaskManager
from env.route_manager import RouteManager
from enums import IO_OPS, NODE_TYPE_GIB, NODE_TYPE_GPE, NODE_TYPE_IOB, SearchMode
from parser import ADGFeatures, DFGFeatures, getAdgEdgeDegrees
from utils import getAdgAdj, getDfgAdj

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

INVALID_ACTION_REWARD = -4000
ROUTE_FAILURE_PENALTY = -10
ROUTE_STEP_COST = -0.1


def softmax_distance(distances: List[int],
                     beta: float = 4.0) -> np.ndarray:
    """Convert distances to a probability distribution via softmax scaling.

    p_i = exp(-beta * d_i) / sum_j exp(-beta * d_j)
    """
    arr = np.array(distances, dtype=float)
    weights = np.exp(-beta * arr)
    return weights / weights.sum()


# ============================================================================
# Environment
# ============================================================================


class Environment:
    """Manages the DFG→ADG mapping simulation state, masks, and graph features."""

    def __init__(self, dfg: Any, adg: Any, operations: Any,
                 is_heuristic: bool = True):
        self.mask_mgr = MaskManager(adg, dfg)
        self.route_mgr = RouteManager()

        self.dfg = dfg
        self.adg = adg
        self.ops = operations

        self.dfgFeatures = DFGFeatures(dfg)
        self.dfg_features = self.dfgFeatures.features
        self.dfg_features_vec = self.dfgFeatures.getFeaturesVector(operations)

        self.adgFeatures = ADGFeatures(adg)
        self.adg_features = self.adgFeatures.features
        self.adg_features_vec = self.adgFeatures.getFeaturesVector()

        self.dfg_adj = getDfgAdj(dfg)
        self.adg_adj = getAdgAdj(adg)

        # P1-2: static one-hot edge-type matrix (used for GNN edge features)
        self.adg_edge_onehot = self.adgFeatures.getEdgeFeatureMatrix()
        # P2-2: cached GIB reged map for the timing-aware route penalty
        self.gib_reged = None

        # P1-6: placement centroid tracking (for resource-spread reward)
        self.placed_count = 0
        self.centroid_x = 0.0
        self.centroid_y = 0.0
        xs = [n.getX() or 0 for n in adg.getNodes().values()]
        ys = [n.getY() or 0 for n in adg.getNodes().values()]
        self.mesh_max_dist = (max(xs) - min(xs)) + (max(ys) - min(ys))
        self.mesh_max_dist = max(self.mesh_max_dist, 1)

        self.init_env = copy.deepcopy((dfg, adg, operations))
        self.reward: Dict[int, float] = {}
        self.adg_actions = self.adg.getMaxNodeId() + 1
        self.heristic_reward = is_heuristic

        self.place_cur_node: List[int] = []
        self.init_mask()
        self.last_action = -1

        self._cur_adg_node_id: Optional[int] = None
        self._cur_dfg_node_id: Optional[int] = None

        self.latency = np.zeros(adg.getMaxNodeId() + 1)
        self.dfg_latency = np.zeros(adg.getMaxNodeId() + 1)

    # ========================================================================
    # Properties delegating to managers (backward-compatible access)
    # ========================================================================

    @property
    def mask(self) -> np.ndarray:
        return self.mask_mgr.mask

    @mask.setter
    def mask(self, value: np.ndarray) -> None:
        self.mask_mgr.mask = value

    @property
    def dfg_mask(self) -> np.ndarray:
        return self.mask_mgr.dfg_mask

    @dfg_mask.setter
    def dfg_mask(self, value: np.ndarray) -> None:
        self.mask_mgr.dfg_mask = value

    @property
    def route_mask(self) -> np.ndarray:
        return self.mask_mgr.route_mask

    @route_mask.setter
    def route_mask(self, value: np.ndarray) -> None:
        self.mask_mgr.route_mask = value

    @property
    def route_cur_node(self) -> List[int]:
        return self.route_mgr.route_cur_node

    @route_cur_node.setter
    def route_cur_node(self, value: List[int]) -> None:
        self.route_mgr.route_cur_node = value

    @property
    def route_cur_edges(self) -> Dict[int, int]:
        return self.route_mgr.route_cur_edges

    @route_cur_edges.setter
    def route_cur_edges(self, value: Dict[int, int]) -> None:
        self.route_mgr.route_cur_edges = value

    @property
    def dfg_links(self) -> defaultdict:
        return self.route_mgr.dfg_links

    @dfg_links.setter
    def dfg_links(self, value: defaultdict) -> None:
        self.route_mgr.dfg_links = value

    @property
    def route_step(self) -> int:
        return self.route_mgr.route_step

    @route_step.setter
    def route_step(self, value: int) -> None:
        self.route_mgr.route_step = value

    # ========================================================================
    # Action handling
    # ========================================================================

    def update(self, action: int, reward: Dict[int, float],
               print_action: bool = False) -> bool:
        """Execute a placement action. Returns True on success."""
        self._set_node_id(action)
        if not self.canMapping(action, reward):
            reward[action] = INVALID_ACTION_REWARD
            return False

        self.updateFeatures()
        self.updateAdj()
        if print_action:
            logger.info("update current action is %d, %d",
                        self._cur_dfg_node_id, self._cur_adg_node_id)
        self.updateMeta(action)
        return True

    def route_update(self, action: int) -> bool:
        """Execute a route action. Delegates core logic to RouteManager."""
        edges = self.adg.getEdges()
        if action not in edges:
            logger.warning("action %d is not in adg edges", action)
            return False

        if self.route_mask[action] <= 0:
            logger.warning("action %d cannot be mapped (mask)", action)
            return False

        edge = edges[action]
        return self.route_mgr.execute_route_step(
            action=action,
            edge=edge,
            mask_mgr=self.mask_mgr,
            target_node=self.last_action % self.adg_actions,
            latency=self.latency,
            dfg_latency=self.dfg_latency,
            adg_adj=self.adg_adj,
            adg_features_vec=self.adg_features_vec,
            last_action=self.last_action,
            adg_actions=self.adg_actions,
        )

    # ========================================================================
    # P2-2: timing-aware routing cost (registered GIB hop penalty)
    # ========================================================================

    def is_registered_hop(self, action: int) -> bool:
        """True when this ADG edge exits a GIB through a REGISTERED output
        port (each such hop adds +1 cycle to the true DFG latency).

        Uses the same reged map as the in-process latency evaluator
        (utils.latency_eval.build_gib_reged), so the RL routing signal
        aligns exactly with the metric being optimized.
        """
        if TRAIN_CFG.registered_hop_penalty <= 0:
            return False
        if self.gib_reged is None:
            from utils.latency_eval import build_gib_reged
            self.gib_reged = build_gib_reged(self.adg)
        edge = self.adg.getEdges().get(action)
        if edge is None:
            return False
        src = edge.getSrcId()
        port = edge.getSrcPortIdx()
        return bool(self.gib_reged.get(src, {}).get(port, False))

    # ========================================================================
    # Mask initialization
    # ========================================================================

    def init_mask(self) -> None:
        """Initialize masks constraining DFG↔ADG node mappings."""
        self.place_cur_node = self.mask_mgr.init_mask(
            self.dfg_features_vec, self.dfg_adj, self.place_cur_node)

    def init_route_mask(self) -> None:
        self.mask_mgr.init_route_mask()

    # ========================================================================
    # Feature and adjacency updates
    # ========================================================================

    def updateFeatures(self) -> None:
        dfg_id = self._cur_dfg_node_id
        adg_id = self._cur_adg_node_id

        self.mask_mgr.clear_placement_rows_cols(dfg_id, adg_id)

        adg_node = self.adg.getNodes()[adg_id]
        if adg_node.getType() == NODE_TYPE_GIB:
            raise ValueError("Cannot map to GIB node!")

        self.adg_features_vec[adg_id][4] = dfg_id
        self.dfg_features_vec[dfg_id][6] = adg_id
        self.dfg_features_vec[dfg_id][3] = len(self.dfg_adj[:, dfg_id])
        self.adg_features_vec[adg_id][2] = adg_node.getNumInputs()
        self.adg_features_vec[adg_id][3] = adg_node.getNumOutputs()

        # P1-6: track running centroid for the resource-spread reward
        self.placed_count += 1
        self.centroid_x += adg_node.getX() or 0
        self.centroid_y += adg_node.getY() or 0

    def updateAdj(self) -> None:
        dfg_id = self._cur_dfg_node_id
        adg_id = self._cur_adg_node_id

        connected = self.dfg_adj[dfg_id] == 1
        self.dfg_adj[dfg_id, connected] = -1
        connected = self.adg_adj[adg_id] == 1
        self.adg_adj[adg_id, connected] = -1

        self.mask_mgr.activate_ready_nodes(self.dfg_adj, self.dfg_features_vec)
        # activate_ready_nodes only sets dfg_mask; the action list is built from
        # place_cur_node, which updateMeta fills from the *successor* row that
        # updateAdj has just rewritten to -1. A node that becomes ready through
        # the adjacency column (e.g. the consumer of a recurrence, whose
        # predecessor is now placed) would therefore never enter the action list
        # -- measured on dfg_acc: dfg_mask[ADD] = 323 but place_cur_node = [].
        for _n in range(self.dfg_mask.shape[0]):
            if self.dfg_mask[_n].sum() > 0 and _n not in self.place_cur_node:
                self.place_cur_node.append(_n)

    def updateMeta(self, action: int) -> None:
        dfg_id = self._cur_dfg_node_id
        self.last_action = action

        pre_adg, pre_dfg = self.find_predecessor_nodes(dfg_id)
        succ = self.find_successors_nodes(dfg_id)

        self.place_cur_node.remove(dfg_id)

        if pre_adg:
            self.route_mgr.route_cur_node = pre_adg
            for pre, pre_node in zip(pre_dfg, pre_adg):
                self.route_mgr.route_cur_edges[pre_node] = self.dfg.findEdge(pre, dfg_id)

            self.mask_mgr.activate_route_edges_for_sources(pre_adg, self.adg.getEdges())

        if succ:
            self.place_cur_node.extend(succ)

    # ========================================================================
    # Legality checks
    # ========================================================================

    def isLegalAction(self, action: int) -> bool:
        dfg_id = action // self.adg_actions
        adg_id = action % self.adg_actions

        if self.mask[dfg_id, adg_id] == 0:
            return False

        adg_node = self.adg.getNode(adg_id)
        node_type = adg_node.getType()
        op_code = self.dfg.getNodes()[dfg_id].getOpCode()

        if node_type == NODE_TYPE_GIB:
            return False
        if node_type == NODE_TYPE_IOB:
            return op_code in IO_OPS
        if node_type == NODE_TYPE_GPE:
            return op_code not in IO_OPS
        return True

    def canMapping(self, action: int, reward: Dict[int, float]) -> bool:
        dfg_id = action // self.adg_actions
        adg_id = action % self.adg_actions

        if self.mask[dfg_id, adg_id] != 1 or self.dfg_mask[dfg_id, adg_id] != 1:
            return False

        sum_cost = self.get_delay(adg_id)

        for src in range(self.dfg_adj.shape[0]):
            if self.dfg_adj[src, dfg_id] == 0:
                continue

            src_pe = self.dfg_features_vec[src][6]
            if src_pe == -1:
                logger.warning("src %d is not mapped", src)
                return False

            dst_pe = adg_id
            # Both heuristic and exact use Manhattan distance as cost heuristic
            path_cost = self.calculateCost(src_pe, dst_pe)

            if path_cost < 0:
                return False
            sum_cost += path_cost

        reward[action] = -sum_cost
        return True

    # ========================================================================
    # Cost / distance
    # ========================================================================

    _cost_cache: Dict[Tuple[int, int], int] = None

    def calculateCost(self, src: int, dst: int) -> int:
        """Manhattan distance between two ADG node coordinates (memoized)."""
        if self._cost_cache is None:
            self._cost_cache = {}
        key = (src, dst)
        if key in self._cost_cache:
            return self._cost_cache[key]
        src_node = self.adg.getNodes()[src]
        dst_node = self.adg.getNodes()[dst]
        cost = (abs(src_node.getX() - dst_node.getX()) +
                abs(src_node.getY() - dst_node.getY()))
        self._cost_cache[key] = cost
        return cost

    def get_delay(self, adg_id: int) -> int:
        mid_type = self.adg_features_vec[adg_id][1]
        if mid_type == 2:
            return 1
        if mid_type == 0:
            return 2
        return 3

    # ========================================================================
    # Graph queries
    # ========================================================================

    def find_predecessor_nodes(self, dfg_id: int) -> Tuple[List[int], np.ndarray]:
        pre_indices = np.where(abs(self.dfg_adj[:, dfg_id]) == 1)[0]
        pre_adg = [int(self.dfg_features_vec[i, 6]) for i in pre_indices]
        return pre_adg, pre_indices

    def find_successors_nodes(self, dfg_id: int) -> List[int]:
        indices = np.abs(self.dfg_adj[dfg_id, :]) == 1
        succ = np.arange(self.dfg_adj.shape[1])[indices]
        col_ready = (self.dfg_adj[:, succ] <= 0).all(axis=0) & (
            self.dfg_features_vec[succ, 6] == -1
        )
        return succ[col_ready].tolist()

    # ========================================================================
    # Action encoding / decoding
    # ========================================================================

    def getAct(self, action: int) -> Tuple[int, int]:
        return action // self.adg_actions, action % self.adg_actions

    def _set_node_id(self, action: int) -> None:
        self._cur_adg_node_id = action % self.adg_actions
        self._cur_dfg_node_id = action // self.adg_actions

    def get_cur_adg_node_id(self) -> Optional[int]:
        return self._cur_adg_node_id

    def get_cur_dfg_node_id(self) -> Optional[int]:
        return self._cur_dfg_node_id

    def get_adg_node_id(self, action: int) -> int:
        return action % self.adg_actions

    def get_dfg_node_id(self, action: int) -> int:
        return action // self.adg_actions

    def get_key(self) -> bytes:
        return self.dfg_features_vec.tobytes()

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def reset(self) -> None:
        self.mask_mgr.reset()
        self.route_mgr.reset()
        self._cost_cache = None  # invalidate

        self.dfgFeatures = DFGFeatures(self.init_env[0])
        self.dfg_features = self.dfgFeatures.features
        self.dfg_features_vec = self.dfgFeatures.getFeaturesVector(self.init_env[2])

        self.adgFeatures = ADGFeatures(self.init_env[1])
        self.adg_features = self.adgFeatures.features
        self.adg_features_vec = self.adgFeatures.getFeaturesVector()

        self.dfg_adj = getDfgAdj(self.init_env[0])
        self.adg_adj = getAdgAdj(self.init_env[1])

        # P1-2: edge features are static (types do not change on reset)
        self.adg_edge_onehot = self.adgFeatures.getEdgeFeatureMatrix()

        # P1-6: reset placement centroid
        self.placed_count = 0
        self.centroid_x = 0.0
        self.centroid_y = 0.0

        self.place_cur_node = []
        self.last_action = -1
        self.init_mask()
        self.init_route_mask()

    # ========================================================================
    # Copy semantics (memory optimization)
    # ========================================================================

    def __deepcopy__(self, memo):
        """Cheap deep copy: share immutable graph structures, copy only the
        mutable simulation state.

        The mapping process never mutates the DFG/ADG/Operations graph
        objects (verified: reged maps stay identical across episodes); it
        only writes to the numpy feature/adjacency arrays and the
        managers. Sharing the graph objects cuts every MappingState copy
        (MCTS per-iteration, collection buffer) from ~1-3 MB to ~100 KB.
        """
        new = type(self).__new__(type(self))
        memo[id(self)] = new

        # Shared (immutable during mapping)
        new.dfg = self.dfg
        new.adg = self.adg
        new.ops = self.ops
        new.init_env = self.init_env
        new.dfgFeatures = self.dfgFeatures
        new.adgFeatures = self.adgFeatures
        new.dfg_features = self.dfg_features
        new.adg_features = self.adg_features
        new.mesh_max_dist = self.mesh_max_dist
        new.adg_actions = self.adg_actions
        new.heristic_reward = self.heristic_reward
        new.gib_reged = self.gib_reged
        new._cost_cache = self._cost_cache

        # Mutable simulation state — copied
        new.mask_mgr = copy.deepcopy(self.mask_mgr, memo)
        new.route_mgr = copy.deepcopy(self.route_mgr, memo)
        new.dfg_features_vec = self.dfg_features_vec.copy()
        new.adg_features_vec = self.adg_features_vec.copy()
        new.dfg_adj = self.dfg_adj.copy()
        new.adg_adj = self.adg_adj.copy()
        new.adg_edge_onehot = self.adg_edge_onehot.copy()
        new.latency = self.latency.copy()
        new.dfg_latency = self.dfg_latency.copy()
        new.place_cur_node = list(self.place_cur_node)
        new.placed_count = self.placed_count
        new.centroid_x = self.centroid_x
        new.centroid_y = self.centroid_y
        new.last_action = self.last_action
        new._cur_adg_node_id = self._cur_adg_node_id
        new._cur_dfg_node_id = self._cur_dfg_node_id
        new.reward = {}
        return new


# ============================================================================
# MappingState
# ============================================================================


class MappingState:
    """Snapshot of the mapping environment used by MCTS tree search."""

    def __init__(self, env: Environment, action: Optional[int] = None):
        self.env = copy.deepcopy(env)
        # mask is a view into env's mask (shared for efficiency)
        self.mask = self.env.mask
        self.route_mask = self.env.route_mask
        self.reward: Dict[int, float] = {}
        self.sum_reward = 0.0
        self.last_reward = 0.0
        # P1-6: accumulated dense shaping (excluded from the final clean reward)
        self.shaping_total = 0.0
        self.mapping_state = True
        self.actions = self.get_actions()
        self.route_actions = self.get_route_actions()
        self.mode = SearchMode.PLACEMENT

    def clone(self) -> 'MappingState':
        """Create a deep copy for MCTS tree search.

        Optimized: avoids re-copying large immutable structures (dfg, adg, ops)
        by cloning only the mutable Environment state.
        """
        new_state = copy.copy(self)
        new_state.env = copy.deepcopy(self.env)
        new_state.env._cost_cache = self.env._cost_cache  # share distance cache
        new_state.mask = new_state.env.mask
        new_state.route_mask = new_state.env.route_mask
        new_state.reward = {}
        # P1-6: copy the shaping accumulator (floats are immutable, += rebinds)
        new_state.shaping_total = self.shaping_total
        # Rebuild mutable copies
        new_state.actions = list(self.actions)
        new_state.route_actions = list(self.route_actions)
        return new_state

    # ========================================================================
    # Terminal / success checks
    # ========================================================================

    def is_terminal(self) -> bool:
        if self.mode == SearchMode.PLACEMENT:
            return (np.all(self.mask <= 0) or
                    len(self.actions) == 0 or
                    not self.mapping_state)
        return (len(self.route_actions) == 0 or
                not self.mapping_state or
                np.all(self.env.route_mask <= 0) or
                len(self.env.route_cur_node) == 0)

    def is_route_success(self) -> bool:
        return len(self.env.route_cur_node) == 0 and self.is_terminal()

    # ========================================================================
    # Available actions
    # ========================================================================

    def get_actions(self) -> List[int]:
        actions = []
        for i in self.env.place_cur_node:
            if np.any(self.mask[i, :] == 1):
                for j in range(self.env.adg_actions):
                    if (self.env.adg.getNode(j) is not None and
                            self.env.mask[i, j] == 1):
                        actions.append(i * self.env.adg_actions + j)
            else:
                logger.warning("No any mask >= 0!")
        return actions

    def get_route_actions(self) -> List[int]:
        if not self.env.route_cur_node:
            return []

        target = self.env.last_action % self.env.adg_actions
        valid = []
        for edge_id, edge in self.env.adg.getEdges().items():
            if self._is_valid_routing_edge(edge_id, edge, target):
                valid.append(int(edge_id))
        return valid

    def get_legal_actions_mask(self, action_dim: int) -> np.ndarray:
        """Return 1D binary mask [action_dim] for the placement policy net.

        Action encoding: action = dfg_id * adg_actions + adg_id.
        """
        flat_mask = np.zeros(action_dim, dtype=np.float32)
        n_dfg = self.env.mask.shape[0]
        for i in range(n_dfg):
            base = i * self.env.adg_actions
            for j in range(self.env.adg_actions):
                if (self.env.mask[i, j] == 1
                        and self.env.adg.getNode(j) is not None):
                    flat_mask[base + j] = 1.0
        return flat_mask

    def get_legal_route_actions_mask(self, action_dim: int) -> np.ndarray:
        """Return 1D binary mask [action_dim] for the routing policy net."""
        route_mask = np.zeros(action_dim, dtype=np.float32)
        n_edges = min(action_dim, len(self.env.adg.getEdges()))
        for edge_id in range(n_edges):
            if edge_id in self.env.route_cur_edges:
                continue
            if self.env.route_mask[edge_id] > 0:
                route_mask[edge_id] = 1.0
        return route_mask

    def get_route_action(self, method: str = "random") -> int:
        actions = self.get_route_actions()
        if not actions:
            return -1 if self.is_terminal() else -2

        distances = [self.get_route_distance(a) for a in actions]
        probs = softmax_distance(distances)

        if method == "greedy":
            return actions[np.argmax(probs)]
        return int(np.random.choice(actions, p=probs))

    def _is_valid_routing_edge(self, edge_id: int, edge: Any,
                                target: int) -> bool:
        if self.env.route_mask[int(edge_id)] != 1:
            return False
        if edge.getSrcId() not in self.env.route_cur_node:
            return False

        dst_type = self.env.adg.getNode(edge.getDstId()).getType()
        return (dst_type == NODE_TYPE_GIB or
                (dst_type in (NODE_TYPE_GPE, NODE_TYPE_IOB) and
                 target == edge.getDstId()))

    # ========================================================================
    # Route completion (used by MCTS during tree search)
    # ========================================================================

    def complete_routing(self) -> bool:
        """Complete all routing steps for current placement. Returns True on success.

        Called by MCTS during tree search to fully evaluate placements that
        require routing.
        """
        self.mode = SearchMode.ROUTE
        while not self.is_terminal():
            act = self.get_route_action()
            if act == -1:
                return True  # routing complete
            if act < 0:
                return False  # routing failed
            self.take_route_action(act)
        self.mode = SearchMode.PLACEMENT
        return True

    # ========================================================================
    # Distance / heuristic
    # ========================================================================

    def get_distance(self, action: int) -> int:
        current = self.env.get_adg_node_id(action)
        dfg_id = self.env.get_dfg_node_id(action)

        pre_adg, _ = self.env.find_predecessor_nodes(dfg_id)
        distance = 0

        if not pre_adg:
            if self.env.last_action == -1:
                return distance
            target = self.env.last_action % self.env.adg_actions
            distance += self.env.calculateCost(current, target)
            return distance

        for pre in pre_adg:
            distance += self.env.calculateCost(pre, current)
        return distance

    def get_route_distance(self, action: int) -> int:
        dst = self.env.adg.getEdges()[action].getDstId()
        last = self.env.get_adg_node_id(self.env.last_action)
        return self.env.calculateCost(last, dst) + 1

    # ========================================================================
    # Action execution
    # ========================================================================

    @profile
    def take_action(self, action: int, print_action: bool = False) -> None:
        self.mode = SearchMode.PLACEMENT
        if print_action:
            logger.info("current action is %d, %d",
                        action // self.env.adg_actions,
                        action % self.env.adg_actions)

        if not self.env.update(action, self.reward, print_action):
            self.mapping_state = False
            self.sum_reward += ROUTE_FAILURE_PENALTY
        elif self.is_terminal() and (
            self.env.dfg_mask.any() or np.all(self.mask > 0)
        ):
            self.mapping_state = False
            self.sum_reward += ROUTE_FAILURE_PENALTY
        else:
            self.last_reward = self.sum_reward
            # P1-6: dense per-step shaping signal for placement quality
            shaping = self._compute_intermediate_reward(action)
            self.shaping_total += shaping
            self.sum_reward += 1 + shaping

        self.env.last_action = action
        self.actions = self.get_actions()
        self.route_actions = self.get_route_actions()

    # ====================================================================
    # P1-6: Dense step-reward shaping
    # ====================================================================

    def _compute_intermediate_reward(self, action: int) -> float:
        """Per-step placement shaping signal (bounded, cheap).

        Three components (weights from ``TRAIN_CFG.step_reward_weights``):
          - comm_penalty:  -w * mean Manhattan distance to already-placed
            predecessors (predicts routing hops; aligned with the -0.1/hop
            ROUTE_STEP_COST scale).
          - affinity_bonus: +w * fraction of predecessors at distance <= 1.
          - spread_bonus:  +w * distance from the running placement centroid
            (normalized by mesh size) to keep PEs spread across the mesh.

        The accumulated total is tracked in ``shaping_total`` and removed from
        the final reported reward (see :meth:`get_clean_reward`) so benchmark
        metrics keep their pre-P1-6 semantics.
        """
        w = TRAIN_CFG.step_reward_weights
        if not w:
            return 0.0
        env = self.env
        dfg_id, adg_id = env.get_dfg_node_id(action), env.get_adg_node_id(action)
        reward = 0.0

        pre_adg, _ = env.find_predecessor_nodes(dfg_id)
        if len(pre_adg) > 0:
            dists = np.array([env.calculateCost(p, adg_id) for p in pre_adg],
                             dtype=np.float64)
            reward += w.get("comm_penalty", 0.0) * (-float(dists.mean()))
            reward += w.get("affinity_bonus", 0.0) * float((dists <= 1).mean())

        if env.placed_count > 1:
            cx = env.centroid_x / env.placed_count
            cy = env.centroid_y / env.placed_count
            node = env.adg.getNodes()[adg_id]
            dist_c = abs((node.getX() or 0) - cx) + abs((node.getY() or 0) - cy)
            reward += w.get("spread_bonus", 0.0) * (dist_c / env.mesh_max_dist)
        return float(reward)

    @profile
    def take_route_action(self, action: int) -> None:
        self.mode = SearchMode.ROUTE
        if not self.env.route_update(action):
            self.mapping_state = False
            self.sum_reward += ROUTE_FAILURE_PENALTY
        elif self.is_terminal():
            if len(self.env.route_cur_node) > 0:
                self.sum_reward += ROUTE_FAILURE_PENALTY
                self.mapping_state = False
            else:
                self.mapping_state = True
                self.sum_reward += ROUTE_STEP_COST
                # P2-2: timing constraint — penalize registered GIB hops
                if self.env.is_registered_hop(action):
                    self.sum_reward -= TRAIN_CFG.registered_hop_penalty
        else:
            self.mapping_state = True
            self.sum_reward += ROUTE_STEP_COST
            # P2-2: timing constraint — penalize registered GIB hops
            if self.env.is_registered_hop(action):
                self.sum_reward -= TRAIN_CFG.registered_hop_penalty
            self.route_actions = self.get_route_actions()

    # ========================================================================
    # Features / metadata
    # ========================================================================

    def get_dfg_meta(self) -> np.ndarray:
        meta = np.zeros_like(self.env.dfg_features_vec)
        if self.env.place_cur_node:
            meta[self.env.place_cur_node] = self.env.dfg_features_vec[
                self.env.place_cur_node
            ]
        return meta

    def get_adg_meta(self) -> np.ndarray:
        meta = np.zeros_like(self.env.adg_features_vec)
        if self.env.route_cur_node:
            meta[self.env.route_cur_node] = self.env.adg_features_vec[
                self.env.route_cur_node
            ]
        return meta

    def get_reward(self) -> float:
        return self.sum_reward

    def get_clean_reward(self) -> float:
        """Final reward excluding P1-6 dense shaping.

        Keeps the benchmark latency metric (``-sum_reward``) comparable to
        pre-shaping runs while the shaping still steers MCTS / PPO during
        training via :meth:`get_reward`.
        """
        return self.sum_reward - self.shaping_total

    # ========================================================================
    # Visualization
    # ========================================================================

    def draw(self, g_visual: Any) -> None:
        g_visual.draw_state(self)

    def start_anime(self, g_visual: Any) -> None:
        g_visual.draw()
        g_visual.setup_plot()
        g_visual.start_anime(self)

    def end_anime(self, g_visual: Any) -> None:
        g_visual.draw_state(self)
        g_visual.end_anime()

    # ========================================================================
    # Configuration dump
    # ========================================================================

    def build_config_dict(self) -> Dict[str, Any]:
        """Build the simulator config (dfg/adg node & edge attrs) as a dict.

        Used by dump_config (file write) and the in-process true-latency
        evaluator (no temp file / subprocess needed).
        """
        config: Dict[str, Any] = {}
        dfg_node_attrs: Dict[int, Dict] = {}
        dfg_edge_attrs: Dict[int, Dict] = {}
        adg_node_attrs: Dict[int, Dict] = {}

        # DFG node attributes
        for node in self.env.dfg.getNodes():
            node_id = node.id
            dfg_node_attrs[node_id + 1] = {
                "minLat": self.env.dfg_latency[node_id],
                "maxLat": self.env.dfg_latency[node_id],
                "lat": self.env.dfg_latency[node_id],
                "adgNode": int(self.env.dfg_features_vec[node_id][6]),
            }

        # DFG edge attributes
        for edge in self.env.dfg.getEdges():
            edge_id = edge.id
            src_id, dst_id = edge.tail, edge.head
            lat = self.env.dfg_latency[dst_id] - self.env.dfg_latency[src_id]
            edge_link_attrs: Dict[int, Dict] = {}

            for e in self.env.dfg_links[edge_id]:
                adg_e = self.env.adg.getEdge(e)
                src_port = adg_e.getSrcPortIdx()
                dst_port = adg_e.getDstPortIdx()
                edge_link_attrs.setdefault(adg_e.getSrcId(), {})[
                    "dstPort"] = src_port
                edge_link_attrs.setdefault(adg_e.getDstId(), {})[
                    "srcPort"] = dst_port

            dfg_edge_attrs[edge_id] = {
                "lat": 0,
                "delay": lat,
                "vio": 0,
                "edgeLinks": dict(edge_link_attrs),
            }

        # ADG node attributes
        for node_id, node in self.env.adg.getNodes().items():
            in_ports: Dict[int, bool] = {}
            out_ports: Dict[int, bool] = {}
            for eid, edge in self.env.adg.getEdges().items():
                if edge.getDstId() == node_id:
                    in_ports[edge.getDstPortIdx()] = True
                if edge.getSrcId() == node_id:
                    out_ports[edge.getSrcPortIdx()] = True

            adg_node_attrs[node_id] = {
                "dfgNode": int(self.env.adg_features_vec[node_id][4]) + 1,
                "inPortUsed": in_ports,
                "outPortUesed": out_ports,
            }

        config = {
            "dfg_node_attrs": dfg_node_attrs,
            "dfg_edge_attrs": dfg_edge_attrs,
            "adg_node_attrs": adg_node_attrs,
        }
        return config

    def dump_config(self, config_path: str = "config.json") -> None:
        config = self.build_config_dict()
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
