import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import copy
from torch.autograd import Variable
from GAT import GraphAttentionLayer
from dfg import DFGFeatures
from adg import ADGFeatures
from loadData import getDfgAdj, getAdgAdj
from visual import GraphVisual
from utils import *


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False
        )

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)

        x = F.dropout(x, self.dropout, self.training)

        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


""" MCT Node definition"""


class MCTNode:
    def __init__(self, prob, distance, parent=None):
        # self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.p = prob
        self.u = 0
        self.q = 0
        self.h = 1 / distance  # heuristic value
        self.policy = None

    def is_fully_expanded(self, state):

        return len(self.children) >= len(state.get_actions()) or self.children == {}

    def is_leaf(self):
        return self.children == {}

    def best_child(self, c_param=5):

        return max(self.children.items(), key=lambda x: x[1].get_value(c_param))

    def get_value(self, c_param):
        self.u = c_param * self.p * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.q + self.u + 10 * self.h

    def expand(self, action_probs, state, mode):
        # tried_actions = [child.state.last_action for action, child in self.children.items()]
        if mode == "placement":
            legal_actions = state.get_actions()
        else:
            legal_actions = state.get_route_actions()
        for action, prob in action_probs:
            if action not in self.children and action in legal_actions:
                # new_state = state.take_action(action)
                distance = 10000000000
                if mode == "placement":
                    distance = state.get_distance(action)
                else:
                    distance = state.get_route_distance(action)
                child = MCTNode(prob, distance + 1, self)
                self.children[action] = child

    def simulate(self, state):
        current_state = copy.deepcopy(state)
        while not current_state.is_terminal():
            current_state.take_ramdom_action()
        ## print("sim finish")
        return current_state.get_reward()

    def backpropagate(self, result):
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(result)
        self.q += 1.0 * (result - self.q) / self.visits


""" MCT definition"""


class MCTS:
    def __init__(self, agent_net, mode="placement"):
        self.root = MCTNode(1.0, 1)
        self.agent_net = agent_net
        self.exploration_weight = 1.4
        self.mode = mode

    def search(self, state, alpha=0.9):

        act, node = self.select(state)
        if self.mode == "placement":
            action_probs, value = self.agent_net.policy_value_fn(state)
        else:
            action_probs, value = self.agent_net.route_policy_value_fn(state)
        node.policy = action_probs
        if not state.is_terminal():
            node.expand(action_probs, state, self.mode)
        else:
            value = state.get_reward()

        # if self.mode == "route":
        #     value = value * alpha + state.get_reward()
        node.backpropagate(value)

    def select(self, state):
        node = self.root
        act = None
        while not node.is_leaf() and not state.is_terminal():
            act, node = node.best_child()
            if self.mode == "placement":
                state.take_action(act)
            else:
                state.take_route_action(act)
        return act, node

    def get_move_probs(self, state, iters, temp=1e-3):
        for i in range(iters):
            state_cp = copy.deepcopy(state)
            self.search(state_cp)

        act_visits = [(act, node.visits) for act, node in self.root.children.items()]
        if act_visits == []:
            return None, 0
        acts, visits = zip(*act_visits)
        visits = np.array(visits)
        act_probs = softmax(1.0 / temp * np.log(visits + 1))
        return acts, act_probs

    def update_with_move(self, last_move):
        """update mcts root with last move"""
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = MCTNode(1.0, 1)


class Environment:
    def __init__(self, dfg, adg, operations, is_heuristic=True):

        self.mask = np.ones((adg.getNodeNums(), adg.getMaxNodeId() + 1))
        self.dfg_mask = np.zeros((adg.getNodeNums(), adg.getMaxNodeId() + 1))
        self.route_mask = np.ones(len(adg.getEdges()))

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
        self.init_env = copy.deepcopy((dfg, adg, operations))
        self.reward = {}
        self.adg_actions = self.adg.getMaxNodeId() + 1
        self.graph_visual = GraphVisual(dfg, adg)
        self.heristic_reward = is_heuristic
        self.init_mask()

        self.route_cur_node = []
        self.place_cur_node = [-1]
        self.route_step = 0

        self._cur_adg_node_id = None
        self._cur_dfg_node_id = None

    def update(self, action, reward, print_action=False):
        min_path = []
        self._set_node_id(action)
        if self.canMapping(action, reward):
            self.updateFeatures()
            self.updateAdj()
            if print_action:
                print(
                    "update current action is {}, {}".format(
                        self._cur_dfg_node_id, self._cur_adg_node_id
                    )
                )

            self.updateMeta(action)
            return True
        else:
            reward[action] = -4000
            return False

    def route_update(self, action):
        inN = None
        outN = None
        if action in self.adg.getEdges():
            inN = self.adg.getEdges()[action].getSrcId()
            outN = self.adg.getEdges()[action].getDstId()
        else:
            print("action is not in adg edges")
            return False

        if inN not in self.route_cur_node:
            if self.route_cur_node.size == 0:
                print("route cur node is empty")
                return True
            print("inN is not equal to route_cur_node")
            return False
        elif self.route_mask[action] == 0:
            print("action is already mapped")
            return False
        else:
            self.route_cur_node.remove(inN)
            reward = self.get_delay(outN)
            if outN != self.place_cur_node[0] % self.adg_actions:
                self.route_cur_node.append(outN)
            else:
                print(
                    "adg success {}".format(self.place_cur_node[0] % self.adg_actions)
                )
                self.adg_adj[inN, outN] = 0
                self.route_mask[action] = 0
                return True
            if self.adg.getNode(outN).getType() != "GIB":
                print("{} not GIB\n".format(outN))
            self.route_step += 1
            self.adg_features_vec[outN][4] = outN
            print("Info# {}".format(outN))
            self.adg_adj[inN, outN] = 0
            self.route_mask[action] = 0
            # print("mask {}".format(action))
            # print("all mask {}".format(np.where(self.route_mask > 0)))

        return True

    def init_mask(self):
        """
        Initialize masks for DFG (Data Flow Graph) and ADG (Architecture Description Graph) mapping.
        The function sets up mapping constraints between DFG nodes and ADG nodes based on their types.

        The mask is a binary matrix where:
        - 0 indicates mapping is not allowed
        - 1 indicates mapping is allowed
        """
        # Initialize DFG mask with zeros
        self.dfg_mask.fill(0)

        # Set mask=1 for nodes with no incoming edges and specific feature
        zero_incoming = np.where(
            (np.all(self.dfg_adj == 0, axis=0)) & (self.dfg_features_vec[:, 6] == -1)
        )[0]
        self.dfg_mask[zero_incoming, :] = 1

        # Get all GIB nodes from ADG and disable their mappings
        gib_nodes = [
            node_id
            for node_id, node in self.adg.getNodes().items()
            if node.getType() == "GIB"
        ]
        self.mask[:, gib_nodes] = 0

        # Get INPUT/OUTPUT nodes from DFG
        io_nodes = [
            node.id
            for node in self.dfg.getNodes()
            if node.getOpCode() in ("INPUT", "OUTPUT")
        ]

        # Get GPE nodes from ADG
        gpe_nodes = [
            node.getId()
            for node in self.adg.getNodes().values()
            if node.getType() == "GPE"
        ]

        # Disable mapping between INPUT/OUTPUT nodes and GPE nodes
        for dfg_id in io_nodes:
            self.mask[dfg_id, gpe_nodes] = 0

        # Get computation nodes (non-INPUT/OUTPUT) from DFG
        comp_nodes = [
            node.id
            for node in self.dfg.getNodes()
            if node.getOpCode() not in ("INPUT", "OUTPUT")
        ]

        # Get IOB nodes from ADG
        iob_nodes = [
            node.getId()
            for node in self.adg.getNodes().values()
            if node.getType() == "IOB"
        ]

        # Disable mapping between computation nodes and IOB nodes
        for dfg_id in comp_nodes:
            self.mask[dfg_id, iob_nodes] = 0

        print("Mask initialization completed successfully")

    def init_route_mask(self):
        self.route_mask = np.ones(len(self.adg.getEdges()))

    def getAct(self, action):
        return action // self.adg_actions, action % self.adg_actions

    # def updateFeatures(self, action):
    #     dfg_act = self._get_dfg_node_id(action)
    #     adg_act = self._get_adg_node_id(action)
    #     self.mask[dfg_act, :] = 0
    #     self.mask[:, adg_act] = 0
    #     self.dfg_mask[dfg_act, :] = 0
    #     # print("update features dfg {} adg {}".format(dfg_act, adg_act))

    #     if self.adg.getNodes()[adg_act].getType() != "GIB":
    #         self.adg_features_vec[adg_act][4] = dfg_act
    #     else:
    #         print("gib error!")
    #     self.dfg_features_vec[dfg_act][6] = adg_act

    #     self.dfg_features_vec[dfg_act][3] = len(self.dfg_adj[:, dfg_act])
    #     self.adg_features_vec[adg_act][2] = self.adg.getNodes()[adg_act].getNumInputs()
    #     self.adg_features_vec[adg_act][3] = self.adg.getNodes()[adg_act].getNumOutputs()
    def updateFeatures(self):
        """
        Update features for both DFG and ADG nodes after an action is taken.

        Args:
            action (int): Combined action value encoding both DFG and ADG nodes

        Updates:
        - Node mappings in feature vectors
        - Input/output counts
        - Masks for both DFG and ADG

        Raises:
            ValueError: If attempting to map to a GIB node
        """
        # Extract node IDs
        dfg_node_id = self._cur_dfg_node_id
        adg_node_id = self._cur_adg_node_id

        # Update mapping masks
        self._update_masks(dfg_node_id, adg_node_id)

        # Update node mappings
        self._update_node_mappings(dfg_node_id, adg_node_id)

        # Update connectivity features
        self._update_connectivity_features(dfg_node_id, adg_node_id)

    def _update_masks(self, dfg_node_id, adg_node_id):
        """
        Update masks to prevent further mapping of used nodes.
        """
        # Disable all mappings for these nodes
        self.mask[dfg_node_id, :] = 0  # Disable row
        self.mask[:, adg_node_id] = 0  # Disable column
        self.dfg_mask[dfg_node_id, :] = 0  # Disable DFG mask row

    def _update_node_mappings(self, dfg_node_id, adg_node_id):
        """
        Update the node mapping information in feature vectors.

        Raises:
            ValueError: If attempting to map to a GIB node
        """
        adg_node = self.adg.getNodes()[adg_node_id]

        # Verify node type and update mappings
        if adg_node.getType() == "GIB":
            raise ValueError("Cannot map to GIB node!")

        # Update bidirectional mapping between DFG and ADG nodes
        self.adg_features_vec[adg_node_id][4] = dfg_node_id  # ADG -> DFG mapping
        self.dfg_features_vec[dfg_node_id][6] = adg_node_id  # DFG -> ADG mapping

    def _update_connectivity_features(self, dfg_node_id, adg_node_id):
        """
        Update input/output connectivity features for both DFG and ADG nodes.
        """
        adg_node = self.adg.getNodes()[adg_node_id]

        # Update DFG node connectivity
        self.dfg_features_vec[dfg_node_id][3] = len(self.dfg_adj[:, dfg_node_id])

        # Update ADG node connectivity
        self.adg_features_vec[adg_node_id][2] = adg_node.getNumInputs()
        self.adg_features_vec[adg_node_id][3] = adg_node.getNumOutputs()

    def updateAdj(self):
        """
        Process an action and update the adjacency matrices and masks accordingly.

        Args:
            action (int): Combined action value encoding both DFG and ADG actions

        The function:
        1. Splits the combined action into DFG and ADG components
        2. Updates adjacency matrices
        3. Updates masks based on new graph state
        """
        # Split combined action into DFG and ADG components
        dfg_action = self._cur_dfg_node_id
        adg_action = self._cur_adg_node_id

        # Update DFG adjacency matrix
        # Find nodes connected to dfg_action node and mark them as processed (-1)
        dfg_connected_nodes = self.dfg_adj[dfg_action] == 1
        self.dfg_adj[dfg_action, dfg_connected_nodes] = -1

        # Update ADG adjacency matrix similarly
        adg_connected_nodes = self.adg_adj[adg_action] == 1
        self.adg_adj[adg_action, adg_connected_nodes] = -1

        # Update masks for nodes with all incoming edges processed
        # Find nodes where all incoming edges are <= 0 and have feature[6] == -1
        ready_nodes = np.where(
            (np.all(self.dfg_adj <= 0, axis=0)) & (self.dfg_features_vec[:, 6] == -1)
        )[0]

        # Enable these nodes in the mask
        self.dfg_mask[ready_nodes, :] = 1

        return self.dfg_mask.copy()

    def isLegalAction(self, action):
        """
        Check if a given action is legal based on DFG and ADG constraints.

        Args:
            action (int): Combined action value encoding both DFG and ADG actions

        Returns:
            bool: True if the action is legal, False otherwise

        Validates:
        1. Mask compatibility
        2. Node type restrictions
        3. Operation code compatibility
        """
        # Decompose action into DFG and ADG components
        dfg_action = action // self.adg_actions
        adg_action = action % self.adg_actions

        # Get node information
        adg_node = self.adg.getNode(adg_action)
        dfg_node = self.dfg.getNodes()[dfg_action]

        # Check mask constraint
        if self.mask[dfg_action, adg_action] == 0:
            return False

        # Check node type constraints
        node_type = adg_node.getType()
        op_code = dfg_node.getOpCode()

        # GIB nodes are not valid targets
        if node_type == "GIB":
            return False

        # IOB nodes can only map to INPUT/OUTPUT operations
        if node_type == "IOB":
            return op_code in ("INPUT", "OUTPUT")

        # GPE nodes cannot map to INPUT/OUTPUT operations
        if node_type == "GPE":
            return op_code not in ("INPUT", "OUTPUT")

        return True

    def canMapping(self, action, reward):
        dfg_act = action // self.adg_actions
        adg_act = action % self.adg_actions
        if self.mask[dfg_act, adg_act] != 1 or self.dfg_mask[dfg_act, adg_act] != 1:
            return False

        # print("test action is {}, {}".format(dfg_act, adg_act))

        if self.mask[dfg_act, adg_act] == 1 and self.dfg_mask[dfg_act, adg_act] == 1:

            # sum_cost = self.adg.getNode(adg_act).getMaxDelay()
            sum_cost = self.get_delay(adg_act)

            for src in range(self.dfg_adj.shape[0]):
                if self.dfg_adj[src, dfg_act] == 0:
                    continue
                src_mapped_pe = self.dfg_features_vec[src][6]
                if src_mapped_pe == -1:
                    print("src {} is not mapped".format(src))
                    return False
                dst_mapped_pe = action % self.adg_actions
                if not self.heristic_reward:
                    # print("src {}, dst {}".format(src_mapped_pe, dst_mapped_pe))
                    cur_path = []
                    cost = 0
                    # path = []
                    min_cost = 100000000
                    path_cost = self.calculateCost(
                        src_mapped_pe, dst_mapped_pe
                    )  # heruistic reward

                    # path_cost = self.findPath(
                    #     src_mapped_pe,
                    #     dst_mapped_pe,
                    #     cost,
                    #     min_cost,
                    #     max_cost * 2,
                    #     cur_path,
                    #     path,
                    #     0,
                    # )
                else:
                    # print("a star reward")
                    path_cost = self.calculateCost(src_mapped_pe, dst_mapped_pe)
                # print("path cost is {}".format(path_cost))
                if path_cost < 0:
                    return False
                sum_cost += path_cost
            # update dfg environment
            # for src in self.dfg_adj[:, action // self.adg_actions]:
            #     # self.dfg_adj[src, action // self.adg_actions] = 0
            #     self.dfg_features[src].out_degree -= 1
            #     self.dfg_features[src].mapped_pe_id = action % self.adg_actions
            # print("sum cost is {}".format(sum_cost))
            reward[action] = -sum_cost
            return True
        return False

    def findPath(self, src, dst, cost, min_cost, max_cost, path, min_path, d):
        # if self.adg_adj[src,dst] == 0:
        #     return -1
        # print("{}: src {} dst {}".format(d, src,dst))
        path_cost = 100000000
        min_src = 0
        min_dst = 0
        if cost > max_cost:
            return -1
        path.append(src)

        if src == dst:
            if cost < min_cost:
                # print("update cost {}".format(cost))
                min_cost = cost
                min_path = path
        else:
            for mid_dst in range(len(self.adg_adj[src, :])):
                if self.adg.getNode(mid_dst) == None:
                    continue

                # print(mid_dst)
                if self.adg.getNode(mid_dst).getType() != "GIB" and mid_dst != dst:
                    continue
                # if mid_dst == dst:
                #     print("{} cur type is {}".format(mid_dst,self.adg.getNode(mid_dst).getType()))
                if self.adg_adj[src, mid_dst] != 0:
                    # temp = port
                    # print("cur type is {}".format(self.adg.getNode(mid_dst).getType()))
                    self.adg_adj[src, mid_dst] = 0
                    self.adg_adj[mid_dst, src] = 0
                    # self.adg_adj[mid_dst, src] = 0

                    mid_delay = self.get_delay(mid_dst)

                    if mid_delay == None:
                        mid_delay = 1
                    path_cost = self.findPath(
                        mid_dst,
                        dst,
                        cost + mid_delay,
                        min_cost,
                        max_cost,
                        path,
                        min_path,
                        d + 1,
                    )

                    # self.adg_adj[src,mid_dst] = 1
                    # self.adg_adj[mid_dst, src] = 1
        path.pop()
        if path_cost == 100000000:
            return -1
        return min_cost

    def calculateCost(self, src, dst):
        src_node = self.adg.getNodes()[src]
        dst_node = self.adg.getNodes()[dst]
        # if src_node.getType() == "GIB" or dst_node.getType() == "GIB":
        #     return 0
        src_x = src_node.getX()
        src_y = src_node.getY()
        dst_x = dst_node.getX()
        dst_y = dst_node.getY()
        ct = 0
        if src_node.getType() != "GIB" and dst_node.getType() != "GIB":
            ct = 1
        return abs(src_x - dst_x) + abs(src_y - dst_y) + ct

    def get_delay(self, adg_act):
        mid_type = self.adg_features[adg_act].type
        if mid_type == 2:
            return 1
        elif mid_type == 0:
            return 2
        else:
            return 3

    def updateMeta(self, action):
        """
        Update metadata after a node mapping action, including:
        - Current placement node tracking
        - Route node predecessors identification

        Args:
            action (int): The current action being processed

        Updates:
        - place_cur_node: Tracks current placement node
        - route_cur_node: List of predecessor nodes that need routing
        """
        # Get current DFG and ADG node IDs
        dfg_node_id = self._cur_dfg_node_id
        adg_node_id = self._cur_adg_node_id

        # Update current placement node
        self.place_cur_node[0] = action

        # Find predecessor nodes that need routing
        predecessor_nodes = self._find_predecessor_nodes(dfg_node_id)

        # Update routing nodes if predecessors exist
        if predecessor_nodes:
            self.route_cur_node = predecessor_nodes

    def _find_predecessor_nodes(self, dfg_node_id):
        """
        Find all predecessor nodes in DFG that have edges to the current node.

        Args:
            dfg_node_id (int): Current DFG node ID

        Returns:
            list[int]: List of ADG node IDs for predecessors that need routing
        """
        # Get indices of nodes that have edges to current node
        predecessor_indices = np.where(self.dfg_adj[:, dfg_node_id] != 0)[0]

        # Map these DFG nodes to their corresponding ADG nodes using feature vector
        predecessor_adg_nodes = [
            self.dfg_features_vec[i, 6] for i in predecessor_indices
        ]

        return predecessor_adg_nodes

    def reset(self):
        self.mask = np.ones((self.adg.getNodeNums(), self.adg.getMaxNodeId() + 1))
        self.dfg_mask = np.zeros((self.adg.getNodeNums(), self.adg.getMaxNodeId() + 1))
        self.route_mask = np.ones(len(self.adg.getEdges()))
        self.dfgDeatures = DFGFeatures(self.init_env[0])
        self.dfg_features = self.dfgDeatures.features

        self.dfg_features_vec = self.dfgDeatures.getFeaturesVector(self.init_env[2])
        self.adgDeatures = ADGFeatures(self.init_env[1])
        self.adg_features = self.adgDeatures.features
        self.adg_features_vec = self.adgDeatures.getFeaturesVector()
        self.dfg_adj = getDfgAdj(self.init_env[0])
        self.adg_adj = getAdgAdj(self.init_env[1])
        self.route_cur_node = []
        self.place_cur_node = [-1]
        self.init_mask()
        self.init_route_mask()

    def draw(self):
        """draw the environment"""
        self.graph_visual.draw()

    def _set_node_id(self, action):
        """
        Extract ADG/DFG node ID from a combined action value.

        Args:
            action (int): Combined action value
        """
        self._cur_adg_node_id = action % self.adg_actions
        self._cur_dfg_node_id = action // self.adg_actions

    def get_cur_adg_node_id(self):
        return self._cur_adg_node_id

    def get_cur_dfg_node_id(self):
        return self._cur_dfg_node_id

    def get_adg_node_id(self, action):
        """
        Extract ADG node ID from a combined action value.

        Args:
            action (int): Combined action value

        Returns:
            int: ADG node ID
        """
        return action % self.adg_actions

    def get_dfg_node_id(self, action):
        """
        Extract DFG node ID from a combined action value.
        Args:
            action (int): Combined action value
        Returns:
            int: DFG node ID
        """
        return action // self.adg_actions


class MappingState:
    def __init__(self, env, action=None):
        super().__init__()
        # fixme: wrong assginment for last action
        self.last_action = env.place_cur_node[0]
        # self.route_cur_node = env.route_cur_node
        self.env = copy.deepcopy(env)
        # self.dfg_mask = {}
        # self.adg_mask = {}
        self.mask = (
            self.env.mask
        )  # np.zeros((len(env.dfg.getNodes()), env.adg_actions))
        self.route_mask = self.env.route_mask

        # self.dfg_features = DFGFeatures(env.dfg)
        # self.dfg_features_vec = self.dfg_features.getFeaturesVector(env.ops)
        # self.adg_features = ADGFeatures(env.adg)
        # self.adg_features_vec = self.adg_features.getFeaturesVector()
        # self.init_mask()
        # self.dfg_adj = getDfgAdj(env.dfg)
        # self.adg_adj = getAdgAdj(env.adg)
        self.reward = {}
        # self.action_map = {}
        self.sum_reward = 0
        self.sum_route_reward = 0
        self.mapping_state = True

        self.mode = "placement"
        # self.plot = plt.subplots()

    def clone(self):
        new_state = copy.deepcopy(self)
        new_state.reward = {}
        return new_state

    def init_mask(self):
        for gib in [
            node.getId() for node in self.env.adg.getNodes() if node.getType() == "GIB"
        ]:
            self.mask[:, gib] = 0

        for dfg_id in [
            node.getId()
            for node in self.env.dfg.getNodes()
            if node.getOpCode() == "INPUT" or node.getOpCode() == "OUTPUT"
        ]:
            for adg_id in [
                node.getId()
                for node in self.env.adg.getNodes()
                if node.getType() == "GPE"
            ]:
                self.mask[dfg_id, adg_id] = 0

        for dfg_id in [
            node.getId()
            for node in self.env.dfg.getNodes()
            if node.getOpCode() != "INPUT" and node.getOpCode() != "OUTPUT"
        ]:
            for adg_id in [
                node.getId()
                for node in self.env.adg.getNodes()
                if node.getType() == "IOB"
            ]:
                self.mask[dfg_id, adg_id] = 0

    def is_terminal(self):
        if self.mode == "placement":
            # print("placement terminal")
            return (
                np.all(self.mask <= 0)
                or (len(self.get_actions()) == 0)
                or self.mapping_state == False
            )
        else:
            # print("route terminal")
            if (
                (len(self.get_route_actions()) == 0)
                or self.mapping_state == False
                or np.all(self.env.route_mask == 0)
            ):
                print("get route actions {}".format(len(self.get_route_actions())))
                print("mapping state {}".format(self.mapping_state))
                print("route mask {}".format(len(self.env.route_cur_node)))
            return (
                (len(self.get_route_actions()) == 0)
                or self.mapping_state == False
                or np.all(self.env.route_mask == 0)
                or len(self.env.route_cur_node) == 0
            )

    def is_route_terminal(self):
        return np.all(self.env.route_mask == 0) or len(self.get_route_actions()) == 0

    def get_actions(self):
        actions = []

        for i in range(len(self.env.dfg_features_vec)):
            if np.all(self.env.dfg_mask[i, :] == 0):
                continue
            if np.any(self.mask[i, :] == 1):
                for j in range(self.env.adg_actions):
                    if self.env.adg.getNode(j) != None and self.env.mask[i, j] == 1:
                        action = i * self.env.adg_actions + j
                        actions.append(action)
        return actions

    # def get_route_actions(self):
    #     #! fixme: wrong implementation
    #     ## fix: distance is not correct, need to calculate the distance between two gpe/iob nodes
    #     actions = []
    #     if len(self.env.route_cur_node) == 0:
    #         return actions
    #     print(self.env.route_cur_node)
    #     for i, edge in self.env.adg.getEdges().items():
    #         if (
    #             self.env.route_mask[int(i)] == 1
    #             and edge.getSrcId() in self.env.route_cur_node
    #             # and edge.getDstId() not in self.env.route_cur_node
    #             and self.env.calculateCost(
    #                 edge.getSrcId(), self.last_action % self.env.adg_actions
    #             )
    #             >= self.env.calculateCost(
    #                 self.last_action % self.env.adg_actions, edge.getDstId()
    #             )
    #             and (
    #                 self.env.adg.getNode(edge.getDstId()).getType() == "GIB"
    #                 or (
    #                     self.env.adg.getNode(edge.getDstId()).getType() == "GPE"
    #                     and self.last_action % self.env.adg_actions == edge.getDstId()
    #                 )
    #             )
    #         ):
    #             actions.append(int(i))
    #     return actions
    def get_route_actions(self):
        """
        Get valid routing actions based on current node positions and edge constraints.
        Note: Implementation needs fixing for correct distance calculations between GPE/IOB nodes.

        Returns:
            list[int]: List of valid edge IDs that can be used for routing
        """
        valid_actions = []

        # Return empty list if no current routing nodes
        if not self.env.route_cur_node:
            return valid_actions

        # print("Current routing nodes:", self.env.route_cur_node)

        # Get the target node ID from last action
        target_node_id = self.last_action % self.env.adg_actions

        # Iterate through all edges to find valid routing options
        for edge_id, edge in self.env.adg.getEdges().items():
            if self._is_valid_routing_edge(edge_id, edge, target_node_id):
                valid_actions.append(int(edge_id))

        return valid_actions

    def _is_valid_routing_edge(self, edge_id, edge, target_node_id):
        """
        Check if an edge is valid for routing based on multiple constraints.

        Args:
            edge_id: ID of the edge being checked
            edge: Edge object containing source and destination information
            target_node_id: ID of the target node from last action

        Returns:
            bool: True if the edge is valid for routing, False otherwise
        """
        # Basic mask check
        if self.env.route_mask[int(edge_id)] != 1:
            return False

        # Source node must be in current routing nodes
        if edge.getSrcId() not in self.env.route_cur_node:
            return False

        # Check distance constraints
        src_cost = self.env.calculateCost(edge.getSrcId(), target_node_id)
        dst_cost = self.env.calculateCost(target_node_id, edge.getDstId())
        if src_cost < dst_cost:
            return False

        # Check destination node type constraints
        dst_node = self.env.adg.getNode(edge.getDstId())

        # Destination must be either:
        # 1. A GIB node, or
        # 2. A GPE node that matches the target node
        return dst_node.getType() == "GIB" or (
            dst_node.getType() == "GPE" and target_node_id == edge.getDstId()
        )

    def get_distance(self, cur_action):
        """
        Calculate distance between current action and last action nodes.

        Args:
            cur_action (int): Current action being considered

        Returns:
            int: Distance between nodes, returns 1 if no last action exists
        """
        if self.last_action is None:
            return 1

        # Extract ADG node IDs from actions
        current_node_id = self.env.get_adg_node_id(cur_action)
        last_node_id = self.env.get_adg_node_id(self.last_action)

        return self.env.calculateCost(last_node_id, current_node_id)

    def get_route_distance(self, cur_action):
        """
        Calculate routing distance from last node to destination of current edge.

        Args:
            cur_action (int): Current routing action (edge ID)

        Returns:
            int: Total routing distance plus 1
        """
        # Get destination node of the current edge
        destination_node = self.env.adg.getEdges()[cur_action].getDstId()

        # Get last node ID
        last_node_id = self.env.get_adg_node_id(self.last_action)

        # Calculate total distance (cost to destination + 1)
        return self.env.calculateCost(last_node_id, destination_node) + 1

    def take_action(self, action, print_action=False):
        self.mode = "placement"
        if print_action:
            print(
                "current action is {}, {}".format(
                    action // self.env.adg_actions, action % self.env.adg_actions
                )
            )
        if not self.env.update(action, self.reward, print_action):
            self.mapping_state = False
            self.sum_reward *= -10

        self.last_action = action
        # self.sum_reward += self.reward[action]  # add reward to sum_reward

    def take_route_action(self, action):
        self.mode = "route"
        if not self.env.route_update(action):
            self.mapping_state = False
            self.sum_reward *= -10
        else:
            self.mapping_state = True
            self.sum_reward -= 1

    def get_reward(self):
        return self.sum_reward

    def take_ramdom_action(self):
        actions = self.get_actions()
        action = random.choice(actions)
        self.take_action(action)

    def get_policy_action(self, policy):
        action = policy.argmax()
        return action

    def draw(self):
        """draw the visual state"""

        self.env.graph_visual.draw_state(self)


class GraphEmbedding(nn.Module):
    def __init__(self, dfg_shape, adg_shape, device):
        super().__init__()
        self.device = device
        self.fc96 = nn.Linear(64, 32)
        self.fc32 = nn.Linear(32, 32)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.dfg_gat = GAT(dfg_shape, 8, 32, 0.6, 0.2, 8)
        self.adg_gat = GAT(adg_shape, 8, 32, 0.6, 0.2, 8)
        self.meta = nn.Linear(7, 32)

    def forward(self, batch_env):
        # dfg embedding
        batch_dfg_features_vec = []
        batch_adg_features_vec = []
        batch_dfg_adj = []
        batch_adg_adj = []
        for e in batch_env:
            batch_dfg_features_vec.append(e.dfg_features_vec)
            batch_adg_features_vec.append(e.adg_features_vec)
            e.dfg_adj[e.dfg_adj == -1] = 0
            batch_dfg_adj.append(e.dfg_adj)
            e.adg_adj[e.adg_adj == -1] = 0
            batch_adg_adj.append(e.adg_adj)

        batch_dfg_features_vec = np.array(batch_dfg_features_vec)
        batch_adg_features_vec = np.array(batch_adg_features_vec)
        batch_dfg_adj = np.array(batch_dfg_adj)
        batch_adg_adj = np.array(batch_adg_adj)

        dfg_x = self.dfg_gat(
            torch.FloatTensor(batch_dfg_features_vec).to(self.device),
            torch.FloatTensor(batch_dfg_adj).to(self.device),
        )

        dfg_x = torch.mean(dfg_x, dim=1, keepdim=True)
        # dfg_x = self.dfg_reduce_fc(dfg_x)

        # adg embedding
        adg_x = self.adg_gat(
            torch.FloatTensor(batch_adg_features_vec).to(self.device),
            torch.FloatTensor(batch_adg_adj).to(self.device),
        )
        adg_x = torch.mean(adg_x, dim=1, keepdim=True)

        # current mapped node information
        # meta = self.fc(env.dfg_features[env.last_action])
        # adg_x = self.adg_reduce_fc(adg_x)
        x = torch.cat([dfg_x, adg_x], dim=2)

        # mlp fc 32-32
        x = F.relu(self.fc96(x))
        x = F.relu(self.fc32(x))

        # output
        x = self.leakyrelu(x)

        return x


"""set leraning rate for optimizer"""


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super().__init__()
        self.fc = nn.Linear(state_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, mask):
        x = F.relu(self.fc(x))
        x = self.leakyrelu(x)

        # x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        # x = F.normalize(x, p=1, dim=1)

        x = F.log_softmax(x, dim=2)

        # print("x shape {}".format(x[...,1:100]))
        x = x * mask
        # print(x[...,1:100])
        # x = x.reshape(-1)
        return x


class ValueNet(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.fc = nn.Linear(32, 1)
        self.mlp = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.mlp(x)
        x = self.leakyrelu(x)
        x = self.fc(x)
        return x


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        dfg_shape,
        adg_shape,
        state_size,
        action_size,
        route_action_size,
        hidden_size=32,
        device="cpu",
    ):
        super().__init__()
        self.embedding = GraphEmbedding(dfg_shape, adg_shape, device)
        self.policyNet = PolicyNet(state_size, action_size, hidden_size)
        self.routePolicyNet = PolicyNet(state_size, route_action_size, hidden_size)
        self.valueNet = ValueNet()

    def forward(self, state, mask, mode="placement"):
        x = self.embedding(state)
        if mode == "placement":
            log_act_probs = self.policyNet(x, mask)
        else:
            log_act_probs = self.routePolicyNet(x, mask)
        value = self.valueNet(x)
        return log_act_probs, value


class AgentNetwork:
    def __init__(
        self,
        dfg_shape,
        adg_shape,
        state_size,
        action_size,
        route_size,
        hidden_size=32,
        use_gpu=False,
        model_file=None,
    ):
        # self.graph_embedding = GraphEmbedding()
        super().__init__()
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu else "cpu")

        self.agentNet = PolicyValueNet(
            dfg_shape,
            adg_shape,
            state_size,
            action_size,
            route_size,
            hidden_size,
            device=self.device,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.agentNet.parameters(), lr=0.01)

        if model_file:
            self.agentNet = torch.load(
                model_file, map_location=self.device
            )  # load_state_dict(net_params)

    def policy_value(self, batch_state):
        # ! state is a  batch list of states
        # ! fix this for batch processing
        # x = self.embedding(state_embedding)
        batch_env = [state.env for state in batch_state]
        batch_mask = [state.env.mask.flatten() for state in batch_state]
        log_act_probs, value = self.agentNet(
            batch_env, torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device)
        )
        act_probs = np.exp(log_act_probs.cpu().data.numpy())

        value = value.cpu().data.numpy()

        return act_probs, value

    def route_policy_value(self, batch_state):
        batch_env = [state.env for state in batch_state]
        batch_mask = [state.env.route_mask for state in batch_state]
        log_act_probs, value = self.agentNet(
            batch_env, torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device)
        )
        act_probs = np.exp(log_act_probs.cpu().data.numpy())
        value = value.cpu().data.numpy()
        return act_probs, value

    def policy_value_pure(self, batch_state):
        batch_env = [state.env for state in batch_state]
        batch_mask = np.array([state.mask.flatten() for state in batch_state])
        log_act_probs, value = self.agentNet(
            batch_env, torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device)
        )
        return log_act_probs, value

    def route_policy_value_pure(self, batch_state):
        batch_env = [state.env for state in batch_state]
        batch_mask = np.array([state.env.route_mask for state in batch_state])
        log_act_probs, value = self.agentNet(
            batch_env, torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device)
        )
        return log_act_probs, value

    def policy_value_fn(self, state):
        legal_actions = state.get_actions()
        mask = state.mask * state.env.dfg_mask
        log_act_probs, value = self.agentNet(
            [state.env], torch.FloatTensor(mask.flatten()).to(self.device)
        )
        log_act_probs = log_act_probs.reshape(-1)
        act_probs = np.exp(log_act_probs.cpu().data.numpy())
        value = value.cpu().data.numpy()[0][0]
        act_probs = zip(legal_actions, act_probs[legal_actions])
        return act_probs, value

    def route_policy_value_fn(self, state):
        legal_actions = state.get_route_actions()
        mask = state.env.route_mask
        log_act_probs, value = self.agentNet(
            [state.env], torch.FloatTensor(mask.flatten()).to(self.device), mode="route"
        )
        log_act_probs = log_act_probs.reshape(-1)
        act_probs = np.exp(log_act_probs.cpu().data.numpy())
        value = value.cpu().data.numpy()[0][0]
        act_probs = zip(legal_actions, act_probs[legal_actions])
        return act_probs, value

    def route_policy_value(self, batch_state):
        batch_env = [state.env for state in batch_state]
        batch_mask = np.array([state.env.route_mask for state in batch_state])
        log_act_probs, value = self.agentNet(
            batch_env,
            torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device),
            mode="route",
        )
        return log_act_probs, value

    def train_step(self, state_batch, mcts_probs, sum_latency, lr, mode="placement"):
        """perform a training step"""
        if self.use_gpu:
            state_batch = state_batch
            mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
            sum_latency = torch.FloatTensor(sum_latency).to(self.device)
        else:
            state_batch = state_batch
            mcts_probs = mcts_probs
            sum_latency = sum_latency
        # reset gradients
        self.optimizer.zero_grad()

        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        if mode == "placement":
            act_probs, value = self.policy_value_pure(state_batch)
        else:
            act_probs, value = self.route_policy_value_pure(state_batch)
        # value = torch.FloatTensor(value).to(self.device)
        # act_probs = torch.FloatTensor(act_probs).to(self.device)

        value_loss = F.mse_loss(value.view(-1), sum_latency)

        policy_loss = -torch.mean(torch.sum(mcts_probs * act_probs, 2))

        # backward
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        # calc policy entropy, for monitoring only

        entropy = -torch.mean(torch.sum(act_probs * torch.exp(act_probs), 2))
        return loss.item(), entropy.item()


# class GATAutoEncoder
