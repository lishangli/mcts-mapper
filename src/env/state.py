from parser import DFGFeatures, ADGFeatures
from utils import getDfgAdj, getAdgAdj

import numpy as np
import random
import copy
import numpy as np
from line_profiler import profile
import json
from collections import defaultdict


class Environment:
    def __init__(self, dfg, adg, operations, is_heuristic=True):

        self.mask = np.ones((adg.getNodeNums(), adg.getMaxNodeId() + 1))
        self.dfg_mask = np.zeros((adg.getNodeNums(), adg.getMaxNodeId() + 1))
        self.route_mask = np.zeros(len(adg.getEdges()))

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
        # self.graph_visual = GraphVisual(dfg, adg)
        self.heristic_reward = is_heuristic

        self.route_cur_node = []
        self.route_cur_edges = {}
        self.dfg_links = defaultdict(list)

        self._precompute_edge_mappings()

        self.place_cur_node = []
        self.init_mask()
        self.last_action = -1
        self.route_step = 0

        self._cur_adg_node_id = None
        self._cur_dfg_node_id = None

        self.latency = np.zeros(adg.getMaxNodeId() + 1)
        self.dfg_latency = np.zeros(adg.getMaxNodeId() + 1)

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

    def _precompute_edge_mappings(self):
        """预计算边缘映射关系，加速后续查询"""
        edges = self.adg.getEdges()
        self._src_to_edges = {}
        self._dst_to_src = {}
        for eid, edge in edges.items():
            src = edge.getSrcId()
            dst = edge.getDstId()
            if src not in self._src_to_edges:
                self._src_to_edges[src] = []
            self._src_to_edges[src].append((eid, dst))
            if dst not in self._dst_to_src:
                self._dst_to_src[dst] = []
            self._dst_to_src[dst].append((eid, src))

    def route_update(self, action):
        """优化后的路由更新函数"""
        # 1. 获取边的信息
        edges = self.adg.getEdges()
        if action not in edges:
            print("action is not in adg edges")
            return False

        edge = edges[action]
        inN = edge.getSrcId()
        outN = edge.getDstId()

        # 2. 验证路由条件
        if not self.route_cur_node:  # 空集检查更高效
            print("route cur node is empty")
            return True

        if inN not in self.route_cur_node:
            print("inN is not equal to route_cur_node")
            return False

        if self.route_mask[action] <= 0:
            print("action can not be mapped")
            return False

        # 3. 执行路由更新 - 通用部分
        self.route_cur_node.remove(inN)
        reward = self.get_delay(outN)

        # 设置标志以避免重复条件评估
        is_terminal = outN == self.last_action % self.adg_actions
        self.latency[outN] = max(self.latency[inN] + 1, self.latency[outN])
        self.route_cur_edges[outN] = self.route_cur_edges[inN]
        self.latency[inN] = 0

        # 4. 根据是否为终点节点执行不同的更新
        if not is_terminal:
            # 中间节点处理 - 使用预计算的映射
            self.route_cur_node.append(outN)

            # latency[outN] = max(laterncy[inN] + 1, latency[outN])
            # latency[inN] = 0

            # 获取与outN相关的出边
            if outN in self._src_to_edges:
                out_edges_data = self._src_to_edges[outN]
                valid_out_edges = [
                    eid for eid, _ in out_edges_data if self.route_mask[eid] >= 0
                ]

                # 向量化更新route_mask和adg_adj
                self.route_mask[valid_out_edges] = 1
                for eid, dst in out_edges_data:
                    if eid in valid_out_edges:
                        self.adg_adj[outN, dst] = 1

            # 获取与inN相关的出边
            if inN in self._src_to_edges:
                in_edges_data = self._src_to_edges[inN]
                valid_in_edges = [
                    eid for eid, _ in in_edges_data if self.route_mask[eid] >= 0
                ]

                # 向量化更新
                self.route_mask[valid_in_edges] = 0
                for eid, dst in in_edges_data:
                    if eid in valid_in_edges:
                        self.adg_adj[inN, dst] = 0

            # 获取指向inN的边
            if inN in self._dst_to_src:
                reversed_edges_data = self._dst_to_src[inN]
                valid_reversed_edges = [
                    eid for eid, _ in reversed_edges_data if self.route_mask[eid] >= 0
                ]

                # 向量化更新
                self.route_mask[valid_reversed_edges] = 0
                for eid, src in reversed_edges_data:
                    if eid in valid_reversed_edges:
                        self.adg_adj[src, inN] = 0

            # 验证节点类型 - 只在必要时调用
            if self.adg.getNode(outN).getType() != "GIB":
                print("{} not GIB\n".format(outN))
                return False

            # 最后更新当前路由状态
            self.route_step += 1
            self.adg_features_vec[outN][4] = outN
        else:
            # save the final latency for terminal dst node.
            self.dfg_latency[self.last_action // self.adg_actions] = self.latency[outN]

        # 共同的最终更新 - 不管是终点还是中间节点
        self.adg_adj[inN, outN] = 0
        self.dfg_links[self.route_cur_edges[outN]].append(action)
        self.route_mask[action] = -1

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
        self.place_cur_node.extend(zero_incoming)

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
            if node.getOpCode() in ("INPUT", "OUTPUT", "input", "output")
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
            if node.getOpCode() not in ("INPUT", "OUTPUT", "input", "output")
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

        # print("Mask initialization completed successfully")

    def init_route_mask(self):
        self.route_mask = np.zeros(len(self.adg.getEdges()))

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
            return op_code in ("INPUT", "OUTPUT", "input", "output")

        # GPE nodes cannot map to INPUT/OUTPUT operations
        if node_type == "GPE":
            return op_code not in ("INPUT", "OUTPUT", "input", "output")

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
        # if src_node.getType() != "GIB" and dst_node.getType() != "GIB":
        #     ct = 1
        return abs(src_x - dst_x) + abs(src_y - dst_y) - ct

    def get_delay(self, adg_act):
        mid_type = self.adg_features_vec[adg_act][1]
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
        self.last_action = action

        # Find predecessor adg nodes that need routing
        predecessor_nodes, predecessor_dfg_nodes = self.find_predecessor_nodes(
            dfg_node_id
        )

        # Find successor dfg nodes that need placing
        successor_nodes = self.find_successors_nodes(dfg_node_id)
        # Remove current node from current placement nodes
        self.place_cur_node.remove(dfg_node_id)

        # Update routing nodes if predecessors exist
        if predecessor_nodes:
            self.route_cur_node = predecessor_nodes
            for pre, pre_node in zip(predecessor_dfg_nodes, predecessor_nodes):
                eid = self.dfg.findEdge(pre, dfg_node_id)
                self.route_cur_edges[pre_node] = eid
            edges = self.adg.getEdges()
            src_ids, edge_ids = np.array(
                [(edge.getSrcId(), id) for id, edge in edges.items()]
            ).T

            pre_set = set(predecessor_nodes)
            mask = np.isin(src_ids, list(pre_set))
            self.route_mask[edge_ids[mask]] = 1

        # Update placing nodes if no predecessors
        if successor_nodes:
            self.place_cur_node.extend(successor_nodes)

    def find_predecessor_nodes(self, dfg_node_id):
        """
        Find all predecessor nodes in DFG that have edges to the current node.

        Args:/
            dfg_node_id (int): Current DFG node ID

        Returns:
            list[int]: List of ADG node IDs for predecessors that need routing
        """
        # Get indices of nodes that have edges to /current node
        predecessor_indices = np.where(abs(self.dfg_adj[:, dfg_node_id]) == 1)[0]

        # Map these DFG nodes to their corresponding ADG nodes using feature vector
        predecessor_adg_nodes = [
            int(self.dfg_features_vec[i, 6]) for i in predecessor_indices
        ]

        return predecessor_adg_nodes, predecessor_indices

    def find_successors_nodes(self, dfg_node_id):
        """
        Find all successor nodes in DFG that have edges to the current node.
        Args:
            dfg_node_id (int): Current DFG node ID
        Returns:
            list[int]: List of ADG node IDs for successors that need routing
        """
        # Get indices of nodes that have edges to current node
        indices = np.abs(self.dfg_adj[dfg_node_id, :]) == 1
        successor_indices = np.arange(self.dfg_adj.shape[1])[indices]

        # Filter successors based on column condition
        # A successor is valid if all values in its column are <= 0
        column_conditions = (self.dfg_adj[:, successor_indices] <= 0).all(axis=0) & (
            self.dfg_features_vec[successor_indices, 6] == -1
        )

        return successor_indices[column_conditions].tolist()

    def reset(self):
        self.mask = np.ones((self.adg.getNodeNums(), self.adg.getMaxNodeId() + 1))
        self.dfg_mask = np.zeros((self.adg.getNodeNums(), self.adg.getMaxNodeId() + 1))
        self.route_mask = np.zeros(len(self.adg.getEdges()))
        self.dfgDeatures = DFGFeatures(self.init_env[0])
        self.dfg_features = self.dfgDeatures.features

        self.dfg_features_vec = self.dfgDeatures.getFeaturesVector(self.init_env[2])
        self.adgDeatures = ADGFeatures(self.init_env[1])
        self.adg_features = self.adgDeatures.features
        self.adg_features_vec = self.adgDeatures.getFeaturesVector()
        self.dfg_adj = getDfgAdj(self.init_env[0])
        self.adg_adj = getAdgAdj(self.init_env[1])
        self.route_cur_node = []
        self.place_cur_node = []
        self.last_action = -1
        self.init_mask()
        self.init_route_mask()

    def draw(self):
        """draw the environment"""
        pass
        # self.graph_visual.draw()

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

    def get_key(self):
        """env key for mcts state cache"""
        return self.dfg_features_vec.tobytes()


class MappingState:
    def __init__(self, env, action=None):
        super().__init__()
        # fixme: wrong assginment for last action
        # self.last_action = env.place_cur_node[0]
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

        self.actions = self.get_actions()
        self.route_actions = self.get_route_actions()

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
            if node.getOpCode() == "INPUT"
            or node.getOpCode() == "OUTPUT"
            or node.getOpCode() == "input"
            or node.getOpCode() == "output"
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
            if node.getOpCode() != "INPUT"
            and node.getOpCode() != "OUTPUT"
            and node.getOpCode() != "input"
            and node.getOpCode() != "output"
        ]:
            for adg_id in [
                node.getId()
                for node in self.env.adg.getNodes()
                if node.getType() == "IOB"
            ]:
                self.mask[dfg_id, adg_id] = 0

    def is_terminal(self):
        if self.mode == "placement":
            # # print("placement terminal")
            # if np.all(self.mask <= 0) or (len(self.get_actions()) == 0):
            #     print("get actions {}".format(len(self.get_actions())))
            #     print("mapping state {}".format(self.mapping_state))
            return (
                np.all(self.mask <= 0)
                or (len(self.actions) == 0)
                or self.mapping_state == False
            )
        else:
            # print("route terminal")
            # if (
            #     (len(self.get_route_actions()) == 0)
            #     or self.mapping_state == False
            #     or np.all(self.env.route_mask <= 0)
            # ):
            #     print("get route actions {}".format(len(self.get_route_actions())))
            #     print("mapping state {}".format(self.mapping_state))
            #     print("route mask {}".format(len(self.env.route_cur_node)))
            return (
                (len(self.route_actions) == 0)
                or self.mapping_state == False
                or np.all(self.env.route_mask <= 0)
                or len(self.env.route_cur_node) == 0
            )

    def is_route_success(self):
        return len(self.env.route_cur_node) == 0 and self.is_terminal()

    def is_route_terminal(self):
        return np.all(self.env.route_mask == 0) or len(self.get_route_actions()) == 0

    def get_actions(self):
        actions = []

        for i in self.env.place_cur_node:
            if np.any(self.mask[i, :] == 1):
                for j in range(self.env.adg_actions):
                    if self.env.adg.getNode(j) != None and self.env.mask[i, j] == 1:
                        action = i * self.env.adg_actions + j
                        actions.append(action)
            else:
                print("No any mask >= 0!")
        return actions

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
        target_node_id = self.env.last_action % self.env.adg_actions

        # Iterate through all edges to find valid routing options
        for edge_id, edge in self.env.adg.getEdges().items():
            if self._is_valid_routing_edge(edge_id, edge, target_node_id):
                valid_actions.append(int(edge_id))

        return valid_actions

    def get_route_action(self, method="random"):
        actions = self.get_route_actions()
        # action_probs = np.zeros(len(self.env.adg.getEdges()))
        if len(actions) > 0:
            distances = [self.get_route_distance(action) for action in actions]
            weights = np.exp(-4 * np.array(distances))
            # weights = 1 / (np.array(distances)**4 + 1e-5)

            probs = weights / np.sum(weights)
            # action_probs[actions] = probs
            if method == "greedy":
                action = actions[np.argmax(probs)]
            else:
                action = np.random.choice(
                    actions,
                    p=1.0 * probs
                    + 0.0 * np.random.dirichlet(0.3 * np.ones(len(probs))),
                )
                return action
        else:
            if self.is_terminal():
                # print("WARNING: the state has no route legal actions")
                return -1
            # print("WARNING: the state don't need to route actions")
            return -2

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
        # src_cost = self.env.calculateCost(edge.getSrcId(), target_node_id)
        # dst_cost = self.env.calculateCost(target_node_id, edge.getDstId())
        # if src_cost + 4 <= dst_cost:
        #     return False

        # Check destination node type constraints
        dst_node = self.env.adg.getNode(edge.getDstId())
        dst_type = dst_node.getType()
        # Destination must be either:
        # 1. A GIB node, or
        # 2. A GPE/IOB node that matches the target node
        return dst_type == "GIB" or (
            (dst_type == "GPE" or dst_type == "IOB")
            and target_node_id == edge.getDstId()
        )

    def get_distance(self, cur_action):
        """
        Calculate distance between current action and precedecessor action nodes.

        Args:
            cur_action (int): Current action being considered

        Returns:
            int: Distance between nodes, returns 1 if no last action exists
        """

        # Extract ADG node IDs from actions
        current_node_id = self.env.get_adg_node_id(cur_action)
        current_dfg_id = self.env.get_dfg_node_id(cur_action)

        predecessors, preindices = self.env.find_predecessor_nodes(current_dfg_id)
        distance = 0

        if predecessors == []:
            # Get the target node ID from last action
            # print("last action is {}".format(self.env.last_action))
            if self.env.last_action == -1:
                return distance
            target_node_id = self.env.last_action % self.env.adg_actions

            distance += self.env.calculateCost(current_node_id, target_node_id)
            # print("no predecessors distance is {}".format(distance))
            return distance

        for pre in predecessors:
            distance += self.env.calculateCost(pre, current_node_id)

        return distance

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
        last_node_id = self.env.get_adg_node_id(self.env.last_action)

        # Calculate total distance (cost to destination + 1)
        return self.env.calculateCost(last_node_id, destination_node) + 1

    @profile
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
            self.sum_reward -= 10
        elif self.is_terminal() and (
            self.env.dfg_mask.any() or np.all(self.mask > 0)
        ):  # FIXME: here condition state is not ture for all case!
            self.mapping_state = False
            self.sum_reward -= 10
        else:
            self.last_reward = self.sum_reward
            self.sum_reward += 1

        self.env.last_action = action
        self.actions = self.get_actions()
        self.route_actions = self.get_route_actions()
        # self.sum_reward += self.reward[action]  # add reward to sum_reward

    @profile
    def take_route_action(self, action):
        self.mode = "route"
        if not self.env.route_update(action):
            self.mapping_state = False
            self.sum_reward -= 10
        elif self.is_terminal():
            if len(self.env.route_cur_node) > 0:
                self.sum_reward -= 10
                self.mapping_state = False
            else:
                self.mapping_state = True
                self.sum_reward -= 0.1
        else:
            self.mapping_state = True
            self.sum_reward -= 0.1
            self.route_actions = self.get_route_actions()

    def get_dfg_meta(self):
        dfg_meta = np.zeros_like(self.env.dfg_features_vec)
        if self.env.place_cur_node:
            dfg_meta[self.env.place_cur_node] = self.env.dfg_features_vec[
                self.env.place_cur_node
            ]
        return dfg_meta

    def get_adg_meta(self):
        adg_meta = np.zeros_like(self.env.adg_features_vec)
        if self.env.route_cur_node:
            adg_meta[self.env.route_cur_node] = self.env.adg_features_vec[
                self.env.route_cur_node
            ]
        return adg_meta

    def get_reward(self):
        return self.sum_reward
        # if self.mode == "placement":
        #     return self.sum_reward
        # else:
        #     return self.sum_reward - self.last_reward

    def take_ramdom_action(self):
        actions = self.get_actions()
        action = random.choice(actions)
        self.take_action(action)

    def get_policy_action(self, policy):
        action = policy.argmax()
        return action

    def draw(self, g_visual):
        """draw the visual state"""

        g_visual.draw_state(self)

    def start_anime(self, g_visual):
        g_visual.draw()
        g_visual.setup_plot()
        g_visual.start_anime(self)

    def end_anime(self, g_visual):
        g_visual.draw_state(self)

        g_visual.end_anime()

    def dump_config(self, config_path="config.json"):
        config = {}

        dfg_node_attrs = {}
        dfg_edge_attrs = {}
        adg_node_attrs = {}

        # get dfg node attr
        for node in self.env.dfg.getNodes():
            id = node.id
            dfg_node_attrs[id + 1] = {
                "minLat": self.env.dfg_latency[id],
                "maxLat": self.env.dfg_latency[id],
                "lat": self.env.dfg_latency[id],
                "adgNode": int(self.env.dfg_features_vec[id][6]),
            }

        for edge in self.env.dfg.getEdges():
            id = edge.id
            srcId, dstId = edge.tail, edge.head
            # adg_src_id = self.env.dfg_features.vec[srcId][6]
            # adg_dst_id = self.env.adg_features.vec[dstId][6]
            lat = self.env.dfg_latency[dstId] - self.env.dfg_latency[srcId]

            #
            edge_links = []
            edge_link_attrs = defaultdict(dict)
            for e in self.env.dfg_links[id]:
                adg_e = self.env.adg.getEdge(e)
                src_port = adg_e.getSrcPortIdx()
                dst_port = adg_e.getDstPortIdx()
                dst_id = adg_e.getDstId()
                src_id = adg_e.getSrcId()
                edge_link_attrs[src_id]["dstPort"] = src_port
                edge_link_attrs[dst_id]["srcPort"] = dst_port

            dfg_edge_attrs[id] = {
                "lat": 0,
                "delay": lat,
                "vio": 0,
                "edgeLinks": edge_link_attrs,
            }

        # ADG Node Attr
        for id, node in self.env.adg.getNodes().items():
            in_ports = {}
            out_ports = {}
            for eid, edge in self.env.adg.getEdges().items():
                if edge.getDstId() == id:
                    in_ports[edge.getDstPortIdx()] = True
                if edge.getSrcId() == id:
                    out_ports[edge.getSrcPortIdx()] = True

            adg_node_attrs[id] = {
                "dfgNode": int(self.env.adg_features_vec[id][4]) + 1,  # dfgNode
                "inPortUsed": in_ports,
                "outPortUesed": out_ports,
            }

        config = {
            "dfg_node_attrs": dfg_node_attrs,
            "dfg_edge_attrs": dfg_edge_attrs,
            "adg_node_attrs": adg_node_attrs,
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
