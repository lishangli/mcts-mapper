# from dfg import DFGFeatures
# from adg import ADGFeatures
# from loadData import getDfgAdj, getAdgAdj
# from visual import GraphVisual
# import copy
# import numpy as np


# class Environment:
#     def __init__(self, dfg, adg, operations, is_heuristic=True):

#         self.mask = np.ones((adg.getNodeNums(), adg.getMaxNodeId() + 1))
#         self.dfg_mask = np.zeros((adg.getNodeNums(), adg.getMaxNodeId() + 1))
#         self.route_mask = np.ones(len(adg.getEdges()))

#         self.dfg = dfg
#         self.adg = adg
#         self.ops = operations
#         self.dfgFeatures = DFGFeatures(dfg)
#         self.dfg_features = self.dfgFeatures.features

#         self.dfg_features_vec = self.dfgFeatures.getFeaturesVector(operations)
#         self.adgFeatures = ADGFeatures(adg)
#         self.adg_features = self.adgFeatures.features
#         self.adg_features_vec = self.adgFeatures.getFeaturesVector()
#         self.dfg_adj = getDfgAdj(dfg)
#         self.adg_adj = getAdgAdj(adg)
#         self.init_env = copy.deepcopy((dfg, adg, operations))
#         self.reward = {}
#         self.adg_actions = self.adg.getMaxNodeId() + 1
#         self.graph_visual = GraphVisual(dfg, adg)
#         self.heristic_reward = is_heuristic
#         self.init_mask()

#         self.route_cur_node = []
#         self.place_cur_node = [-1]
#         self.route_step = 0

#         self._cur_adg_node_id = None
#         self._cur_dfg_node_id = None

#     def update(self, action, reward, print_action=False):
#         min_path = []
#         self._set_node_id(action)
#         if self.canMapping(action, reward):
#             self.updateFeatures()
#             self.updateAdj()
#             if print_action:
#                 print(
#                     "update current action is {}, {}".format(
#                         self._cur_dfg_node_id, self._cur_adg_node_id
#                     )
#                 )

#             self.updateMeta(action)
#             return True
#         else:
#             reward[action] = -4000
#             return False

#     def route_update(self, action):
#         inN = None
#         outN = None
#         if action in self.adg.getEdges():
#             inN = self.adg.getEdges()[action].getSrcId()
#             outN = self.adg.getEdges()[action].getDstId()
#         else:
#             print("action is not in adg edges")
#             return False

#         if inN not in self.route_cur_node:
#             if self.route_cur_node.size == 0:
#                 print("route cur node is empty")
#                 return True
#             print("inN is not equal to route_cur_node")
#             return False
#         elif self.route_mask[action] == 0:
#             print("action is already mapped")
#             return False
#         else:
#             self.route_cur_node.remove(inN)
#             reward = self.get_delay(outN)
#             if outN != self.place_cur_node[0] % self.adg_actions:
#                 self.route_cur_node.append(outN)
#             else:
#                 print(
#                     "adg success {}".format(self.place_cur_node[0] % self.adg_actions)
#                 )
#                 self.adg_adj[inN, outN] = 0
#                 self.route_mask[action] = 0
#                 return True
#             if self.adg.getNode(outN).getType() != "GIB":
#                 print("{} not GIB\n".format(outN))
#             self.route_step += 1
#             self.adg_features_vec[outN][4] = outN
#             print("Info# {}".format(outN))
#             self.adg_adj[inN, outN] = 0
#             self.route_mask[action] = 0
#             # print("mask {}".format(action))
#             # print("all mask {}".format(np.where(self.route_mask > 0)))

#         return True

#     def init_mask(self):
#         """
#         Initialize masks for DFG (Data Flow Graph) and ADG (Architecture Description Graph) mapping.
#         The function sets up mapping constraints between DFG nodes and ADG nodes based on their types.

#         The mask is a binary matrix where:
#         - 0 indicates mapping is not allowed
#         - 1 indicates mapping is allowed
#         """
#         # Initialize DFG mask with zeros
#         self.dfg_mask.fill(0)

#         # Set mask=1 for nodes with no incoming edges and specific feature
#         zero_incoming = np.where(
#             (np.all(self.dfg_adj == 0, axis=0)) & (self.dfg_features_vec[:, 6] == -1)
#         )[0]
#         self.dfg_mask[zero_incoming, :] = 1

#         # Get all GIB nodes from ADG and disable their mappings
#         gib_nodes = [
#             node_id
#             for node_id, node in self.adg.getNodes().items()
#             if node.getType() == "GIB"
#         ]
#         self.mask[:, gib_nodes] = 0

#         # Get INPUT/OUTPUT nodes from DFG
#         io_nodes = [
#             node.id
#             for node in self.dfg.getNodes()
#             if node.getOpCode() in ("INPUT", "OUTPUT")
#         ]

#         # Get GPE nodes from ADG
#         gpe_nodes = [
#             node.getId()
#             for node in self.adg.getNodes().values()
#             if node.getType() == "GPE"
#         ]

#         # Disable mapping between INPUT/OUTPUT nodes and GPE nodes
#         for dfg_id in io_nodes:
#             self.mask[dfg_id, gpe_nodes] = 0

#         # Get computation nodes (non-INPUT/OUTPUT) from DFG
#         comp_nodes = [
#             node.id
#             for node in self.dfg.getNodes()
#             if node.getOpCode() not in ("INPUT", "OUTPUT")
#         ]

#         # Get IOB nodes from ADG
#         iob_nodes = [
#             node.getId()
#             for node in self.adg.getNodes().values()
#             if node.getType() == "IOB"
#         ]

#         # Disable mapping between computation nodes and IOB nodes
#         for dfg_id in comp_nodes:
#             self.mask[dfg_id, iob_nodes] = 0

#         print("Mask initialization completed successfully")

#     def init_route_mask(self):
#         self.route_mask = np.ones(len(self.adg.getEdges()))

#     def getAct(self, action):
#         return action // self.adg_actions, action % self.adg_actions

#     # def updateFeatures(self, action):
#     #     dfg_act = self._get_dfg_node_id(action)
#     #     adg_act = self._get_adg_node_id(action)
#     #     self.mask[dfg_act, :] = 0
#     #     self.mask[:, adg_act] = 0
#     #     self.dfg_mask[dfg_act, :] = 0
#     #     # print("update features dfg {} adg {}".format(dfg_act, adg_act))

#     #     if self.adg.getNodes()[adg_act].getType() != "GIB":
#     #         self.adg_features_vec[adg_act][4] = dfg_act
#     #     else:
#     #         print("gib error!")
#     #     self.dfg_features_vec[dfg_act][6] = adg_act

#     #     self.dfg_features_vec[dfg_act][3] = len(self.dfg_adj[:, dfg_act])
#     #     self.adg_features_vec[adg_act][2] = self.adg.getNodes()[adg_act].getNumInputs()
#     #     self.adg_features_vec[adg_act][3] = self.adg.getNodes()[adg_act].getNumOutputs()
#     def updateFeatures(self):
#         """
#         Update features for both DFG and ADG nodes after an action is taken.

#         Args:
#             action (int): Combined action value encoding both DFG and ADG nodes

#         Updates:
#         - Node mappings in feature vectors
#         - Input/output counts
#         - Masks for both DFG and ADG

#         Raises:
#             ValueError: If attempting to map to a GIB node
#         """
#         # Extract node IDs
#         dfg_node_id = self._cur_dfg_node_id
#         adg_node_id = self._cur_adg_node_id

#         # Update mapping masks
#         self._update_masks(dfg_node_id, adg_node_id)

#         # Update node mappings
#         self._update_node_mappings(dfg_node_id, adg_node_id)

#         # Update connectivity features
#         self._update_connectivity_features(dfg_node_id, adg_node_id)

#     def _update_masks(self, dfg_node_id, adg_node_id):
#         """
#         Update masks to prevent further mapping of used nodes.
#         """
#         # Disable all mappings for these nodes
#         self.mask[dfg_node_id, :] = 0  # Disable row
#         self.mask[:, adg_node_id] = 0  # Disable column
#         self.dfg_mask[dfg_node_id, :] = 0  # Disable DFG mask row

#     def _update_node_mappings(self, dfg_node_id, adg_node_id):
#         """
#         Update the node mapping information in feature vectors.

#         Raises:
#             ValueError: If attempting to map to a GIB node
#         """
#         adg_node = self.adg.getNodes()[adg_node_id]

#         # Verify node type and update mappings
#         if adg_node.getType() == "GIB":
#             raise ValueError("Cannot map to GIB node!")

#         # Update bidirectional mapping between DFG and ADG nodes
#         self.adg_features_vec[adg_node_id][4] = dfg_node_id  # ADG -> DFG mapping
#         self.dfg_features_vec[dfg_node_id][6] = adg_node_id  # DFG -> ADG mapping

#     def _update_connectivity_features(self, dfg_node_id, adg_node_id):
#         """
#         Update input/output connectivity features for both DFG and ADG nodes.
#         """
#         adg_node = self.adg.getNodes()[adg_node_id]

#         # Update DFG node connectivity
#         self.dfg_features_vec[dfg_node_id][3] = len(self.dfg_adj[:, dfg_node_id])

#         # Update ADG node connectivity
#         self.adg_features_vec[adg_node_id][2] = adg_node.getNumInputs()
#         self.adg_features_vec[adg_node_id][3] = adg_node.getNumOutputs()

#     def updateAdj(self):
#         """
#         Process an action and update the adjacency matrices and masks accordingly.

#         Args:
#             action (int): Combined action value encoding both DFG and ADG actions

#         The function:
#         1. Splits the combined action into DFG and ADG components
#         2. Updates adjacency matrices
#         3. Updates masks based on new graph state
#         """
#         # Split combined action into DFG and ADG components
#         dfg_action = self._cur_dfg_node_id
#         adg_action = self._cur_adg_node_id

#         # Update DFG adjacency matrix
#         # Find nodes connected to dfg_action node and mark them as processed (-1)
#         dfg_connected_nodes = self.dfg_adj[dfg_action] == 1
#         self.dfg_adj[dfg_action, dfg_connected_nodes] = -1

#         # Update ADG adjacency matrix similarly
#         adg_connected_nodes = self.adg_adj[adg_action] == 1
#         self.adg_adj[adg_action, adg_connected_nodes] = -1

#         # Update masks for nodes with all incoming edges processed
#         # Find nodes where all incoming edges are <= 0 and have feature[6] == -1
#         ready_nodes = np.where(
#             (np.all(self.dfg_adj <= 0, axis=0)) & (self.dfg_features_vec[:, 6] == -1)
#         )[0]

#         # Enable these nodes in the mask
#         self.dfg_mask[ready_nodes, :] = 1

#         return self.dfg_mask.copy()

#     def isLegalAction(self, action):
#         """
#         Check if a given action is legal based on DFG and ADG constraints.

#         Args:
#             action (int): Combined action value encoding both DFG and ADG actions

#         Returns:
#             bool: True if the action is legal, False otherwise

#         Validates:
#         1. Mask compatibility
#         2. Node type restrictions
#         3. Operation code compatibility
#         """
#         # Decompose action into DFG and ADG components
#         dfg_action = action // self.adg_actions
#         adg_action = action % self.adg_actions

#         # Get node information
#         adg_node = self.adg.getNode(adg_action)
#         dfg_node = self.dfg.getNodes()[dfg_action]

#         # Check mask constraint
#         if self.mask[dfg_action, adg_action] == 0:
#             return False

#         # Check node type constraints
#         node_type = adg_node.getType()
#         op_code = dfg_node.getOpCode()

#         # GIB nodes are not valid targets
#         if node_type == "GIB":
#             return False

#         # IOB nodes can only map to INPUT/OUTPUT operations
#         if node_type == "IOB":
#             return op_code in ("INPUT", "OUTPUT")

#         # GPE nodes cannot map to INPUT/OUTPUT operations
#         if node_type == "GPE":
#             return op_code not in ("INPUT", "OUTPUT")

#         return True

#     def canMapping(self, action, reward):
#         dfg_act = action // self.adg_actions
#         adg_act = action % self.adg_actions
#         if self.mask[dfg_act, adg_act] != 1 or self.dfg_mask[dfg_act, adg_act] != 1:
#             return False

#         # print("test action is {}, {}".format(dfg_act, adg_act))

#         if self.mask[dfg_act, adg_act] == 1 and self.dfg_mask[dfg_act, adg_act] == 1:

#             # sum_cost = self.adg.getNode(adg_act).getMaxDelay()
#             sum_cost = self.get_delay(adg_act)

#             for src in range(self.dfg_adj.shape[0]):
#                 if self.dfg_adj[src, dfg_act] == 0:
#                     continue
#                 src_mapped_pe = self.dfg_features_vec[src][6]
#                 if src_mapped_pe == -1:
#                     print("src {} is not mapped".format(src))
#                     return False
#                 dst_mapped_pe = action % self.adg_actions
#                 if not self.heristic_reward:
#                     # print("src {}, dst {}".format(src_mapped_pe, dst_mapped_pe))
#                     cur_path = []
#                     cost = 0
#                     # path = []
#                     min_cost = 100000000
#                     path_cost = self.calculateCost(
#                         src_mapped_pe, dst_mapped_pe
#                     )  # heruistic reward

#                     # path_cost = self.findPath(
#                     #     src_mapped_pe,
#                     #     dst_mapped_pe,
#                     #     cost,
#                     #     min_cost,
#                     #     max_cost * 2,
#                     #     cur_path,
#                     #     path,
#                     #     0,
#                     # )
#                 else:
#                     # print("a star reward")
#                     path_cost = self.calculateCost(src_mapped_pe, dst_mapped_pe)
#                 # print("path cost is {}".format(path_cost))
#                 if path_cost < 0:
#                     return False
#                 sum_cost += path_cost
#             # update dfg environment
#             # for src in self.dfg_adj[:, action // self.adg_actions]:
#             #     # self.dfg_adj[src, action // self.adg_actions] = 0
#             #     self.dfg_features[src].out_degree -= 1
#             #     self.dfg_features[src].mapped_pe_id = action % self.adg_actions
#             # print("sum cost is {}".format(sum_cost))
#             reward[action] = -sum_cost
#             return True
#         return False

#     def findPath(self, src, dst, cost, min_cost, max_cost, path, min_path, d):
#         # if self.adg_adj[src,dst] == 0:
#         #     return -1
#         # print("{}: src {} dst {}".format(d, src,dst))
#         path_cost = 100000000
#         min_src = 0
#         min_dst = 0
#         if cost > max_cost:
#             return -1
#         path.append(src)

#         if src == dst:
#             if cost < min_cost:
#                 # print("update cost {}".format(cost))
#                 min_cost = cost
#                 min_path = path
#         else:
#             for mid_dst in range(len(self.adg_adj[src, :])):
#                 if self.adg.getNode(mid_dst) == None:
#                     continue

#                 # print(mid_dst)
#                 if self.adg.getNode(mid_dst).getType() != "GIB" and mid_dst != dst:
#                     continue
#                 # if mid_dst == dst:
#                 #     print("{} cur type is {}".format(mid_dst,self.adg.getNode(mid_dst).getType()))
#                 if self.adg_adj[src, mid_dst] != 0:
#                     # temp = port
#                     # print("cur type is {}".format(self.adg.getNode(mid_dst).getType()))
#                     self.adg_adj[src, mid_dst] = 0
#                     self.adg_adj[mid_dst, src] = 0
#                     # self.adg_adj[mid_dst, src] = 0

#                     mid_delay = self.get_delay(mid_dst)

#                     if mid_delay == None:
#                         mid_delay = 1
#                     path_cost = self.findPath(
#                         mid_dst,
#                         dst,
#                         cost + mid_delay,
#                         min_cost,
#                         max_cost,
#                         path,
#                         min_path,
#                         d + 1,
#                     )

#                     # self.adg_adj[src,mid_dst] = 1
#                     # self.adg_adj[mid_dst, src] = 1
#         path.pop()
#         if path_cost == 100000000:
#             return -1
#         return min_cost

#     def calculateCost(self, src, dst):
#         src_node = self.adg.getNodes()[src]
#         dst_node = self.adg.getNodes()[dst]
#         # if src_node.getType() == "GIB" or dst_node.getType() == "GIB":
#         #     return 0
#         src_x = src_node.getX()
#         src_y = src_node.getY()
#         dst_x = dst_node.getX()
#         dst_y = dst_node.getY()
#         ct = 0
#         if src_node.getType() != "GIB" and dst_node.getType() != "GIB":
#             ct = 1
#         return abs(src_x - dst_x) + abs(src_y - dst_y) + ct

#     def get_delay(self, adg_act):
#         mid_type = self.adg_features[adg_act].type
#         if mid_type == 2:
#             return 1
#         elif mid_type == 0:
#             return 2
#         else:
#             return 3

#     def updateMeta(self, action):
#         """
#         Update metadata after a node mapping action, including:
#         - Current placement node tracking
#         - Route node predecessors identification

#         Args:
#             action (int): The current action being processed

#         Updates:
#         - place_cur_node: Tracks current placement node
#         - route_cur_node: List of predecessor nodes that need routing
#         """
#         # Get current DFG and ADG node IDs
#         dfg_node_id = self._cur_dfg_node_id
#         adg_node_id = self._cur_adg_node_id

#         # Update current placement node
#         self.place_cur_node[0] = action

#         # Find predecessor nodes that need routing
#         predecessor_nodes = self._find_predecessor_nodes(dfg_node_id)

#         # Update routing nodes if predecessors exist
#         if predecessor_nodes:
#             self.route_cur_node = predecessor_nodes

#     def _find_predecessor_nodes(self, dfg_node_id):
#         """
#         Find all predecessor nodes in DFG that have edges to the current node.

#         Args:
#             dfg_node_id (int): Current DFG node ID

#         Returns:
#             list[int]: List of ADG node IDs for predecessors that need routing
#         """
#         # Get indices of nodes that have edges to current node
#         predecessor_indices = np.where(self.dfg_adj[:, dfg_node_id] != 0)[0]

#         # Map these DFG nodes to their corresponding ADG nodes using feature vector
#         predecessor_adg_nodes = [
#             self.dfg_features_vec[i, 6] for i in predecessor_indices
#         ]

#         return predecessor_adg_nodes

#     def reset(self):
#         self.mask = np.ones((self.adg.getNodeNums(), self.adg.getMaxNodeId() + 1))
#         self.dfg_mask = np.zeros((self.adg.getNodeNums(), self.adg.getMaxNodeId() + 1))
#         self.route_mask = np.ones(len(self.adg.getEdges()))
#         self.dfgDeatures = DFGFeatures(self.init_env[0])
#         self.dfg_features = self.dfgDeatures.features

#         self.dfg_features_vec = self.dfgDeatures.getFeaturesVector(self.init_env[2])
#         self.adgDeatures = ADGFeatures(self.init_env[1])
#         self.adg_features = self.adgDeatures.features
#         self.adg_features_vec = self.adgDeatures.getFeaturesVector()
#         self.dfg_adj = getDfgAdj(self.init_env[0])
#         self.adg_adj = getAdgAdj(self.init_env[1])
#         self.route_cur_node = []
#         self.place_cur_node = [-1]
#         self.init_mask()
#         self.init_route_mask()

#     def draw(self):
#         """draw the environment"""
#         self.graph_visual.draw()

#     def _set_node_id(self, action):
#         """
#         Extract ADG/DFG node ID from a combined action value.

#         Args:
#             action (int): Combined action value
#         """
#         self._cur_adg_node_id = action % self.adg_actions
#         self._cur_dfg_node_id = action // self.adg_actions

#     def get_cur_adg_node_id(self):
#         return self._cur_adg_node_id

#     def get_cur_dfg_node_id(self):
#         return self._cur_dfg_node_id

#     def get_adg_node_id(self, action):
#         """
#         Extract ADG node ID from a combined action value.

#         Args:
#             action (int): Combined action value

#         Returns:
#             int: ADG node ID
#         """
#         return action % self.adg_actions

#     def get_dfg_node_id(self, action):
#         """
#         Extract DFG node ID from a combined action value.
#         Args:
#             action (int): Combined action value
#         Returns:
#             int: DFG node ID
#         """
#         return action // self.adg_actions
