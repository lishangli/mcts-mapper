import numpy as np


class DFGFeature:
    def __init__(self):
        self.id = None
        self.topo_order = None
        self.sche_time = None
        self.in_degree = None
        self.out_degree = None
        self.opcode = None
        self.mapped_pe_id = None


class DFGFeatures:
    def __init__(self, dfg):
        self.features = {}
        self.dfg = dfg
        self.getFeatures()

    def getFeatures(self):
        for node in self.dfg.nodes:
            feature = DFGFeature()
            feature.id = node.id
            feature.topo_order = 0
            feature.sche_time = 0
            feature.in_degree = 0
            feature.out_degree = 0
            feature.opcode = node.op
            feature.mapped_pe_id = -1
            self.features[feature.id] = feature

    def getInDegreeAndPredecessors(self, nodeId):
        in_degree = 0
        predecessors = []
        for edge in self.dfg.edges:
            if edge.head == nodeId:
                in_degree += 1
                predecessors.append(edge.tail)
        return (in_degree, predecessors)

    def getOutDegreeAndSuccessors(self, nodeId):
        out_degree = 0
        successors = []
        for edge in self.dfg.edges:
            if edge.tail == nodeId:
                out_degree += 1
                successors.append(edge.head)
        return (out_degree, successors)

    def findNodeById(self, id):
        for node in self.dfg.nodes:
            if node.id == id:
                return node
        return None

    # def dfs(self, node, topo_order, vis):
    #     vis[node.id] = 1
    #     for edge in self.dfg.edges:
    #         if edge.tail == node.id and edge.head not in vis:
    #             tailNode = self.findNodeById(edge.head)
    #             self.dfs(tailNode, topo_order, vis)
    #     topo_order[node.id] = len(topo_order)

    # def getTopoOrder(self, reverse=False):
    #     topo_order = {}
    #     vis = {}
    #     for node in self.dfg.nodes:
    #         if (
    #             node.id not in topo_order
    #             and self.getInDegreeAndPredecessors(node)[0] == 0
    #         ):
    #             self.dfs(node, topo_order, vis)
    #     for id, topo in topo_order.items():
    #         topo_order[id] = len(self.dfg.nodes) - topo - 1
    #     # if reverse:
    #     #     topo_order = {v: k for k, v in topo_order.items()}
    #     return topo_order
    def dfs(self, node, topo_order, vis):
        # Mark node as visited (using 1 for visiting, 2 for visited)
        vis[node.id] = 1

        # Visit all neighbors
        for edge in self.dfg.edges:
            if edge.tail == node.id:
                head_id = edge.head
                # Skip if already processed
                if head_id not in vis:
                    head_node = self.findNodeById(head_id)
                    self.dfs(head_node, topo_order, vis)
                # Check for cycle (if we're visiting a node currently in the stack)
                elif vis[head_id] == 1:
                    # You might want to handle cycles here
                    pass

        # Mark as completely visited and add to order
        vis[node.id] = 2
        # Insert at the beginning of topo_order (this gives reverse topological order directly)
        topo_order.insert(0, node.id)

    def getTopoOrder(self, reverse=False):
        topo_order = []  # Use a list for efficiency instead of dictionary
        vis = {}  # Track visited nodes

        # Find all nodes with in-degree 0 (precompute in-degrees)
        in_degrees = {}
        for node in self.dfg.nodes:
            in_degrees[node.id] = 0

        for edge in self.dfg.edges:
            if edge.head in in_degrees:
                in_degrees[edge.head] += 1

        # Start DFS from nodes with in-degree 0
        for node in self.dfg.nodes:
            if node.id not in vis and in_degrees[node.id] == 0:
                self.dfs(node, topo_order, vis)

        # Any unvisited nodes (could happen in disconnected graph)
        for node in self.dfg.nodes:
            if node.id not in vis:
                self.dfs(node, topo_order, vis)

        # Handle reverse parameter properly
        if reverse:
            topo_order.reverse()

        # Convert to dictionary with indices for compatibility
        topo_dict = {node_id: idx for idx, node_id in enumerate(topo_order)}
        return topo_dict

    def getDFGFeatures(self):
        dfg_features = {}
        topo_order = self.getTopoOrder()
        for node in self.dfg.nodes:
            feature = DFGFeature()
            feature.id = node.id
            feature.topo_order = topo_order[node.id]
            feature.sche_time = topo_order[node.id]
            feature.in_degree = self.getInDegreeAndPredecessors(node)[0]
            feature.out_degree = self.getOutDegreeAndSuccessors(node)[0]
            feature.opcode = node.op
            feature.mapped_pe_id = -1
            dfg_features[feature.id] = feature
        return dfg_features

    def getFeaturesVector(self, operations):
        features = np.zeros((len(self.dfg.nodes), 7))
        dfg_features = self.getDFGFeatures()
        for id, feature in dfg_features.items():
            op = operations.name2op[feature.opcode.lower()]
            features[feature.topo_order] = [
                feature.id,
                feature.topo_order,
                feature.sche_time,
                feature.in_degree,
                feature.out_degree,
                op,
                feature.mapped_pe_id,
            ]
        return features

    def __len__(self):
        return len(self.features)
