
from dfgParser import DFGParser

from operations import Operation, Operations   

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

    def dfs(self, node, topo_order):
        topo_order[node.id] = len(topo_order)
        for edge in self.dfg.edges:
            if edge.tail == node.id and edge.head not in topo_order:
                tailNode = self.findNodeById( edge.head)
                self.dfs( tailNode, topo_order)

    def getTopoOrder(self):
        topo_order = {}
        for node in self.dfg.nodes:
            if node.id not in topo_order and self.getInDegreeAndPredecessors(node)[0] == 0:
                self.dfs(node, topo_order)
        return topo_order



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
        features = []
        dfg_features = self.getDFGFeatures()
        for id, feature in dfg_features.items():
            op = operations.name2op[feature.opcode]
            features.append([feature.id, feature.topo_order, feature.sche_time, feature.in_degree, feature.out_degree, op, feature.mapped_pe_id])
        return features
    
    def __len__(self):
        return len(self.features)

# dfg_features = DFGFeatures(dfg)
# features = []
# for feature in dfg_features:
#     print(feature.id, feature.topo_order, feature.sche_time, feature.in_degree, feature.out_degree, name2op[feature.opcode], feature.mapped_pe_id)
#     features.append([feature.id, feature.topo_order, feature.sche_time, feature.in_degree, feature.out_degree, name2op[feature.opcode], feature.mapped_pe_id])

    