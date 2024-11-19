### Author: Shangli Li
### Date: 2024-09-13
### A parser for DFG
import json
import networkx as nx

class DFGParser (object):
    def __init__(self, jsonFile):
        self.dfg = DFG()
        self.jsonFile = jsonFile
        # self.nodes = []
        # self.edges = []
        self.parser()

    # def parse(self):
    #     for node in self.dfg.nodes:
    #         self.nodes.append(node.id)
    #         for edge in node.edges:
    #             self.edges.append(edge.id)
    
    def getDFG(self):
        return self.dfg
    
    def parser(self):
        with open(self.jsonFile) as f:
            data = json.load(f)
            self.dfg.setName(data["name"])
            for object in data["objects"]:
                info = {}
                if object["opcode"] in ["input", "output", "INPUT", "OUTPUT"]:
                    info["pattern"] = object["pattern"]
                    info["ref_name"] = object["ref_name"]
                    info["size"] = object["size"]
                    info["offset"] = object["offset"]
                elif object["opcode"] in ["CONST", "const"]:
                    info = object["value"]
                node = DFGNode(object["_gvid"], object["opcode"], object["name"],info)
                self.dfg.addNode(node)
            for edge in data["edges"]:
                edge = DFGEdge(edge["_gvid"], edge["tail"], edge["head"], edge["operand"])
                self.dfg.addEdge(edge)
    
class DFGNode:
    def __init__(self, id, opCode, opName, opInfo):
        self.id = id
        self.op = opCode
        self.name = opName
        self.info = opInfo

    def setOpInfo(self, infoName, infoValue):
        self.info[infoName] = infoValue 

    def getOpInfo(self, infoName):
        return self.info[infoName]
    
    def setOpName(self, name):
        self.name = name
    
    def getOpName(self):
        return self.name

    def setOpCode(self, opCode):
        self.op = opCode

    def getOpCode(self):
        return self.op
        
    
class DFGEdge:
    def __init__(self, id, tail, head, operand):
        self.id = id
        self.tail = tail
        self.head = head
        self.operand = operand

    def setTail(self, tail):
        self.tail = tail
    
    def getTail(self):
        return self.tail 
    
    def setHead(self, head):
        self.head = head

    def getHead(self):
        return self.head
    
    def setId(self, id):
        self.id = id

    def getId(self):
        return self.id
    
    def getOperand(self):
        return self.operand
    
    def setOperand(self, operand):
        self.operand = operand
    
class DFG:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.name = ""

    def setName(self, name):
        self.name = name

    def addNode(self, node):
        self.nodes.append(node)

    def addEdge(self, edge):
        self.edges.append(edge)

    def getNodes(self):
        return self.nodes

    def getEdges(self):
        return self.edges

    def getDFG(self):
        return self

    def setNodes(self, nodes):
        self.nodes = nodes

    def setEdges(self, edges):
        self.edges = edges

    def setDFG(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def __iter__(self):
        return iter(self.nodes + self.edges)
    
    

class Graph:
    def __init__(self, dfg):
        self.graph = nx.Graph(name = dfg.name)
        self.dfg = dfg
        self.initNodes()
        self.initEdges()

    def initNodes(self):
        for node in self.dfg.nodes:
            self.graph.add_node(node.id, name=node.name)

    def initEdges(self):
        for edge in self.dfg.edges:
            self.graph.add_edge(edge.tail, edge.head, operand=edge.operand)



# 使用示例
# parser = DFGParser('dfg.json')
# dfg = parser.getDFG()

# G = Graph(dfg)
# print(G.graph.graph)

# nx.draw(G.graph, with_labels=True, labels=nx.get_node_attributes(G.graph, 'name'), arrows=True, arrowstyle='-|>')



# # # 打印节点和边
# # print("Nodes:", dfg.nodes)
# # print("Edges:", dfg.edges)

# # # 绘制图
# import matplotlib.pyplot as plt
# # pos = nx.spring_layout(dfg)
# # nx.draw(dfg, pos, with_labels=True, labels=nx.get_node_attributes(dfg, 'name'))
# plt.show()