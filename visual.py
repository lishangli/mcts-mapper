import networkx as nx
import matplotlib.pyplot as plt

# from adgParser import *
# from dfgParser import *
# from model import *
# from operations import Operations

class GraphVisual:
    def __init__(self, dfg, adg):
        self.dfg_graph = nx.Graph(name = dfg.name)
        self.dfg = dfg
        self.adg_graph = nx.Graph(name = "arch")
        self.adg = adg
        self.state_graph = nx.Graph(name = "mapping_state")
        
        print("initliazing nodes and edges...")
        self.initNodes()
        self.initEdges()

    def initNodes(self):
        for node in self.dfg.nodes:
            self.dfg_graph.add_node(node.id, name=node.name)
        for id, node in self.adg.getNodes().items():
            self.adg_graph.add_node(node.getId(), name=node.getName(), node_type = node.getType(), pos = (node.getX(), node.getY()))
        for id, node in self.adg.getNodes().items():
            self.state_graph.add_node(node.getId(), name=node.getName(), node_type = node.getType(), pos = (node.getX(), node.getY()))
        

    def initEdges(self):
        for edge in self.dfg.edges:
            self.dfg_graph.add_edge(edge.tail, edge.head)
        for id, edge in self.adg.getEdges().items():
            self.adg_graph.add_edge(edge.getSrcId(), edge.getDstId())
        for id, edges in self.adg.getEdges().items():
            self.state_graph.add_edge(edges.getSrcId(), edges.getDstId())

    def draw(self):
        # nx.draw(self.dfg_graph, with_labels=True, font_weight='bold')

        # collect GPE nodes
        gpe_nodes = [node for node,data in self.adg_graph.nodes(data=True) if data["node_type"] == "GPE"]
        # collect IOB nodes
        iob_nodes = [node for node,data in self.adg_graph.nodes(data=True) if data["node_type"] == "IOB"]
        #pos = nx.spring_layout(self.adg_graph, k=0.5)
        
        pos = nx.get_node_attributes(self.adg_graph, 'pos')
        # for node in self.adg_graph.nodes():
        #     pos[node] = (node, 0)
        
        nx.draw_networkx_nodes(self.adg_graph, pos=pos, nodelist=gpe_nodes,node_size = 100, node_color='r')
        nx.draw_networkx_nodes(self.adg_graph, pos=pos, nodelist=iob_nodes,node_size = 100, node_color='b')
        gib_nodes = [node for node,data in self.adg_graph.nodes(data=True) if data["node_type"] == "GIB"]
        nx.draw_networkx_nodes(self.adg_graph, pos=pos, nodelist=gib_nodes,node_size = 50, node_color='y')
        edges = self.adg_graph.edges()
        nx.draw_networkx_edges(self.adg_graph, pos=pos, edgelist=edges, edge_color='g')
        nx.draw_networkx_labels(self.adg_graph, pos=pos, font_size=4, font_family='sans-serif', labels = nx.get_node_attributes(self.adg_graph, 'name'))
        # nx.draw(self.adg_graph, with_labels=True, font_weight='bold')
        plt.show()

    def draw_state(self, state):
        """draw the state of the mapping"""
        for id, node in self.adg.getNodes().items():
            if state.env.adg_features_vec[id][4] > -1:
                print("node {} is mapped to dfg node {}".format(id, state.env.adg_features_vec[id][4]))
            self.state_graph.nodes[id]["mapped_dfg_id"] = state.env.adg_features_vec[id][4]
        # collect GPE nodes
        gpe_nodes = [node for node,data in self.state_graph.nodes(data=True) if data["node_type"] == "GPE" and data["mapped_dfg_id"] == -1]
        # collect IOB nodes
        iob_nodes = [node for node,data in self.state_graph.nodes(data=True) if data["node_type"] == "IOB" and data["mapped_dfg_id"] == -1]
        
        pos = nx.get_node_attributes(self.state_graph, 'pos')
        # for node in self.adg_graph.nodes():
        #     pos[node] = (node, 0)

        mapped_gpe_nodes = [node for node,data in self.state_graph.nodes(data=True) if data["node_type"] == "GPE" and data["mapped_dfg_id"] != -1]
        mapped_iob_nodes = [node for node,data in self.state_graph.nodes(data=True) if data["node_type"] == "IOB" and data["mapped_dfg_id"] != -1]
        # mapped_gib_nodes = [node for node,data in self.state_graph.nodes(data=True) if data["node_type"] == "GIB" and data["mapped_dfg_id"] != -1]
        nx.draw_networkx_nodes(self.state_graph, pos=pos, nodelist=gpe_nodes, node_size = 200, node_color='r')
        nx.draw_networkx_nodes(self.state_graph, pos=pos, nodelist=iob_nodes, node_size = 200, node_color='b')
        nx.draw_networkx_nodes(self.state_graph, pos=pos, nodelist=mapped_gpe_nodes, node_size = 200, node_color='0.5')
        nx.draw_networkx_nodes(self.state_graph, pos=pos, nodelist=mapped_iob_nodes, node_size = 200, node_color='0.5')
        gib_nodes = [node for node,data in self.state_graph.nodes(data=True) if data["node_type"] == "GIB"]
        nx.draw_networkx_nodes(self.state_graph, pos=pos, nodelist=gib_nodes, node_size = 100, node_color='y')
        edges = self.state_graph.edges()
        nx.draw_networkx_edges(self.state_graph, pos=pos, edgelist=edges, edge_color='g')
        nx.draw_networkx_labels(self.state_graph, pos=pos, font_size=10, font_family='sans-serif', labels = nx.get_node_attributes(self.state_graph, 'mapped_dfg_id'))
        # nx.draw(self.adg_graph, with_labels=True, font_weight='bold')
        plt.show()

# # 使用示例
# dfg_parser = DFGParser('dfg.json')
# adg_parser = ADGIR('cgra_adg.json')
# operations = Operations().OpParser("operations.json")

# dfg = dfg_parser.getDFG()
# adg = adg_parser.getADG()
# env = Environment(dfg, adg, operations)
# state = MappingState(env)
# state.draw()
# gv = GraphVisual(dfg, adg)
# gv.draw()

