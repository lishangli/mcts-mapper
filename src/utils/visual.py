import networkx as nx
import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np



class GraphVisual:
    def __init__(self, dfg, adg):
        self.dfg_graph = nx.Graph(name=dfg.name)
        self.dfg = dfg
        self.adg_graph = nx.Graph(name="arch")
        self.adg = adg
        self.writer = None
        self.ani = None
        self.state_graph = nx.Graph(name="mapping_state")
        self.fig = None
        self.ax = None
        print("initliazing nodes and edges...")
        self.initNodes()
        self.initEdges()

    def initNodes(self):
        for node in self.dfg.nodes:
            self.dfg_graph.add_node(node.id, name=node.name)
        for id, node in self.adg.getNodes().items():
            self.adg_graph.add_node(
                node.getId(),
                name=node.getName(),
                node_type=node.getType(),
                pos=(node.getX(), node.getY()),
            )
        for id, node in self.adg.getNodes().items():
            self.state_graph.add_node(
                node.getId(),
                name=node.getName(),
                node_type=node.getType(),
                pos=(node.getX(), node.getY()),
            )

    def initEdges(self):
        for edge in self.dfg.edges:
            self.dfg_graph.add_edge(edge.tail, edge.head)
        for id, edge in self.adg.getEdges().items():
            self.adg_graph.add_edge(edge.getSrcId(), edge.getDstId())
        for id, edges in self.adg.getEdges().items():
            self.state_graph.add_edge(edges.getSrcId(), edges.getDstId(), mask=0)

    def setup_plot(self):
        # plt.figure()
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.writer = animation.FFMpegWriter(fps=1)
        self.writer.fig = self.fig
        # self.ani = animation.FuncAnimation(
        #     fig=self.fig, func=self.draw_state, interval=1000
        # )
        self.fig.canvas.draw()

    def draw(self):
        # nx.draw(self.dfg_graph, with_labels=True, font_weight='bold')
        if self.fig is None:
            self.setup_plot()

        plt.cla()
        # collect GPE nodes
        gpe_nodes = [
            node
            for node, data in self.adg_graph.nodes(data=True)
            if data["node_type"] == "GPE"
        ]
        # collect IOB nodes
        iob_nodes = [
            node
            for node, data in self.adg_graph.nodes(data=True)
            if data["node_type"] == "IOB"
        ]
        # pos = nx.spring_layout(self.adg_graph, k=0.5)

        pos = nx.get_node_attributes(self.adg_graph, "pos")
        # for node in self.adg_graph.nodes():
        #     pos[node] = (node, 0)

        nx.draw_networkx_nodes(
            self.adg_graph, pos=pos, nodelist=gpe_nodes, node_size=100, node_color="r"
        )
        nx.draw_networkx_nodes(
            self.adg_graph, pos=pos, nodelist=iob_nodes, node_size=100, node_color="b"
        )
        gib_nodes = [
            node
            for node, data in self.adg_graph.nodes(data=True)
            if data["node_type"] == "GIB"
        ]
        nx.draw_networkx_nodes(
            self.adg_graph, pos=pos, nodelist=gib_nodes, node_size=50, node_color="y"
        )
        edges = self.adg_graph.edges()
        nx.draw_networkx_edges(self.adg_graph, pos=pos, edgelist=edges, edge_color="g")
        nx.draw_networkx_labels(
            self.adg_graph,
            pos=pos,
            font_size=4,
            font_family="sans-serif",
            labels=nx.get_node_attributes(self.adg_graph, "name"),
        )
        # nx.draw(self.adg_graph, with_labels=True, font_weight='bold')
        plt.show()

    def draw_state(self, state):
        """draw the state of the mapping"""
        if self.fig is None:
            self.setup_plot()

        self.ax.clear()
        for id, node in self.adg.getNodes().items():
            if state.env.adg_features_vec[id][4] > -1:
                print(
                    "node {} is mapped to dfg node {}".format(
                        id, state.env.adg_features_vec[id][4]
                    )
                )
            self.state_graph.nodes[id]["mapped_dfg_id"] = state.env.adg_features_vec[
                id
            ][4]

        for id, edges in self.adg.getEdges().items():
            if state.env.route_mask[id] < 0:
                print(
                    "edge {} is routed from {} to {}".format(
                        id, edges.getSrcId(), edges.getDstId()
                    )
                )
                self.state_graph.edges[edges.getSrcId(), edges.getDstId()]["mask"] = (
                    state.env.route_mask[id]
                )

        print(
            "draw state route mask is {}".format(
                len(np.where(state.env.route_mask == 0))
            )
        )
        # collect GPE nodes
        gpe_nodes = [
            node
            for node, data in self.state_graph.nodes(data=True)
            if data["node_type"] == "GPE" and data["mapped_dfg_id"] == -1
        ]
        # collect IOB nodes
        iob_nodes = [
            node
            for node, data in self.state_graph.nodes(data=True)
            if data["node_type"] == "IOB" and data["mapped_dfg_id"] == -1
        ]
        # collect GIB nodes
        gib_nodes = [
            node
            for node, data in self.state_graph.nodes(data=True)
            if data["node_type"] == "GIB" and data["mapped_dfg_id"] == -1
        ]
        pos = nx.get_node_attributes(self.state_graph, "pos")
        edges = [
            (u, v)
            for u, v, data in self.state_graph.edges(data=True)
            if data["mask"] < 0
        ]

        routed_edges = [
            (u, v)
            for u, v, data in self.state_graph.edges(data=True)
            if data["mask"] < 0
        ]

        mapped_gpe_nodes = [
            node
            for node, data in self.state_graph.nodes(data=True)
            if data["node_type"] == "GPE" and data["mapped_dfg_id"] != -1
        ]
        mapped_iob_nodes = [
            node
            for node, data in self.state_graph.nodes(data=True)
            if data["node_type"] == "IOB" and data["mapped_dfg_id"] != -1
        ]
        mapped_gib_nodes = [
            node
            for node, data in self.state_graph.nodes(data=True)
            if data["node_type"] == "GIB" and data["mapped_dfg_id"] != -1
        ]

        nx.draw_networkx_nodes(
            self.state_graph,
            pos=pos,
            nodelist=gpe_nodes,
            node_size=200,
            node_color="r",
            ax=self.ax,
        )
        nx.draw_networkx_nodes(
            self.state_graph,
            pos=pos,
            nodelist=iob_nodes,
            node_size=200,
            node_color="b",
            ax=self.ax,
        )
        nx.draw_networkx_nodes(
            self.state_graph,
            pos=pos,
            nodelist=gib_nodes,
            node_size=100,
            node_color="y",
            ax=self.ax,
        )
        nx.draw_networkx_nodes(
            self.state_graph,
            pos=pos,
            nodelist=mapped_gpe_nodes,
            node_size=200,
            node_color="0.5",
            ax=self.ax,
        )
        nx.draw_networkx_nodes(
            self.state_graph,
            pos=pos,
            nodelist=mapped_iob_nodes,
            node_size=200,
            node_color="0.5",
            ax=self.ax,
        )
        nx.draw_networkx_nodes(
            self.state_graph,
            pos=pos,
            nodelist=mapped_gib_nodes,
            node_size=100,
            node_color="0.5",
            ax=self.ax,
        )
        nx.draw_networkx_edges(
            self.state_graph, pos=pos, edgelist=edges, edge_color="g", ax=self.ax
        )

        nx.draw_networkx_labels(
            self.state_graph,
            pos=pos,
            font_size=10,
            font_family="sans-serif",
            labels=nx.get_node_attributes(self.state_graph, "mapped_dfg_id"),
            ax=self.ax,
        )
        self.fig.canvas.draw_idle()
        plt.pause(1)  # 控制动画速度
        plt.savefig("figures/mapping_state.png")
        self.writer.grab_frame()
        print("draw state finish")

    def start_anime(self, state):
        # self.writer.setup(self.fig, "animation.mp4", dpi=100)
        self.writer_context = self.writer.saving(self.fig, "videos/ani1.avi", 100)
        self.writer_context.__enter__()
        self.draw_state(state)

    def end_anime(self):
        # plt.pause(1)
        # plt.draw()
        plt.show()
        self.writer_context.__exit__(None, None, None)
        self.writer.finish()
