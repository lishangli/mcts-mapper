import numpy as np


def getDfgAdj(dfg):
    nodes = dfg.getNodes()
    edges = dfg.getEdges()
    num_nodes = len(nodes)
    adj = np.zeros((num_nodes, num_nodes))
    for edge in edges:
        src = edge.getTail()
        dest = edge.getHead()
        adj[src, dest] = 1
    return adj


def getAdgAdj(adg):
    edges = adg.getEdges()
    # print(len(nodes))
    cnt = 0
    max_node_id = adg.getMaxNodeId()
    adj = np.zeros((max_node_id + 1, max_node_id + 1))
    for id, edge in edges.items():
        src = edge.getSrcId()
        srd_port = edge.getSrcPortIdx()
        dest = edge.getDstId()
        dest_port = edge.getDstPortIdx()
        adj[src, dest] = 1
        adj[dest, src] = 1
        cnt += 1
    return adj
