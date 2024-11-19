# import json 

# def load_data(json_file):
#     with open(json_file) as f:
#         data = json.load(f)
#     return data

# def save_data(data, json_file):
#     with open(json_file, 'w') as f:
#         json.dump(data, f)

# def load_dfg_data(json_file):
#     data = load_data(json_file)
#     dfg = DFG()
#     dfg.nodes = [Node(node['id'], node['op'], node['inputs'], node['outputs']) for node in data['nodes']]
#     for node in dfg.nodes:
#         node.edges = [Edge(edge[0], edge[1]) for edge in data['edges'] if edge[0] == node.id]
#     return dfg