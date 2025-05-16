import re
import argparse 
import os

def parse_dot_graph(dot_string):
    """
    解析 DOT 字符串，提取节点和边的信息。

    Args:
        dot_string: DOT 格式的图字符串。

    Returns:
        一个字典，包含 'nodes' 和 'edges' 键，
        'nodes' 是一个字典，键为节点 ID，值为节点属性字典。
        'edges' 是一个列表，包含边的属性字典。
    """
    nodes = {}
    edges = []

    # 解析节点
    node_pattern = re.compile(r'\s*(\w+)\s*\[(.*?)\];')
    for line in dot_string.strip().split('\n'):
        node_match = node_pattern.match(line.strip())
        print(node_match)
        print(line.strip())
        if node_match:
            print(1)
            node_id = node_match.group(1).strip()
            attrs_str = node_match.group(2).strip()
            attrs = {}
            for attr_pair in attrs_str.split(','):
                if '=' in attr_pair:
                    key, value = attr_pair.split('=')
                    attrs[key.strip()] = value.strip()
            nodes[node_id] = attrs

    # 解析边
    edge_pattern = re.compile(r'\s*(\w+)\s*->\s*(\w+)\s*\[(.*?)\];')
    for line in dot_string.strip().split('\n'):
        edge_match = edge_pattern.match(line.strip())
        if edge_match:
            tail_id = edge_match.group(1).strip()
            head_id = edge_match.group(2).strip()
            attrs_str = edge_match.group(3).strip()
            attrs = {}
            for attr_pair in attrs_str.split(','):
                if '=' in attr_pair:
                    key, value = attr_pair.split('=')
                    attrs[key.strip()] = value.strip()
            edges.append({'tail': tail_id, 'head': head_id, **attrs}) # 将已解析的属性合并到边的字典中

    return {'nodes': nodes, 'edges': edges}

def duplicate_graph(graph_data, suffix1="_1", suffix2="_2"):
    """
    复制数据流图为两个新的图。

    Args:
        graph_data: parse_dot_graph 函数解析后的图数据。
        suffix1: 第一个复制图的节点ID后缀。
        suffix2: 第二个复制图的节点ID后缀。

    Returns:
        包含两个复制图数据的字典，键为 'graph1' 和 'graph2'。
    """
    graph_data_1 = {'nodes': {}, 'edges': []}
    graph_data_2 = {'nodes': {}, 'edges': []}

    # 复制节点并添加后缀
    for node_id, attrs in graph_data['nodes'].items():
        graph_data_1['nodes'][node_id + suffix1] = attrs.copy()
        graph_data_2['nodes'][node_id + suffix2] = attrs.copy()

    # 复制边并更新节点ID
    for edge_attr in graph_data['edges']:
        edge_attr_1 = edge_attr.copy()
        edge_attr_2 = edge_attr.copy()

        edge_attr_1['tail'] = edge_attr['tail'] + suffix1
        edge_attr_1['head'] = edge_attr['head'] + suffix1
        graph_data_1['edges'].append(edge_attr_1)

        edge_attr_2['tail'] = edge_attr['tail'] + suffix2
        edge_attr_2['head'] = edge_attr['head'] + suffix2
        graph_data_2['edges'].append(edge_attr_2)

    return {'graph1': graph_data_1, 'graph2': graph_data_2}

def to_dot_string_combined(graph_dict, graph_name="combined_duplicated_graph"):
    """
    将多个图数据合并到一个 DOT 字符串中。

    Args:
        graph_dict:  一个字典，键为图的名称，值为图数据字典（parse_dot_graph 返回的格式）。
        graph_name:  总的 DOT 图的名称。

    Returns:
        一个包含所有图的 DOT 格式字符串。
    """
    dot_string = f"digraph {graph_name} {{\n"

    for sub_graph_name, graph_data in graph_dict.items():
        dot_string += f"  subgraph cluster_{sub_graph_name} {{\n" # 使用 subgraph 创建子图，并添加 cluster_ 前缀使其在 Graphviz 中视觉上分组
        dot_string += f"    label = \"{sub_graph_name}\";\n" # 子图标签

        # 输出节点
        for node_id, attrs in graph_data['nodes'].items():
            attr_list = ', '.join([f'{key}={value}' for key, value in attrs.items()])
            dot_string += f"    {node_id} [{attr_list}];\n"

        # 输出边
        for edge_attr in graph_data['edges']:
            attr_list = ', '.join([f'{key}={value}' for key, value in edge_attr.items() if key not in ['tail', 'head']])
            edge_str = f"    {edge_attr['tail']} -> {edge_attr['head']}" # 子图内的边需要缩进
            if attr_list:
                edge_str += f" [{attr_list}]"
            dot_string += edge_str + ";\n"

        dot_string += "  }\n" # subgraph 结束

    dot_string += "}"
    return dot_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复制 DOT 文件中的图并合并到一个文件中。")

    parser.add_argument(
        "--input_dot_file", "-i",  # --input_dot_file 是长参数名，-i 是短参数名 (可选)
        required=True,         # 必须提供的参数
        help="输入 DOT 文件的路径" # 参数帮助信息
    )

    parser.add_argument(
        "--output_dir", "-o",
        default=".",  # 默认输出到当前目录
        help="输出 DOT 文件保存的目录 (默认为当前目录)"
    )

    args = parser.parse_args()
    input_dot_file_path = args.input_dot_file # 从解析结果中获取文件路径
    output_dir = args.output_dir

    try:
        with open(input_dot_file_path, "r") as f:
            dot_graph_string = f.read()
    except FileNotFoundError:
        print(f"错误: 文件 '{input_dot_file_path}' 未找到，请检查文件路径是否正确。")
        exit()

    # 解析 DOT 字符串
    parsed_graph = parse_dot_graph(dot_graph_string)

    # 复制图
    duplicated_graphs = duplicate_graph(parsed_graph)
    graph1_data = duplicated_graphs['graph1']
    graph2_data = duplicated_graphs['graph2']

    # 将两个复制图的数据放入一个字典中，键为图的名称
    combined_graph_data = {
        "graph_copy_1": graph1_data,
        "graph_copy_2": graph2_data
    }

    # 使用 to_dot_string_combined 函数生成包含两个子图的 DOT 字符串
    dot_string_combined_graphs = to_dot_string_combined(combined_graph_data, "combined_duplicated_graphs")

    # 构建输出文件名：输入文件名 + "+aug" + ".dot"
    input_filename_base = os.path.splitext(os.path.basename(input_dot_file_path))[0] # 获取输入文件名 (不含扩展名)
    output_filename = input_filename_base + "+aug.dot"
    
     # 构建完整的输出文件路径：输出目录 + 输出文件名
    output_file_path = os.path.join(output_dir, output_filename) # 使用 os.path.join 拼接路径

    
    # 将 DOT 字符串写入文件
    with open("combined_duplicated_graphs.dot", "w") as f:
        f.write(dot_string_combined_graphs)

    print("包含两个复制图的 DOT 文件 'combined_duplicated_graphs.dot' 已生成。")