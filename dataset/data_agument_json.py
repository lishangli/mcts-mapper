import json
import argparse

def expand_dataflow_graph(json_data):
    """
    扩展数据流图 JSON，通过复制子图一次。

    参数:
        json_data (dict): 表示 JSON 格式数据流图的字典。

    返回:
        dict: 扩展后的字典，表示包含两个相同子图的数据流图。
    """

    graph = json_data.copy()
    original_objects = graph['objects']
    original_edges = graph['edges']

    # 找到最大的 _gvid 以偏移新的 gvid
    max_gvid = -1
    for obj in original_objects:
        max_gvid = max(max_gvid, obj['_gvid'])
    for edge in original_edges:
        max_gvid = max(max_gvid, edge['_gvid'])

    offset = max_gvid + 1

    duplicated_objects = []
    object_gvid_map = {}  # 映射原始 gvid 到复制后的 gvid

    # 复制对象并更新 _gvid 和 name
    for obj in original_objects:
        original_gvid = obj['_gvid']
        new_obj = obj.copy()
        new_gvid = obj['_gvid'] + offset
        new_obj['_gvid'] = new_gvid

        new_name = obj['name'] + "_copy2"
        new_obj['name'] = new_name

        duplicated_objects.append(new_obj)
        object_gvid_map[original_gvid] = new_gvid

    duplicated_edges = []

    # 复制边并更新 _gvid, tail, 和 head 以指向复制的对象
    for edge in original_edges:
        new_edge = edge.copy()
        new_gvid = edge['_gvid'] + offset
        new_edge['_gvid'] = new_gvid
        new_edge['tail'] = object_gvid_map.get(edge['tail'], edge['tail'] + offset if edge['tail'] >= 0 else edge['tail']) # 对象中不应该有负 gvid
        new_edge['head'] = object_gvid_map.get(edge['head'], edge['head'] + offset if edge['head'] >= 0 else edge['head']) # 对象中不应该有负 gvid

        duplicated_edges.append(new_edge)

    # 添加复制的对象和边到图中
    graph['objects'].extend(duplicated_objects)
    graph['edges'].extend(duplicated_edges)

    return graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="扩展数据流图 JSON 文件，通过复制子图一次。")
    parser.add_argument("input_file", help="输入 JSON 文件路径")
    parser.add_argument("output_file", help="输出 JSON 文件路径")

    args = parser.parse_args()

    input_filepath = args.input_file
    output_filepath = args.output_file

    try:
        with open(input_filepath, 'r') as f:
            data_flow_json = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_filepath}' 未找到。")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误: 输入文件 '{input_filepath}' 不是有效的 JSON 文件。")
        exit(1)

    expanded_graph_json = expand_dataflow_graph(data_flow_json)

    try:
        with open(output_filepath, 'w') as f:
            json.dump(expanded_graph_json, f, indent=2)
        print(f"扩展后的数据流图已保存到 '{output_filepath}'")
    except Exception as e:
        print(f"错误: 保存输出文件 '{output_filepath}' 失败: {e}")
        exit(1)