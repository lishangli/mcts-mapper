# import networkx as nx
# import matplotlib.pyplot as plt

# # 创建一个空图
# G = nx.Graph()

# plt.figure()

# # 动态添加节点并更新绘图
# for i in range(10):
#     G.add_node(i)
#     if i > 0:
#         G.add_edge(i, i-1)

#     plt.cla()  # 清除当前图形
#     pos = nx.spring_layout(G)  # 重新计算布局
#     nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500)
#     plt.pause(1)  # 暂停 1 秒

# plt.show()  # 最后显示静态图

import torch
from torchviz import make_dot

# 假设你有一个 PyTorch 模型和输入
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = SimpleModel()
x = torch.randn(1, 10)  # 输入数据
y = model(x)

# 使用 torchviz 可视化
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("model_structure", format="png")  # 保存为 PNG 文件
