import torch
from data import GraphDataset
from draw import draw_ground_truth_graph  # 你自己在 draw.py 中定义的函数

def test_draw_ground_truth():
    # 初始化数据集（可改为其他图类型）
    dataset = GraphDataset(num_graphs=1, max_num_nodes=8, graph_type='star')

    # 获取一个样本
    adj_seq = dataset[0]  # Tensor, shape: [max_num_nodes, max_prev_node]

    print("真实邻接序列如下：")
    print(adj_seq)

    # 绘图
    draw_ground_truth_graph(adj_seq)

if __name__ == '__main__':
    test_draw_ground_truth()
