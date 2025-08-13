import torch
from torch.utils.data import Dataset
import random
from utils import graph_to_adj_sequence, generate_simple_graph

class GraphDataset(Dataset):
    """
    自定义图数据集类，继承自 PyTorch 的 Dataset，用于 GraphRNN 输入训练
    """
    def __init__(self, num_graphs=1000, max_num_nodes=10, graph_type='path'):
        """
        初始化图数据集

        参数：
        - num_graphs: 要生成的图数量
        - max_num_nodes: 每个图最多允许的节点数（用于补齐）
        - graph_type: 图的类型（path, complete, star, cycle）
        """
        self.graphs = []  # 用于存放生成的图对象
        self.max_num_nodes = max_num_nodes  # 最大节点数

        for _ in range(num_graphs):
            # 随机生成图的节点数（防止全部一样）
            num_nodes = random.randint(4, max_num_nodes)
            graph = generate_simple_graph(num_nodes=num_nodes, graph_type=graph_type)
            self.graphs.append(graph)

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        根据索引返回一个图的邻接序列

        返回值：
        - adj_seq: Tensor 形状为 [max_num_nodes, max_num_nodes]
        """
        graph = self.graphs[idx]  # 取出第 idx 个图
        adj_seq = graph_to_adj_sequence(graph, self.max_num_nodes)  # 转换为邻接序列
        return adj_seq  # 返回邻接序列（作为模型输入）
