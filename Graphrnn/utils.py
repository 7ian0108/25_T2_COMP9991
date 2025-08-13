import numpy as np
import torch
import networkx as nx

def graph_to_adj_sequence(graph, max_num_nodes):
    """
    将图转换为邻接序列（上三角部分），用于 GraphRNN 的输入

    参数：
    - graph: NetworkX 图对象
    - max_num_nodes: 限定最大节点数（用于补齐）

    返回：
    - adj_seq: Tensor 形状为 [max_num_nodes, max_num_nodes]，仅使用上三角部分
    """
    adj = nx.to_numpy_array(graph)  # 获取邻接矩阵（二维 numpy 数组）
    adj = adj[:max_num_nodes, :max_num_nodes]  # 截断最大节点数
    adj_seq = np.zeros((max_num_nodes, max_num_nodes))  # 初始化邻接序列矩阵

    for i in range(1, min(len(graph.nodes), max_num_nodes)):
        # 只保留上三角（之前节点的连边信息）
        adj_seq[i, :i] = adj[i, :i]

    return torch.FloatTensor(adj_seq)  # 返回为 float 类型 Tensor

def adj_sequence_to_graph(adj_seq, threshold=0.5):
    """
    将邻接序列还原为 NetworkX 图

    参数：
    - adj_seq: Tensor [num_nodes, num_nodes]，只使用下三角有效
    - threshold: 如果是概率形式，超过此阈值视为有边

    返回：
    - graph: 还原的 NetworkX 无向图
    """
    adj_seq = adj_seq.detach().cpu().numpy()  # 转换为 numpy 数组
    num_nodes = adj_seq.shape[0]  # 获取节点数
    G = nx.Graph()  # 初始化无向图

    for i in range(num_nodes):
        G.add_node(i)  # 添加节点

    for i in range(1, num_nodes):
        for j in range(i):
            if adj_seq[i, j] > threshold:  # 只有超过阈值的才加边
                G.add_edge(i, j)

    return G

def generate_simple_graph(num_nodes=5, graph_type='path'):
    """
    生成一个简单图结构，用于训练示例

    参数：
    - num_nodes: 图中的节点数
    - graph_type: 图的类型（path/complete/star/cycle）

    返回：
    - graph: 生成的 NetworkX 图对象
    """
    if graph_type == 'path':
        return nx.path_graph(num_nodes)  # 线性链图
    elif graph_type == 'complete':
        return nx.complete_graph(num_nodes)  # 完全图
    elif graph_type == 'star':
        return nx.star_graph(num_nodes - 1)  # 星形图（中心 + n-1 边）
    elif graph_type == 'cycle':
        return nx.cycle_graph(num_nodes)  # 环图
    else:
        raise ValueError(f'未知图类型: {graph_type}')
