import torch
import torch.nn as nn

class ParallelGraphRNN(nn.Module):
    """
    改进版 GraphRNN 模型：
    - 保留节点级 GRU 结构（建模节点间的上下文）
    - 使用 Bilinear 并行预测所有边（不是逐个预测）
    """
    def __init__(self, node_input_size=10, node_hidden_size=64, max_num_nodes=10):
        """
        参数：
        - node_input_size: 节点输入特征维度（默认为 max_num_nodes）
        - node_hidden_size: GRU 隐状态维度
        - max_num_nodes: 图中最大节点数
        """
        super(ParallelGraphRNN, self).__init__()

        self.max_num_nodes = max_num_nodes

        # 节点级 GRU，batch_first=True 允许输入 [B, T, F]
        self.node_rnn = nn.GRU(input_size=node_input_size, hidden_size=node_hidden_size, batch_first=True)

        # 双线性边解码器：输入两个节点的隐藏状态，输出边概率
        self.edge_decoder = nn.Bilinear(node_hidden_size, node_hidden_size, 1)

        # 激活函数：将边 logits 映射为概率
        self.sigmoid = nn.Sigmoid()

    def forward(self, adj_seq):
        """
        前向传播（并行版本）：
        - adj_seq: [N, max_prev_node]，真实邻接序列作为输入
        - 返回值： [N, N]，所有边概率矩阵（只用下三角）
        """
        device = adj_seq.device
        num_nodes = adj_seq.size(0)

        # 构造 batch 输入给 GRU：形状 [1, N, max_prev_node]
        input_seq = adj_seq.unsqueeze(0)

        # 通过节点级 GRU 计算所有节点的隐藏状态 [1, N, H]
        rnn_out, _ = self.node_rnn(input_seq)
        node_embeddings = rnn_out.squeeze(0)  # [N, H]

        # 初始化边概率矩阵
        edge_probs = torch.zeros((num_nodes, num_nodes), device=device)

        # 使用 Bilinear 解码器并行计算所有边（仅下三角）
        for i in range(num_nodes):
            for j in range(i):  # 只处理下三角（i > j）
                prob = self.edge_decoder(node_embeddings[i], node_embeddings[j])  # [1]
                edge_probs[i, j] = self.sigmoid(prob)

        return edge_probs  # [N, N] 下三角为边概率
