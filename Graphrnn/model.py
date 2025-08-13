import torch
import torch.nn as nn

class GraphRNN(nn.Module):
    """
    简化版 GraphRNN 模型，包括：
    - 节点级 RNN（node-level RNN）：生成每一个新节点的状态；
    - 边级 RNN（edge-level RNN）：生成新节点与之前节点之间的边。
    """
    def __init__(self, node_hidden_size=64, edge_hidden_size=64, max_prev_node=10):
        """
        初始化 GraphRNN 模型

        参数：
        - node_hidden_size: 节点级 GRU 隐状态维度
        - edge_hidden_size: 边级 GRU 隐状态维度
        - max_prev_node: 每个节点最多可能连接的之前节点数（用来限制输入维度）
        """
        super(GraphRNN, self).__init__()

        self.max_prev_node = max_prev_node  # 记录最大可连接节点数

        # 节点级 RNN（GRUCell）
        self.node_rnn = nn.GRUCell(input_size=max_prev_node, hidden_size=node_hidden_size)

        # 边级 RNN（GRUCell）
        self.edge_rnn = nn.GRUCell(input_size=1, hidden_size=edge_hidden_size)

        # 边级输出层：将边的隐藏状态映射为概率
        self.edge_output = nn.Linear(edge_hidden_size, 1)

        # 激活函数（用于边预测）
        self.sigmoid = nn.Sigmoid()

    def forward(self, adj_seq):
        """
        前向传播函数

        参数：
        - adj_seq: Tensor，形状为 [num_nodes, max_prev_node]，每行表示一个节点与前面节点的连接情况

        返回：
        - output_seq: Tensor，形状为 [num_nodes, max_prev_node]，每个值是预测的连接概率
        """
        num_nodes = adj_seq.size(0)  # 获取图中节点数
        device = adj_seq.device  # 获取当前 tensor 所在设备

        # 初始化节点级隐状态为 0
        node_hidden = torch.zeros((1, self.node_rnn.hidden_size)).to(device)

        output_seq = []  # 保存所有预测边的输出结果

        for i in range(num_nodes):
            # 当前节点的输入是与前面节点的连接状态
            node_input = adj_seq[i].unsqueeze(0)  # 形状变为 [1, max_prev_node]

            # 节点级 GRU 更新隐藏状态
            node_hidden = self.node_rnn(node_input, node_hidden)

            # 初始化边级隐藏状态为 0
            edge_hidden = torch.zeros((1, self.edge_rnn.hidden_size)).to(device)
            edge_outputs = []  # 当前节点每条边的输出

            for j in range(self.max_prev_node):
                edge_input = node_input[:, j].view(-1, 1)  # 取第 j 个前向连接作为输入
                edge_hidden = self.edge_rnn(edge_input, edge_hidden)  # 更新边级隐藏状态
                edge_out = self.sigmoid(self.edge_output(edge_hidden))  # 预测边的概率
                edge_outputs.append(edge_out)

            # 将当前节点所有边输出合并为一行
            edge_outputs = torch.cat(edge_outputs, dim=1)  # [1, max_prev_node]
            output_seq.append(edge_outputs)

        # 所有节点的边输出拼接为 [num_nodes, max_prev_node]
        return torch.cat(output_seq, dim=0)
