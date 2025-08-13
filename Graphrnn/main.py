# main.py

import torch
from model import GraphRNN
from model_parallel import ParallelGraphRNN
from data import GraphDataset
from train import train_graph_rnn
from test import test_graph_rnn

def main():
    # === 参数配置 ===
    num_epochs = 2
    batch_size = 1
    max_num_nodes = 10
    graph_type = 'star'
    learning_rate = 0.0001
    model_type = 'sequential'  # 'sequential' 或 'parallel'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # === 初始化模型 ===
    if model_type == 'sequential':
        model = GraphRNN(
            node_hidden_size=64,
            edge_hidden_size=64,
            max_prev_node=max_num_nodes
        )
    elif model_type == 'parallel':
        model = ParallelGraphRNN(
            node_input_size=max_num_nodes,
            node_hidden_size=64,
            max_num_nodes=max_num_nodes
        )
        model.model_type = 'parallel'  # 用于训练时判断
    else:
        raise ValueError("Unknown model_type")

    # === 训练模型 ===
    train_graph_rnn(
        model=model,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_num_nodes=max_num_nodes,
        graph_type=graph_type,
        learning_rate=learning_rate,
        device=device
    )

    # === 测试模型 ===
    train_dataset = GraphDataset(num_graphs=1000, max_num_nodes=max_num_nodes, graph_type=graph_type)
    test_dataset = GraphDataset(num_graphs=100, max_num_nodes=max_num_nodes, graph_type=graph_type)

    test_graph_rnn(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_samples=100,
        threshold=0.5,
        device=device
    )

if __name__ == '__main__':
    main()
