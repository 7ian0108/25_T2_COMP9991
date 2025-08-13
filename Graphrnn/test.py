import torch
import networkx as nx
from utils import adj_sequence_to_graph
import random

def test_graph_rnn(model, train_dataset, test_dataset, num_samples=100, threshold=0.5, device='cpu'):
    """
    对训练后的模型进行图生成测试，并输出 Valid, Accurate, Unique, Novel 指标

    参数：
    - model: 已训练模型
    - train_dataset: 用于判断 Novel 的训练集
    - test_dataset: 评估准确性时使用（可选）
    - num_samples: 要生成的图数量
    - threshold: 二值化边连接的阈值
    """
    model.eval()
    generated_graphs = []
    valid_graphs = []
    accurate_matches = 0
    seen_graphs = set()

    # 用于记录训练集中已有图（用于 Novel 计算）
    train_graph_set = set()
    for adj in train_dataset:
        g = adj_sequence_to_graph(adj, threshold)
        train_graph_set.add(nx.to_numpy_array(g).tobytes())

    for _ in range(num_samples):
        # 选一个测试样本（可改为随机采样或 dummy 输入）
        with torch.no_grad():
            sample = random.choice(test_dataset)
            adj_seq = sample.to(device)
            output = model(adj_seq)
            if output.shape != adj_seq.shape:
                output = output[:, :adj_seq.shape[1]]  # 只截取下三角区域

            pred_graph = adj_sequence_to_graph(output, threshold)
            generated_graphs.append(pred_graph)

            # ======= Valid =======
            is_valid = nx.is_connected(pred_graph)
            valid_graphs.append(is_valid)

            # ======= Accurate =======
            true_graph = adj_sequence_to_graph(adj_seq)
            if nx.is_isomorphic(pred_graph, true_graph):
                accurate_matches += 1

    # ======= Unique =======
    unique_graphs = set(nx.to_numpy_array(g).tobytes() for g in generated_graphs)
    unique_ratio = len(unique_graphs) / len(generated_graphs)

    # ======= Novel =======
    novel_graphs = [g for g in unique_graphs if g not in train_graph_set]
    novel_ratio = len(novel_graphs) / len(unique_graphs)

    # ======= Valid, Accurate =======
    valid_ratio = sum(valid_graphs) / len(valid_graphs)
    accurate_ratio = accurate_matches / len(generated_graphs)

    print("\n=== Generation Quality Report ===")
    print(f"Valid:   {valid_ratio:.4f}")
    print(f"Accurate:{accurate_ratio:.4f}")
    print(f"Unique:  {unique_ratio:.4f}")
    print(f"Novel:   {novel_ratio:.4f}")
