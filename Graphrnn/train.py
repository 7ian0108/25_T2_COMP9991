# train.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data import GraphDataset
import time

def train_graph_rnn(
    model,
    num_epochs=10,
    batch_size=1,
    max_num_nodes=10,
    graph_type='path',
    learning_rate=0.001,
    print_every=1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    训练 GraphRNN 模型

    参数：
    - model: 已实例化的模型（sequential 或 parallel）
    - 其他参数与之前相同
    """

    model.to(device)

    # 加载训练数据集
    dataset = GraphDataset(num_graphs=1000, max_num_nodes=max_num_nodes, graph_type=graph_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # BCE 作为损失函数
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        if epoch == 1:
            start_time = time.time()

        for adj_seq in dataloader:
            adj_seq = adj_seq.squeeze(0).to(device)

            # 前向传播
            output = model(adj_seq)

            # 匹配不同模型结构的 loss 逻辑
            if hasattr(model, 'model_type') and model.model_type == 'parallel':
                loss = criterion(output[:, :max_num_nodes], adj_seq)
            else:
                loss = criterion(output, adj_seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch == 10:
            elapsed = time.time() - start_time
            print(f"[Timing] Epoch 1 duration: {elapsed:.4f} seconds")

        if epoch % print_every == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
