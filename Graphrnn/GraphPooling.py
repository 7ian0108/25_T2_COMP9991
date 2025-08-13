import torch
import torch.nn.functional as F
from torch_geometric.nn.dense import dense_diff_pool
import matplotlib.pyplot as plt
import networkx as nx

# ======= 1. 构造一个 8 节点的原始图 =======
num_nodes = 8
feat_dim = 4
cluster_num = 4  # 池化后减少一半节点

# 原始节点特征 [1, N, F]
x = torch.randn(1, num_nodes, feat_dim)

# 原始邻接矩阵 [1, N, N] (无向)
adj = torch.randint(0, 2, (1, num_nodes, num_nodes)).float()
adj = (adj + adj.transpose(1, 2)) / 2
adj = (adj > 0).float()

# ======= 2. 模拟 DiffPool 的 assignment 矩阵 S =======
S = torch.softmax(torch.randn(1, num_nodes, cluster_num), dim=-1)

# ======= 3. DiffPool 池化 =======
x_pool, adj_pool, link_loss, ent_loss = dense_diff_pool(x, adj, S)

# ======= 4. Unpool 回原始大小 =======
x_unpool = torch.matmul(S, x_pool)        # 还原节点特征
adj_unpool = torch.matmul(S, torch.matmul(adj_pool, S.transpose(1,2)))  # 还原邻接

print(f"原始图节点数: {num_nodes}, 池化后节点数: {adj_pool.shape[1]}")

# ======= 5. 可视化函数 =======
def visualize_adj(adj_matrix, title, threshold=0.3):
    """把邻接矩阵转成 networkx 图可视化"""
    adj_np = adj_matrix.cpu().detach().numpy()
    if threshold is not None:
        adj_np = (adj_np > threshold).astype(float)  # 阈值化
    G = nx.from_numpy_array(adj_np)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(4,4))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

# ======= 6. 画图对比 =======
visualize_adj(adj[0], "yuanshi")
visualize_adj(adj_pool[0], "pooling")
visualize_adj(adj_unpool[0], "Unpool")

# ======= 7. 还原误差 =======
diff = torch.abs(adj - adj_unpool)
print("还原误差 (邻接矩阵差值)：\n", diff[0])
