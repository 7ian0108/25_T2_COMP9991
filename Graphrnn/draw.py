import torch
import networkx as nx
import matplotlib.pyplot as plt

print("plt type:", type(plt))

def draw_ground_truth_graph(adj_seq, threshold=0.5):
    adj_seq = adj_seq.detach().cpu().numpy()
    num_nodes = adj_seq.shape[0]
    G = nx.Graph()

    for i in range(num_nodes):
        G.add_node(i)
    for i in range(num_nodes):
        for j in range(i):
            if adj_seq[i, j] >= threshold:
                G.add_edge(i, j)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(4, 4))  # ✅ 显式创建图像窗口
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=600, font_size=12)
    plt.title("Ground Truth Graph")
    plt.axis('off')
    plt.show()
