from torch import tensor

node_input = tensor([0, 1, 0, 0, 0]).unsqueeze(0)

print(node_input.shape)