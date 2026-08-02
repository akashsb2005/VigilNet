import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class FraudGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.3)
        self.lin = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        return self.lin(x)