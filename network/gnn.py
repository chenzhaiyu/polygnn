"""
Graph neural networks.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, TAGConv, MLP


class GCN(torch.nn.Module):
    """
    Graph convolution network from the "Semi-supervised
    Classification with Graph Convolutional Networks" paper.
    """

    def __init__(self, num_features, num_classes, dropout=False):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 16)
        self.mlp = MLP([16, 64, 16, num_classes], act='leaky_relu', act_kwargs={'negative_slope': 0.02})

    def forward(self, x, edge_index):
        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.conv3(x, edge_index)

        # MLP layers
        x = self.mlp(x)  # (num_batch_cells, 2)

        # return the log softmax value
        return F.log_softmax(x, dim=1)


class TransformerGCN(torch.nn.Module):
    """
    Graph transformer operator from the "Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification" paper.
    """

    def __init__(self, num_features, num_classes, dropout=False):
        super().__init__()
        self.dropout = dropout
        self.conv1 = TransformerConv(num_features, 16)
        self.conv2 = TransformerConv(16, 16)
        self.conv3 = TransformerConv(16, 16)
        self.mlp = MLP([16, 64, 16, num_classes], act='leaky_relu', act_kwargs={'negative_slope': 0.02})

    def forward(self, x, edge_index):
        # Graph transformer layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.conv3(x, edge_index)

        # MLP layers
        x = self.mlp(x)  # (num_batch_cells, 2)

        # return the log softmax value
        return F.log_softmax(x, dim=1)


class TAGCN(torch.nn.Module):
    """
    Topology adaptive graph convolutional network operator from
    the "Topology Adaptive Graph Convolutional Networks" paper.
    """

    def __init__(self, num_features, num_classes, dropout=False):
        super().__init__()
        self.dropout = dropout
        self.conv1 = TAGConv(num_features, 16)
        self.conv2 = TAGConv(16, 16)
        self.conv3 = TAGConv(16, 16)
        self.mlp = MLP([16, 64, 16, num_classes], act='leaky_relu', act_kwargs={'negative_slope': 0.02})

    def forward(self, x, edge_index):
        # Topology Adaptive Graph Convolutional layers
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.02)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.02)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.conv3(x, edge_index)

        # MLP layers
        x = self.mlp(x)  # (num_batch_cells, 2)

        # return the log softmax value
        return F.log_softmax(x, dim=1)


class GNN(torch.nn.Module):
    def __init__(self, backend, num_features, num_classes, **kwargs):
        super().__init__()
        self.backend = backend

        if backend.casefold() == 'GCN'.casefold():
            self.network = GCN(num_features, num_classes, *kwargs)
        elif backend.casefold() == 'TransformerGCN'.casefold():
            self.network = TransformerGCN(num_features, num_classes, *kwargs)
        elif backend.casefold() == 'TAGCN'.casefold():
            self.network = TAGCN(num_features, num_classes, *kwargs)
        else:
            raise ValueError(f'Expected backend GCN, GraphTransformer or TAGCN, got {backend} instead')

    def forward(self, x, data):
        x = torch.squeeze(x)  # (num_cells_in_batch, num_queries_per_cell)
        return self.network(x, data)
