"""
PolyGNN architecture.
"""

import torch
import torch.nn.functional as F

from network import Encoder, Decoder, GNN


class PolyGNN(torch.nn.Module):
    """
    PolyGNN.
    """
    def __init__(self, cfg):
        super().__init__()
        self.gnn = cfg.gnn
        self.dataset_suffix = cfg.dataset_suffix
        if cfg.sample.strategy == 'grid':
            self.points_suffix = f'_{cfg.sample.resolution}'
        elif cfg.sample.strategy == 'random':
            self.points_suffix = f'_{cfg.sample.length}'
        else:
            self.points_suffix = ''

        # encoder
        self.encoder = Encoder(backbone=cfg.encoder, latent_dim=cfg.latent_dim_light,
                               use_spatial_transformer=cfg.use_spatial_transformer, convonet_kwargs=cfg.convonet_kwargs)

        # decoder
        latent_dim = cfg.latent_dim_light if cfg.decoder.casefold() == 'MLP'.casefold() else cfg.latent_dim_conv
        self.decoder = Decoder(backbone=cfg.decoder, latent_dim=latent_dim, num_queries=cfg.num_queries)

        # GNN
        if cfg.gnn is not None:
            self.gnn = GNN(backend=cfg.gnn, num_features=latent_dim, num_classes=2, dropout=cfg.dropout)
        else:
            self.gnn = None

    def forward(self, data):
        x = self.encoder(data[f'points{self.points_suffix}'], data[f'batch_points{self.points_suffix}'])
        # x: {(num_graphs_in_batch, 128, 128, 128) * 3} from ConvOEncoder
        # x: (num_graphs_in_batch, 256) from other encoders

        x = self.decoder(x, data.queries, data.batch)
        # x: (num_cells_in_batch, latent_dim)

        if self.gnn:
            x = self.gnn(x, data.edge_index)
            # x: (num_cells_in_batch, 2)
        else:
            # cell-wise voting from query points
            x = torch.mean(x, dim=1)
            # x: (num_cells_in_batch, 1)
            x = self.lin(x)
            # x: (num_cells_in_batch, 2)
            x = F.log_softmax(x, dim=1)
            # x: (num_cells_in_batch, 2)
        return x
