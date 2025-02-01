"""
Polyhedral feature decoders.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP

from network.convonet.ResnetBlockFC import ResnetBlockFC
from utils import normalize_coordinate, normalize_3d_coordinate


class MLPDecoder(torch.nn.Module):
    """
    Global decoder to fuse queries and latent. Adapted from IM-Net.
    """

    def __init__(self, latent_dim, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.mlp = MLP([latent_dim + self.num_queries * 3, latent_dim * 4, latent_dim * 4, latent_dim * 2, latent_dim], plain_last=False)

    def forward(self, latent, queries, batch):
        # concat queries and latent
        latent = latent[batch]  # (num_cells_in_batch, 256)
        queries = queries.view(queries.shape[0], -1)  # (num_cells_in_batch, num_queries_per_cell * 3)
        pointz = torch.cat([queries, latent], 1)  # (num_cells_in_batch, num_queries_per_cell * 3 + 256)
        return self.mlp(pointz)  # (num_cells_in_batch, num_queries_per_cell, 1)


class ConvONetDecoder(torch.nn.Module):
    """
    Convolutional occupancy decoder. Instead of conditioning on global features, on plane/volume local features.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bi-linear | nearest
        padding (float): conventional padding parameter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(self, dim=3, c_dim=128, hidden_size=256, latent_dim=4096, num_queries = 16 ,n_blocks=5,
                 leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.num_queries = num_queries
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = torch.nn.ModuleList([
                torch.nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)
            ])

        self.fc_p = torch.nn.Linear(dim, hidden_size)

        self.blocks = torch.nn.ModuleList([
            ResnetBlockFC(hidden_size) for _ in range(n_blocks)
        ])
        if latent_dim != hidden_size * num_queries:
            # not recommended as breaks explicit per-query feature
            self.fc_out = torch.nn.Linear(hidden_size * num_queries, latent_dim)
        else:
            self.fc_out = None

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # bilinear interpolation
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(
            -1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        # aggregate to cell-wise features
        out = net.squeeze()
        out = out.view(out.shape[0] // self.num_queries , -1)  # (num_cells, latent_b_dim * num_queries == 4096)
        if self.fc_out is not None:
            out = self.fc_out(out)

        return out


class Decoder(torch.nn.Module):
    """
    Global decoder to fuse queries and latent. Adapted from IM-Net.
    """

    def __init__(self, backbone, latent_dim, num_queries):
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim
        if backbone.casefold() == 'MLP'.casefold():
            self.network = MLPDecoder(latent_dim=latent_dim, num_queries=num_queries)
        elif backbone.casefold() == 'ConvONet'.casefold():
            self.network = ConvONetDecoder(latent_dim=latent_dim, num_queries=num_queries)
        else:
            raise ValueError(f'Unexpected backbone: {backbone}')

    def forward(self, latent, queries, batch):
        # latent: (num_graphs_in_batch, latent_a_dim)
        # queries: (num_cells_in_batch, num_queries_per_cell, 3)
        # batch: (num_cells_in_batch, 1)

        if self.backbone.casefold() == 'MLP'.casefold():
            outs = self.network(latent, queries, batch)

        else:
            # occupancy network decoder
            outs = torch.zeros([len(batch), self.latent_dim]).to(batch.device)
            for i in range(batch[-1] + 1):
                latent_i = {'xz': latent['xz'][i].unsqueeze(0), 'xy': latent['xy'][i].unsqueeze(0),
                            'yz': latent['yz'][i].unsqueeze(0)}
                queries_i = queries[batch == i].view(-1, 3).unsqueeze(0)
                outs[batch == i] = self.network(queries_i, latent_i)

        # (num_cells_in_batch, num_features_per_cell == latent_b_dim)
        return outs
