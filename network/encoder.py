"""
Point cloud encoders.
"""

import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
from torch_geometric.nn import PointNetConv, XConv, DynamicEdgeConv
from torch_geometric.nn import fps, global_mean_pool, global_max_pool, knn_graph
from torch_geometric.nn import MLP, PointTransformerConv, knn, radius
from torch_geometric.utils import scatter
from torch_geometric.nn.aggr import MaxAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool.decimation import decimation_indices
from torch_geometric.utils import softmax

from network.convonet.pointnet import LocalPoolPointnet


class PointNet(torch.nn.Module):
    """
    PointNet with PointNetConv.
    """

    def __init__(self, latent_dim):
        super().__init__()
        # conv layers
        self.conv1 = PointNetConv(MLP([3 + 3, 64, 64]))
        self.conv2 = PointNetConv(MLP([64 + 3, 64, 128, 1024]))
        self.lin = MLP([1024, latent_dim], plain_last=False)

    def forward(self, pos, batch):
        x, pos, batch = None, pos, batch
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)

        # point-wise features
        x = self.conv1(x, pos, edge_index)
        x = self.conv2(x, pos, edge_index)

        # instance-wise features
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x


class SAModule(torch.nn.Module):
    """
    SA Module for PointNet++.
    """

    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    """
    Global SA Module for PointNet++.
    """

    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2(torch.nn.Module):
    """
    PointNet++ with SAModule and GlobalSAModule.
    """

    def __init__(self, latent_dim):
        super().__init__()

        # input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        self.lin = MLP([1024, latent_dim], plain_last=False)

    def forward(self, pos, batch):
        # point-wise features
        sa0_out = (pos, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = sa3_out

        # instance-wise features
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x


class PointCNN(torch.nn.Module):
    """
    PointCNN with XConv.
    """

    def __init__(self, latent_dim):
        super().__init__()

        self.conv1 = XConv(3, 32, dim=3, kernel_size=8, dilation=2)
        self.conv2 = XConv(32, 128, dim=3, kernel_size=12, dilation=2)
        self.conv3 = XConv(128, 256, dim=3, kernel_size=16, dilation=1)
        self.conv4 = XConv(256, 256, dim=3, kernel_size=16, dilation=2)
        self.conv5 = XConv(256, 256, dim=3, kernel_size=16, dilation=3)
        self.conv6 = XConv(256, 690, dim=3, kernel_size=16, dilation=4)
        self.lin = MLP([690, latent_dim], plain_last=False)

    def forward(self, pos, batch):
        # point-wise features
        x, pos, batch = None, pos, batch
        x = self.conv1(x, pos, batch)
        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = self.conv2(x, pos, batch)
        idx = fps(pos, batch, ratio=0.325)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = self.conv3(x, pos, batch)
        x = self.conv4(x, pos, batch)
        x = self.conv5(x, pos, batch)
        x = self.conv6(x, pos, batch)

        # instance-wise features
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x


class DGCNN(torch.nn.Module):
    """
    DGCNN with DynamicEdgeConv.
    """

    def __init__(self, latent_dim):
        super().__init__()

        self.conv1 = DynamicEdgeConv(Sequential(Linear(3 * 2, 64, bias=True), LeakyReLU(negative_slope=0.02)), k=20)
        self.conv2 = DynamicEdgeConv(Sequential(Linear(64 * 2, 64, bias=True), LeakyReLU(negative_slope=0.02)), k=20)
        self.conv3 = DynamicEdgeConv(Sequential(Linear(64 * 2, 128, bias=True), LeakyReLU(negative_slope=0.02)), k=20)
        self.conv4 = DynamicEdgeConv(Sequential(Linear(128 * 2, 256, bias=True), LeakyReLU(negative_slope=0.02)), k=20)
        self.lin = MLP([512, latent_dim], plain_last=False)

    def forward(self, pos, batch):
        x, batch = pos, batch

        # point-wise features
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)
        x = torch.concat([x1, x2, x3, x4], dim=-1)

        # instance-wise features
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x


class SpatialTransformer(torch.nn.Module):
    """
    Optional spatial transform block in DGCNN to align a point set to a canonical space.
    """

    def __init__(self, k=16):
        super().__init__()
        # to estimate the 3 x 3 matrix, a tensor concatenating the coordinates of each point
        # and the coordinate differences between its k neighboring points is used.
        self.k = k
        self.mlp = Sequential(Linear(k * 6, 1024), ReLU(), Linear(1024, 256), ReLU(), Linear(256, 9))

    def forward(self, pos, batch):
        # pos: (num_batch_points, 3)
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True).to(batch.device)
        neighbours = pos[edge_index[0].reshape([-1, 16])]  # (num_batch_points, k, 3)
        pos = torch.unsqueeze(pos, dim=1) # (num_batch_points, k, 6)
        # concatenating the coordinates of each point and the coordinate differences between its k neighboring points
        x = torch.concat([pos.repeat(1, self.k, 1), pos - neighbours], dim=2)  # (num_batch_points, k, 6)
        x = x.reshape([x.shape[0], -1])  # (num_batch_points, k * 6)
        x = self.mlp(x)  # (num_batch_points, 9)
        x = global_mean_pool(x, batch)  # (batch_size, 9)
        x = x[batch].reshape([-1, 3, 3])  # (num_batch_points, 3, 3)
        x = torch.squeeze(torch.bmm(pos, x))  # (num_batch_points, 3)
        return x


class TransformerBlock(torch.nn.Module):
    """
    Transformer block for PointTransformer.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Linear(in_channels, in_channels)
        self.lin_out = Linear(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)
        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None, plain_last=False)
        self.transformer = PointTransformerConv(in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """
    TransitionDown for PointTransformer.
    Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an MLP to augment features dimensionality.
    """

    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute k-nearest points for each cluster
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0, dim_size=id_clusters.size(0), reduce='max')

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class PointTransformer(torch.nn.Module):
    """
    PointTransformer.
    """

    def __init__(self, latent_dim, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = 1

        # hidden channels
        dim_model = [32, 64, 128, 256, 512]

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)
        self.transformer_input = TransformerBlock(in_channels=dim_model[0], out_channels=dim_model[0])

        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i], out_channels=dim_model[i + 1], k=self.k))
            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]))
        self.lin = MLP([dim_model[-1], latent_dim], plain_last=False)

    def forward(self, pos, batch=None, x=None):
        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # MLP blocks
        out = self.lin(x)
        return out


class SharedMLP(MLP):
    """

    """

    def __init__(self, *args, **kwargs):
        # BN + Act always active even at last layer.
        kwargs['plain_last'] = False
        # LeakyRelu with 0.2 slope by default.
        kwargs['act'] = kwargs.get('act', 'LeakyReLU')
        kwargs['act_kwargs'] = kwargs.get('act_kwargs', {'negative_slope': 0.2})
        # BatchNorm with 1 - 0.99 = 0.01 momentum
        # and 1e-6 eps by defaut (tensorflow momentum != pytorch momentum)
        kwargs['norm_kwargs'] = kwargs.get('norm_kwargs', {'momentum': 0.01, 'eps': 1e-6})
        super().__init__(*args, **kwargs)


class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""

    def __init__(self, channels):
        super().__init__(aggr='add')
        self.mlp_encoder = SharedMLP([10, channels // 2])
        self.mlp_attention = SharedMLP([channels, channels], bias=False,
                                       act=None, norm=None)
        self.mlp_post_attention = SharedMLP([channels, channels])

    def forward(self, edge_index, x, pos):
        out = self.propagate(edge_index, x=x, pos=pos)  # N, d_out
        out = self.mlp_post_attention(out)  # N, d_out
        return out

    def message(self, x_j, pos_i, pos_j, index):
        """Local Spatial Encoding (locSE) and attentive pooling of features.
        Args:
            x_j (Tensor): neighbors features (K,d)
            pos_i (Tensor): centroid position (repeated) (K,3)
            pos_j (Tensor): neighbors positions (K,3)
            index (Tensor): index of centroid positions
                (e.g. [0,...,0,1,...,1,...,N,...,N])
        returns:
            (Tensor): locSE weighted by feature attention scores.
        """
        # Encode local neighborhood structural information
        pos_diff = pos_j - pos_i
        distance = torch.sqrt((pos_diff * pos_diff).sum(1, keepdim=True))
        relative_infos = torch.cat([pos_i, pos_j, pos_diff, distance], dim=1)  # N * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N * K, d
        local_features = torch.cat([x_j, local_spatial_encoding], dim=1)  # N * K, 2d

        # Attention will weight the different features of x
        # along the neighborhood dimension.
        att_features = self.mlp_attention(local_features)  # N * K, d_out
        att_scores = softmax(att_features, index=index)  # N * K, d_out

        return att_scores * local_features  # N * K, d_out


class DilatedResidualBlock(torch.nn.Module):
    """
    Dilated residual block for RandLANet.
    """

    def __init__(self, num_neighbors, d_in: int, d_out: int):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.d_in = d_in
        self.d_out = d_out

        # MLP on input
        self.mlp1 = SharedMLP([d_in, d_out // 8])
        # MLP on input, and the result is summed with the output of mlp2
        self.shortcut = SharedMLP([d_in, d_out], act=None)
        # MLP on output
        self.mlp2 = SharedMLP([d_out // 2, d_out], act=None)

        self.lfa1 = LocalFeatureAggregation(d_out // 4)
        self.lfa2 = LocalFeatureAggregation(d_out // 2)
        self.lrelu = torch.nn.LeakyReLU(**{'negative_slope': 0.2})

    def forward(self, x, pos, batch):
        edge_index = knn_graph(pos, self.num_neighbors, batch=batch, loop=True)

        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out//8
        x = self.lfa1(edge_index, x, pos)  # N, d_out//2
        x = self.lfa2(edge_index, x, pos)  # N, d_out//2
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out

        return x, pos, batch


def decimate(tensors, ptr: Tensor, decimation_factor: int):
    """
    Decimates each element of the given tuple of tensors for RandLANet.
    """
    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)
    tensors_decim = tuple(tensor[idx_decim] for tensor in tensors)
    return tensors_decim, ptr_decim


class RandLANet(torch.nn.Module):
    """
    An adaptation of RandLA-Net for point cloud encoding, which was not addressed in the paper:
    RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds.
    """

    def __init__(self, latent_dim, decimation: int = 4, num_neighbors: int = 16):
        super().__init__()
        self.decimation = decimation
        self.fc0 = Linear(in_features=3, out_features=8)
        # 2 DilatedResidualBlock converges better than 4 on ModelNet
        self.block1 = DilatedResidualBlock(num_neighbors, 8, 32)
        self.block2 = DilatedResidualBlock(num_neighbors, 32, 128)
        self.block3 = DilatedResidualBlock(num_neighbors, 128, 512)
        self.mlp1 = SharedMLP([512, 512])
        self.max_agg = MaxAggregation()
        self.mlp2 = Linear(512, latent_dim)

    def forward(self, pos, batch):
        x = pos
        ptr = torch.where(batch[1:] - batch[:-1])[0] + 1  # use ptr elsewhere
        ptr = torch.cat([torch.tensor([0]).cuda(), ptr])
        ptr = torch.cat([ptr, torch.tensor([len(batch)]).cuda()])
        b1 = self.block1(self.fc0(x), pos, batch)
        b1_decimated, ptr1 = decimate(b1, ptr, self.decimation)

        b2 = self.block2(*b1_decimated)
        b2_decimated, ptr2 = decimate(b2, ptr1, self.decimation)

        b3 = self.block3(*b2_decimated)
        b3_decimated, _ = decimate(b3, ptr2, self.decimation)

        x = self.mlp1(b3_decimated[0])
        x = self.max_agg(x, b3_decimated[2])
        x = self.mlp2(x)

        return x


class Encoder(torch.nn.Module):
    """
    Point cloud encoder.
    """
    def __init__(self, backbone, latent_dim, use_spatial_transformer=False, convonet_kwargs=None):
        super().__init__()
        self.backbone = backbone

        # spatial transformer placeholder
        self.use_spatial_transformer = False
        if use_spatial_transformer:
            self.use_spatial_transformer = True
            self.spatial_transformer = SpatialTransformer()

        backbone_mapping = {
            'pointnet': PointNet,
            'pointnet2': PointNet2,
            'pointcnn': PointCNN,
            'dgcnn': DGCNN,
            'randlanet': RandLANet,
            'pointtransformer': PointTransformer,
            'convonet': lambda d: LocalPoolPointnet(**convonet_kwargs)}
        self.backbone_key = backbone.casefold()
        if self.backbone_key in backbone_mapping:
            self.network = backbone_mapping[self.backbone_key](latent_dim)
        else:
            raise ValueError(f'Unexpected backbone: {self.backbone_key}')

    def forward(self, points, batch_points):
        # points: (total_num_points, 3)
        # batch_points: (total_num_points)

        # spatial transformation
        if self.use_spatial_transformer:
            points = self.spatial_transformer(points, batch_points)

        if self.backbone_key == 'convonet':
            # reshape to split it into a single tensor
            points_split = points.view(batch_points[-1] + 1, -1, 3)
            return self.network(points_split)
        else:
            return self.network(points, batch_points)
