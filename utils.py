"""
Utility functions.
"""

import os
import time
import glob
from pathlib import Path
import atexit
from itertools import repeat
import math
import logging
import random
import pickle
import csv
import multiprocessing

from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig
import torch_geometric as pyg
import trimesh
from plyfile import PlyData

from abspy import VertexGroup, CellComplex


def setup_runner(rank, world_size, master_addr, master_port):
    """
    Set up runner for distributed parallelization.
    """
    # initialize torch.distributed
    os.environ['MASTER_ADDR'] = str(master_addr)
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def attach_to_log(level=logging.INFO,
                  filepath=None,
                  colors=True,
                  capture_warnings=True):
    """
    Attach a stream handler to all loggers.

    Parameters
    ------------
    level : enum (int)
        Logging level, like logging.INFO
    colors : bool
        If True try to use colorlog formatter
    capture_warnings: bool
        If True capture warnings
    filepath: None or str
        path to save the logfile

    Returns
    -------
    logger: Logger object
        Logger attached with a stream handler
    """

    # make sure we log warnings from the warnings module
    logging.captureWarnings(capture_warnings)

    # create a basic formatter
    formatter_file = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
        "%Y-%m-%d %H:%M:%S")
    if colors:
        try:
            from colorlog import ColoredFormatter
            formatter_stream = ColoredFormatter(
                ("%(log_color)s%(levelname)-8s%(reset)s " +
                 "%(filename)17s:%(lineno)-4s  %(blue)4s%(message)s"),
                datefmt=None,
                reset=True,
                log_colors={'DEBUG': 'cyan',
                            'INFO': 'green',
                            'WARNING': 'yellow',
                            'ERROR': 'red',
                            'CRITICAL': 'red'})
        except ImportError:
            formatter_stream = formatter_file
    else:
        formatter_stream = formatter_file

    # if no handler was passed use a StreamHandler
    logger = logging.getLogger()
    logger.setLevel(level)

    if not any([isinstance(handler, logging.StreamHandler) for handler in logger.handlers]):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter_stream)
        logger.addHandler(stream_handler)

    if filepath and not any([isinstance(handler, logging.FileHandler) for handler in logger.handlers]):
        file_handler = logging.FileHandler(filepath)
        file_handler.setFormatter(formatter_file)
        logger.addHandler(file_handler)

    # set nicer numpy print options
    np.set_printoptions(precision=5, suppress=True)

    return logger


def edge_index_from_dict(graph_dict):
    """
    Convert adjacency dict to edge index.

    Parameters
    ----------
    graph_dict: dict
        Adjacency dict

    Returns
    -------
        as_tensor: torch.Tensor
        Edge index
    """
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index


def index_to_mask(index, size):
    """
    Convert index to binary mask.

    Parameters
    ----------
    index: range Object
        Index of 1s
    size: int
        Size of mask

    Returns
    -------
    as_tensor: torch.Tensor
        Binary mask
    """
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def freeze_vram(cuda_devices, timeout=500):
    """
    Freeze VRAM for a short time at program exit. For debugging.

    Parameters
    ----------
    cuda_devices: list of int
        Indices of CUDA devices
    timeout: int
        Timeout seconds
    """
    torch.cuda.empty_cache()
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader') \
        .read().strip().split("\n")
    for i, device in enumerate(cuda_devices):
        total, used = devices_info[int(device)].split(',')
        total = int(total)
        used = int(used)
        max_mem = int(total * 0.90)
        block_mem = max_mem - used
        if block_mem > 0:
            x = torch.FloatTensor(256, 1024, block_mem).to(torch.device(f'cuda:{i}'))
            del x
    for _ in tqdm(range(timeout), desc='VRAM freezing'):
        time.sleep(1)


def init_device(gpu_ids, register_freeze=False):
    """
    Init devices.

    Parameters
    ----------
    gpu_ids: list of int
        GPU indices to use
    register_freeze: bool
        Register GPU memory freeze if set True
    """
    # set multiprocessing sharing strategy
    torch.multiprocessing.set_sharing_strategy('file_system')

    # does not work for DP after import torch with PyTorch 2.0, but works for DDP nevertheless
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)[1:-1]

    # raise soft limit from 1024 to 4096 for open files to address RuntimeError: received 0 items of ancdata
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    if register_freeze:
        atexit.register(freeze_vram, gpu_ids)


class Sampler:
    """
    Sampler to sample points from a point cloud.
    """

    def __init__(self, strategy, length, ratio, resolutions, duplicate, seed=None):
        self.strategy = strategy
        self.length = length
        self.ratio = ratio
        self.resolutions = resolutions
        self.duplicate = duplicate

        # seed once in initialization
        self.seed = seed

    def sample(self, data):
        with torch.no_grad():
            if self.seed is not None:
                set_seed(self.seed)
            if self.strategy is None:
                return data
            if self.strategy == 'fps':
                return self.farthest_sample(data)
            elif self.strategy == 'random':
                return self.random_sample(data)
            elif self.strategy == 'grid':
                return self.grid_sample(data)
            else:
                raise ValueError(f'Unexpected sampling strategy={self.strategy}.')

    def random_sample(self, data):
        """
        Random uniform sampling.
        """
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/fixed_points.html#FixedPoints
        if not self.duplicate:
            choice = torch.randperm(data.num_points)[:self.length]
        else:
            choice = torch.cat(
                [torch.randperm(data.num_points) for _ in range(math.ceil(self.length / data.num_points))],
                dim=0)[:self.length]
        data[f'batch_points_{self.length}'] = data.batch_points[choice]
        data[f'points_{self.length}'] = data.points[choice]
        return data

    def grid_sample(self, data):
        """
        Sampling points into fixed-sized voxels.
        Each cluster returned is the cluster barycenter.
        """
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/grid_sampling.html#GridSampling
        for size in self.resolutions:
            c = pyg.nn.voxel_grid(data.points, size, data.batch_points, None, None)
            _, perm = pyg.nn.pool.consecutive.consecutive_cluster(c)
            data[f'batch_points_{size}'] = data.batch_points[perm]
            data[f'points_{size}'] = data.points[perm]
        return data

    def farthest_sample(self, data):
        """
        Farthest sampling which iteratively samples the most distant point with regard to the rest points. Inplace.
        """
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.fps
        perm = pyg.nn.pool.fps(data.points, data.batch_points, ratio=self.ratio)
        data[f'batch_points_fps'] = data.batch_points[perm]
        data[f'points_fps'] = data.points[perm]
        return data


def reverse_translation_and_scale(mesh):
    """
    Translation and scale for reverse normalisation of mesh.
    """
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = - (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)

    # scale to unit cube
    scale = bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)

    return translation, scale_trafo


def normalise_mesh(args):
    """
    Normalize mesh (or point cloud).
    First translation, then scaling, if any.

    Parameters
    ----------
    args[0]: input_path: str or Path
        Path to input mesh (results from reconstruction, normalised)
    args[1]: reference_path: str or Path
        Path to reference mesh (used to determine the transformation)
    args[2]: output_path: str or Path
        Path to output mesh (final reconstruction, with reversed normalisation)
    args[3]: force: str
        Force loading type ('mesh' or 'scene')
    args[4]: offset: list
        Coordinate offset (x, y, z)
    args[5]: scaling: bool
        Switch on scaling if set True
    args[6]: translation: bool
        Switch on translation if set True
    """
    input_path, reference_path, output_path, force, offset, is_scaling, is_translation = args

    reference_mesh = trimesh.load(reference_path)
    translation, scale_trafo = reverse_translation_and_scale(reference_mesh)
    if offset is not None:
        translation[0][-1] = translation[0][-1] + offset[0]
        translation[1][-1] = translation[1][-1] + offset[1]
        translation[2][-1] = translation[2][-1] + offset[2]

    # trimesh built-in transform would result in an issue of missing triangles
    with open(input_path, 'r') as fin:
        lines = fin.readlines()
    lines_ = []
    for line in lines:
        if line.startswith('v'):
            vertices = np.array(line.split()[1:], dtype=float)
            if is_translation is True:
                vertices = vertices - translation[:3, 3]
            if is_scaling is True:
                vertices = vertices / scale_trafo[0][0]
            line_ = f'v {vertices[0]} {vertices[1]} {vertices[2]}\n'
        else:
            line_ = line
        lines_.append(line_)
    with open(output_path, 'w') as fout:
        fout.writelines(lines_)


def reverse_normalise_mesh(args):
    """
    Reverse normalisation for reconstructed mesh.
    First scaling, then translation, if any.

    Parameters
    ----------
    args[0]: input_path: str or Path
        Path to input mesh (results from reconstruction, normalised)
    args[1]: reference_path: str or Path
        Path to reference mesh (used to determine the transformation)
    args[2]: output_path: str or Path
        Path to output mesh (final reconstruction, with reversed normalisation)
    args[3]: force: str
        Force loading type ('mesh' or 'scene')
    args[4]: offset: list
        Coordinate offset (x, y, z)
    args[5]: scaling: bool
        Switch on scaling if set True
    args[6]: translation: bool
        Switch on translation if set True
    """
    input_path, reference_path, output_path, force, offset, is_scaling, is_translation = args

    reference_mesh = trimesh.load(reference_path)
    translation, scale_trafo = reverse_translation_and_scale(reference_mesh)
    if offset is not None:
        translation[0][-1] = translation[0][-1] + offset[0]
        translation[1][-1] = translation[1][-1] + offset[1]
        translation[2][-1] = translation[2][-1] + offset[2]

    # trimesh built-in transform would result in an issue of missing triangles
    with open(input_path, 'r') as fin:
        lines = fin.readlines()
    lines_ = []
    for line in lines:
        if line.startswith('v'):
            vertices = np.array(line.split()[1:], dtype=float)
            if is_scaling is True:
                vertices = vertices * scale_trafo[0][0]
            if is_translation is True:
                vertices = vertices + translation[:3, 3]
            line_ = f'v {vertices[0]} {vertices[1]} {vertices[2]}\n'
        else:
            line_ = line
        lines_.append(line_)
    with open(output_path, 'w') as fout:
        fout.writelines(lines_)


def reverse_normalise_cloud(args):
    """
    Reverse normalisation for normalised point cloud.

    Parameters
    ----------
    args[0]: input_path: str or Path
        Path to input point cloud (normalised)
    args[1]: reference_path: str or Path
        Path to reference mesh (used to determine the transformation)
    args[2]: output_path: str or Path
        Path to output point cloud (with reversed normalisation)
    args[3]: force: str
        Force loading type ('mesh' or 'scene')
    args[4]: offset: list
        Coordinate offset (x, y, z)
    """
    input_path, reference_path, output_path, force, offset = args
    plydata = PlyData.read(input_path)['vertex']
    points = np.array([plydata['x'], plydata['y'], plydata['z']]).T
    cloud = trimesh.PointCloud(points)
    reference_mesh = trimesh.load(reference_path)
    translation, scale_trafo = reverse_translation_and_scale(reference_mesh)
    if offset is not None:
        translation[0][-1] = translation[0][-1] + offset[0]
        translation[1][-1] = translation[1][-1] + offset[1]
        translation[2][-1] = translation[2][-1] + offset[2]

    cloud.apply_transform(scale_trafo)
    cloud.apply_transform(translation)
    cloud.export(str(output_path))


def coerce(data):
    """
    Coercion for legacy data.
    """
    data.points = torch.as_tensor(data.points, dtype=torch.float)
    data.queries = torch.as_tensor(data.queries, dtype=torch.float)

    if not hasattr(data, 'num_points'):
        data.num_points = len(data.points)
    if not hasattr(data, 'batch_points'):
        data.batch_points = torch.zeros(len(data.points), dtype=torch.long)
    return data


def set_seed(seed: int) -> None:
    """
    Set singular seed to fix randomness.
    May need to be repeatedly invoked (at least for np.random).
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def append_labels(args):
    """
    Append occupancy labels to existing cell complex file.

    Parameters
    ----------
    args[0]: input_cc: str or Path
        Path to input CC file
    args[1]: input_manifold: str or Path
        Path to input manifold mesh file
    args[2]: output_cc: str or Path
        Path to output CC file
    """
    with open(args[0], 'rb') as handle:
        cell_complex = pickle.load(handle)
    cells_in_mesh = cell_complex.cells_in_mesh(args[1])
    # one-hot encoding
    labels = [0] * cell_complex.num_cells
    for i in cells_in_mesh:
        labels[i] = 1
    cell_complex.labels = labels
    cell_complex.save(args[2])


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def multi_append_labels(cfg: DictConfig):
    # initialize logging
    logger = logging.getLogger('Labels')

    filenames = glob.glob('data/munich_16star/raw/05_complexes/*.cc')
    args = []
    for filename_input in filenames:
        stem = Path(filename_input).stem
        filename_output = Path(filename_input).with_suffix('.cc.new')
        filename_manifold = Path(filename_input).parent.parent / '03_meshes_manifold' / (stem + '.obj')
        if not filename_output.exists():
            args.append((filename_input, filename_manifold, filename_output))

    logger.info('Start complex labeling')
    with multiprocessing.Pool(processes=cfg.num_workers if cfg.num_workers else multiprocessing.cpu_count()) as pool:
        # call with multiprocessing
        for _ in tqdm(pool.imap_unordered(append_labels, args), desc='Appending labels', total=len(args)):
            pass

    # exit with a message
    logger.info('Done complex labeling')


def append_samples(args):
    """
    Append multi-grid-size samples to existing data file.

    Parameters
    ----------
    args[0]: input_path: str or Path
        Path to input torch file
    args[1]: output_path: str or Path
        Path to output torch file
    args[2]: sample_func: func
        Function to sample
    """
    data = torch.load(args[0])
    data = coerce(data)
    data = args[2](data)
    torch.save(data, args[1])


def append_queries(args):
    """
    Append queries to existing data files.

    Parameters
    ----------
    args[0]: input_path: str or Path
        Path to input torch file
    args[1]: complex_dir: str or Path
        Dir to complexes
    args[2]: output_path: str or Path
        Path to output torch file
    """
    data = torch.load(args[0])
    with open(os.path.join(args[1], data.name + '.cc'), 'rb') as handle:
        cell_complex = pickle.load(handle)

    queries_random = np.array(cell_complex.cell_representatives(location='random_t', num=16))
    queries_boundary = np.array(cell_complex.cell_representatives(location='boundary', num=16))
    queries_skeleton = np.array(cell_complex.cell_representatives(location='skeleton', num=16))
    data.queries_random = torch.as_tensor(queries_random, dtype=torch.float)
    data.queries_boundary = torch.as_tensor(queries_boundary, dtype=torch.float)
    data.queries_skeleton = torch.as_tensor(queries_skeleton, dtype=torch.float)

    torch.save(data, args[2])


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def multi_append_queries(cfg: DictConfig):
    # initialize logging
    logger = logging.getLogger('Querying')

    filenames = glob.glob(f'{cfg.data_dir}/processed/*[0-9].pt')
    args = []
    for filename_input in filenames:
        filename_output = Path(filename_input).with_suffix('.pt.queries_appended')
        if not filename_output.exists():
            args.append((filename_input, cfg.complex_dir, filename_output))

    logger.info('Start polyhedra sampling')
    with multiprocessing.Pool(processes=cfg.num_workers if cfg.num_workers else multiprocessing.cpu_count()) as pool:
        # call with multiprocessing
        for _ in tqdm(pool.imap_unordered(append_queries, args), desc='Appending queries', total=len(args)):
            pass

    # exit with a message
    logger.info('Done polyhedra sampling')


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def multi_append_samples(cfg: DictConfig):
    # initialize logging
    logger = logging.getLogger('Sampling')

    filenames = glob.glob(f'{cfg.data_dir}/processed/*[0-9].pt')
    sampler = Sampler(strategy=cfg.sample.strategy, length=cfg.sample.length, ratio=cfg.sample.ratio,
                      resolutions=cfg.sample.resolutions, duplicate=cfg.sample.duplicate, seed=cfg.seed)
    args = []
    for filename_input in filenames:
        filename_output = Path(filename_input).with_suffix('.pt.samples_appended')
        if not filename_output.exists():
            args.append((filename_input, filename_output, sampler.sample))

    logger.info('Start point cloud sampling')
    with multiprocessing.Pool(processes=cfg.num_workers if cfg.num_workers else multiprocessing.cpu_count()) as pool:
        # call with multiprocessing
        for _ in tqdm(pool.imap_unordered(append_samples, args), desc='Appending samples', total=len(args)):
            pass

    # exit with a message
    logger.info('Done point cloud sampling')


def count_facets(mesh_path):
    """
    Count the number of facets given a mesh.
    """
    mesh = trimesh.load(mesh_path)
    faces_extracted = np.concatenate(mesh.facets)
    faces_left = np.setdiff1d(np.arange(len(mesh.faces)), faces_extracted)
    num_facets = len(mesh.facets) + len(faces_left)
    return num_facets


def dict_count_facets(mesh_dir):
    """
    Count the number of facets given a directory of meshes.
    """
    filenames = glob.glob(f'{mesh_dir}/*.obj')
    facet_dict = {}
    for filename_input in filenames:
        stem = Path(filename_input).stem
        num_facets = count_facets(filename_input)
        facet_dict[stem] = num_facets

    # sorted by facet number
    print({k: v for k, v in sorted(facet_dict.items(), key=lambda item: item[1])})


def append_scale_to_csv(input_csv, output_csv):
    """
    Append scale into an existing csv file.
    Note that scale has been implemented in stats.py.
    """
    rows = []
    with open(input_csv, 'r', newline='') as input_csvfile:
        reader = csv.reader(input_csvfile)
        next(reader)  # skip header
        for r in reader:
            filename_input = r[1]
            row = r
            if not filename_input.endswith('.obj'):
                filename_input = filename_input + '.obj'
            mesh = trimesh.load(filename_input)
            extents = mesh.extents
            scale = extents.max()
            row.append(scale)
            rows.append(row)

    with open(output_csv, 'w') as output_csvfile:
        writer = csv.writer(output_csvfile, lineterminator='\n')
        writer.writerows(rows)


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def calculate(cfg):
    """
    Calculate number of parameters.
    """
    from network import PolyGNN

    # initialize model
    model = PolyGNN(cfg)

    # calculate params
    total_params = sum(
        param.numel() for param in model.parameters()
    )

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f'total_params: {total_params}')
    print(f'trainable_params: {trainable_params}')


def merge_vg2cc(args):
    """
    Merge vg from RANSAC and vg from City3D, then generate complexes.
    """
    vg_ransac_path, vg_city3d_path, cc_output_path = args
    epsilon = 0.0001
    vertex_group_ransac = VertexGroup(filepath=vg_ransac_path, refit=True, global_group=False, quiet=True)
    vertex_group_city3d = VertexGroup(filepath=vg_city3d_path, refit=False, global_group=True, quiet=True)
    additional_planes = [p for p in vertex_group_city3d.planes if -epsilon < p[2] < epsilon or
                         (-epsilon < p[0] < epsilon and -epsilon < p[1] < epsilon and 1 - epsilon < p[2] < 1 + epsilon)]

    if len(vertex_group_ransac.planes) == 0:
        return
    cell_complex = CellComplex(vertex_group_ransac.planes, vertex_group_ransac.bounds,
                               vertex_group_ransac.points_grouped,
                               build_graph=True, additional_planes=additional_planes,
                               initial_bound=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], quiet=True)
    cell_complex.refine_planes()  # additional planes are not refined
    cell_complex.prioritise_planes()
    cell_complex.construct()

    cell_complex.save_obj(filepath=Path(cc_output_path).with_suffix('.obj'))
    cell_complex.save(filepath=cc_output_path)


def multi_merge_vg2cc(vg_ransac_dir, vg_city3d_dir, cc_output_dir):
    """
    Merge vertex groups from RANSAC and from City3D, then generate complexes, with multiprocessing.
    """
    args = []
    num_workers = 42
    vg_filenames_ransac = glob.glob(vg_ransac_dir + '/*.vg')

    for vg_filename_ransac in vg_filenames_ransac:
        stem = Path(vg_filename_ransac).stem
        vg_filenames_city3d = (Path(vg_city3d_dir) / stem).with_suffix('.vg')
        if not vg_filenames_city3d.exists():
            continue
        cc_filenames_output = (Path(cc_output_dir) / stem).with_suffix('.cc')
        if cc_filenames_output.exists():
            continue
        args.append([vg_filename_ransac, vg_filenames_city3d, cc_filenames_output])

    with multiprocessing.Pool(processes=num_workers) as pool:
        # call with multiprocessing
        for _ in tqdm(pool.imap_unordered(merge_vg2cc, args), desc='Creating complexes from vertex groups',
                      total=len(args)):
            pass


def vg2cc(args):
    """
    Create cell complex from vertex group.
    """
    vg_path, cc_path = args
    # print(vg_path)

    vertex_group = VertexGroup(filepath=vg_path, refit=False, global_group=True)

    cell_complex = CellComplex(vertex_group.planes, vertex_group.bounds, vertex_group.points_grouped,
                               build_graph=True, additional_planes=None,
                               initial_bound=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    cell_complex.refine_planes(theta=5 * 3.1416 / 180, epsilon=0.002)
    cell_complex.construct()

    cell_complex.save_obj(filepath=Path(cc_path).with_suffix('.obj'))
    cell_complex.save(filepath=cc_path)


def multi_vg2cc(vg_dir, cc_dir):
    """
    Create cell complexes from vertex groups with multiprocessing.
    """
    args = []
    num_workers = 38
    vg_filenames = glob.glob(vg_dir + '/*.vg')

    for vg_filename in vg_filenames:
        stem = Path(vg_filename).stem
        cc_filename = (Path(cc_dir) / stem).with_suffix('.cc')
        args.append([vg_filename, cc_filename])

    with multiprocessing.Pool(processes=num_workers) as pool:
        # call with multiprocessing
        for _ in tqdm(pool.imap_unordered(vg2cc, args), desc='Creating complexes from vertex groups', total=len(args)):
            pass


def coordinate2index(x, reso, coord_type='2d'):
    """ Generate grid index of points

    Args:
        x (tensor): points (normalized to [0, 1])
        reso (int): defined resolution
        coord_type (str): coordinate type
    """
    x = (x * reso).long()
    if coord_type == '2d':  # plane
        index = x[:, :, 0] + reso * x[:, :, 1]  # [B, N, 1]
    index = index[:, None, :]  # [B, 1, N]
    return index


def normalize_coordinate(p, padding=0, plane='xz', scale=1.0):
    """ Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding parameter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
        scale: normalize scale
    """
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane == 'xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
    xy_new = xy_new + 0.5  # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def normalize_3d_coordinate(p, padding=0):
    """ Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """
    p_nor = p / (1 + padding + 10e-4)  # (-0.5, 0.5)
    p_nor = p_nor + 0.5  # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


class map2local(object):
    """
    Add new keys to the given input

    Args:
        s (float): the defined voxel size
        pos_encoding (str): method for the positional encoding, linear|sin_cos
    """

    def __init__(self, s, pos_encoding='linear'):
        super().__init__()
        self.s = s
        self.pe = positional_encoding(basis_function=pos_encoding)

    def __call__(self, p):
        p = torch.remainder(p, self.s) / self.s  # always positive
        # p = torch.fmod(p, self.s) / self.s # same sign as input p!
        p = self.pe(p)
        return p


class positional_encoding(object):
    """ Positional Encoding (presented in NeRF)

    Args:
        basis_function (str): basis function
    """

    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 10
        freq_bands = 2. ** (np.linspace(0, L - 1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = 2.0 * p - 1.0  # chagne to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p


def make_3d_grid(bb_min, bb_max, shape):
    """
    Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    """
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], int(shape[0]))
    pys = torch.linspace(bb_min[1], bb_max[1], int(shape[1]))
    pzs = torch.linspace(bb_min[2], bb_max[2], int(shape[2]))

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


if __name__ == '__main__':
    pass
