"""
Definitions of polyhedral graph data structure and datasets.
"""

import os
import logging
import glob
import collections
import multiprocessing
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from abspy import VertexGroup, VertexGroupReference, CellComplex

from utils import edge_index_from_dict, index_to_mask

logger = logging.getLogger('dataset')


class PolyGraph:
    """
    Cell-based graph data structure.
    """

    def __init__(self, use_reference=False, num_queries=None):
        self.vertex_group = None
        self.cell_complex = None
        self.vertex_group_reference = None
        self.use_reference = use_reference
        self.num_queries = num_queries

    def cell_adjacency(self):
        """
        Create adjacency among cells.
        """
        # mapping gaped adjacency indices to contiguous ones
        adj = self.cell_complex.graph.adj
        uid = list(self.cell_complex.graph.nodes)
        mapping = {c: i for i, c in enumerate(uid)}
        adj_ = collections.defaultdict(set)
        for key in adj:
            adj_[mapping[key]] = {mapping[value] for value in adj[key]}

        # graph edge index in COO format
        return edge_index_from_dict(adj_)

    def cell_labels(self, mesh_path):
        """
        Labels of cells, one-hot encoding.
        """
        labels = np.zeros(self.cell_complex.num_cells).astype(np.int64)

        # cells inside reference mesh
        cells_in_mesh = self.cell_complex.cells_in_mesh(mesh_path, engine='distance')

        for cell in cells_in_mesh:
            labels[cell] = 1

        return torch.tensor(labels)

    def data_loader(self, cloud_path, mesh_path=None, complex_path=None, vertex_group_path=None):
        """
        Load bvg file and obj file in network readable format.
        """
        if complex_path is not None and os.path.exists(complex_path):
            # load existing cell complex
            import pickle
            with open(complex_path, 'rb') as handle:
                self.cell_complex = pickle.load(handle)
        else:
            # construct cell complex
            if not self.use_reference:
                # load point cloud as vertex group
                if vertex_group_path:
                    self.vertex_group = VertexGroup(vertex_group_path, quiet=True)
                    # initialise cell complex from planar primitives
                    self.cell_complex = CellComplex(self.vertex_group.planes, self.vertex_group.aabbs,
                                                    self.vertex_group.points_grouped, build_graph=True, quiet=True)
                else:
                    # cannot process vertex group from points alone
                    raise NotImplementedError
            else:
                # load mesh as vertex group reference
                self.vertex_group_reference = VertexGroupReference(mesh_path, quiet=True)
                # initialise cell complex from planar primitives
                self.cell_complex = CellComplex(np.array(self.vertex_group_reference.planes),
                                                np.array(self.vertex_group_reference.aabbs),
                                                build_graph=True, quiet=True)

            # prioritise certain planes (e.g., vertical ones)
            self.cell_complex.prioritise_planes(prioritise_verticals=True)

            try:
                # construct cell complex
                self.cell_complex.construct()
            except (AssertionError, IndexError) as e:
                logger.error(f'Error [{e}] occurred with {cloud_path}.')
                return

            # save cell complex to CC files
            if complex_path is not None:
                Path(complex_path).parent.mkdir(exist_ok=True)
                self.cell_complex.save(complex_path)

        # points
        if cloud_path is not None:
            # npy and vg may contain different point sets
            points = np.load(cloud_path)
        else:
            points = self.vertex_group.points

        # queries
        queries = np.array(self.cell_complex.cell_representatives(location='skeleton', num=self.num_queries))

        # cell adjacency
        adjacency = self.cell_adjacency()

        # cell ground truth labels
        if mesh_path:
            labels = self.cell_labels(mesh_path)
        else:
            labels = None

        # construct data for pytorch geometric
        data = Data(x=None, edge_index=adjacency, y=labels)

        # store sizes
        len_cells = queries.shape[0]
        len_points = len(points)
        data.num_nodes = len_cells
        data.num_points = len_points

        # store points and queries
        data.points = torch.as_tensor(points, dtype=torch.float)
        data.queries = torch.as_tensor(queries, dtype=torch.float)

        # batch indices of points
        data.batch_points = torch.zeros(len_points, dtype=torch.long)

        # specify masks
        data.train_mask = index_to_mask(range(len_cells), size=len_cells)
        data.val_mask = index_to_mask(range(len_cells), size=len_cells)
        data.test_mask = index_to_mask(range(len_cells), size=len_cells)

        # name for reference
        data.name = Path(cloud_path).stem

        # validate data
        data.validate(raise_on_error=True)

        return data


class CityDataset(Dataset):
    """
    Base building dataset. Applies to Munich and Nuremberg.
    """

    def __init__(self, root, name=None, split=None, num_workers=1, num_queries=16, **kwargs):
        self.name = name
        self.split = split
        self.num_workers = num_workers
        self.cloud_suffix = '.npy'
        self.mesh_suffix = '.obj'
        self.complex_suffix = '.cc'
        self.num_queries = num_queries
        super().__init__(root, **kwargs)  # this line calls download() and process()

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        with open(os.path.join(self.raw_dir, f'{self.split}set.txt'), 'r') as f:
            return f.read().splitlines()

    @property
    def processed_file_names(self):
        return [f'data_{self.split}_{i}.pt' for i in range(len(self.raw_file_names))]

    def download(self):
        pass

    def thread(self, kwargs):
        """
        Process one file.
        """
        path_save = os.path.join(self.processed_dir, f'data_{kwargs["split"]}_{kwargs["index"]}.pt')
        if os.path.exists(path_save):
            return
        logger.info(f'processing {Path(kwargs["cloud"]).stem}')
        try:
            data = PolyGraph(use_reference=True, num_queries=self.num_queries).data_loader(kwargs['cloud'],
                                                                                           kwargs['mesh'],
                                                                                           kwargs['complex'])
        except (ValueError, IndexError, EOFError) as e:
            logger.error(f'error with file {kwargs["mesh"]}: {e}')
            return
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        if data is not None:
            torch.save(data, path_save)

    def process(self):
        """
        Start multiprocessing.
        """
        with open(os.path.join(self.raw_dir, 'trainset.txt'), 'r') as f_train:
            filenames_train = f_train.read().splitlines()
        with open(os.path.join(self.raw_dir, 'testset.txt'), 'r') as f_test:
            filenames_test = f_test.read().splitlines()

        args = []
        for i, filename_train in enumerate(filenames_train):
            cloud_train = os.path.join(self.raw_dir, '04_pts', filename_train + self.cloud_suffix)
            mesh_train = os.path.join(self.raw_dir, '03_meshes', filename_train + self.mesh_suffix)
            complex_train = os.path.join(self.raw_dir, '05_complexes', filename_train + self.complex_suffix)
            args.append(
                {'index': i, 'split': 'train', 'cloud': cloud_train, 'mesh': mesh_train, 'complex': complex_train})

        for j, filename_test in enumerate(filenames_test):
            cloud_test = os.path.join(self.raw_dir, '04_pts', filename_test + self.cloud_suffix)
            mesh_test = os.path.join(self.raw_dir, '03_meshes', filename_test + self.mesh_suffix)
            complex_test = os.path.join(self.raw_dir, '05_complexes', filename_test + self.complex_suffix)
            args.append(
                {'index': j, 'split': 'test', 'cloud': cloud_test, 'mesh': mesh_test, 'complex': complex_test})

        with multiprocessing.Pool(
                processes=self.num_workers if self.num_workers else multiprocessing.cpu_count()) as pool:
            # call with multiprocessing
            for _ in tqdm(pool.imap(self.thread, args), desc='Preparing dataset', total=len(args)):
                pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{self.split}_{idx}.pt'))
        # to disable UserWarning: Unable to accurately infer 'num_nodes' from the attribute set
        # '{'points', 'train_mask', 'val_mask', 'queries', 'test_mask', 'edge_index', 'y'}'
        data.num_nodes = len(data.y)
        return data


class HelsinkiDataset(CityDataset):
    """
    Helsinki dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def processed_file_names(self):
        """
        Modified processed filenames due to discontinuity.
        """
        return [os.path.basename(filename) for filename in
                glob.glob(os.path.join(self.processed_dir, f'data_{self.split}_*.pt'))]

    def get(self, idx):
        """
        Modified data retrieval due to discontinuity.
        """
        data = torch.load(self.processed_paths[idx])
        # to disable UserWarning: Unable to accurately infer 'num_nodes' from the attribute set
        # '{'points', 'train_mask', 'val_mask', 'queries', 'test_mask', 'edge_index', 'y'}'
        data.num_nodes = len(data.y)
        return data


class TestOnlyDataset(CityDataset):
    """
    Test-only dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self):
        """
        Start multiprocessing.
        """
        with open(os.path.join(self.raw_dir, 'testset.txt'), 'r') as f_test:
            filenames_test = f_test.read().splitlines()

        args = []

        for j, filename_test in enumerate(filenames_test):
            cloud_test = os.path.join(self.raw_dir, '04_pts', filename_test + self.cloud_suffix)
            mesh_test = os.path.join(self.raw_dir, '03_meshes', filename_test + self.mesh_suffix)
            complex_test = os.path.join(self.raw_dir, '05_complexes', filename_test + self.complex_suffix)
            args.append(
                {'index': j, 'split': 'test', 'cloud': cloud_test, 'mesh': mesh_test, 'complex': complex_test})

        with multiprocessing.Pool(
                processes=self.num_workers if self.num_workers else multiprocessing.cpu_count()) as pool:
            # call with multiprocessing
            for _ in tqdm(pool.imap(self.thread, args), desc='Preparing dataset', total=len(args)):
                pass
