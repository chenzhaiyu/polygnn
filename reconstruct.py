"""
Conversion from polyhedral labels to tangible mesh file.
"""

import pickle
import glob
import multiprocessing
from pathlib import Path

from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig

from abspy import AdjacencyGraph
from utils import attach_to_log


def reconstruct_from_numpy(args):
    """
    Reconstruct from numpy and cell complex.
    args[0]: numpy_filepath
    args[1]: complex_filepath
    args[2]: mesh_filepath
    args[3]: reconstruction type ('cell' or 'mesh')
    args[4]: seal by excluding boundary cells
    """
    with open(args[1], 'rb') as handle:
        cell_complex = pickle.load(handle)

    pred = np.load(args[0])

    # exclude boundary cells
    if args[4]:
        cells_boundary = cell_complex.cells_boundary()
        pred[cells_boundary] = 0

    indices_cells = np.where(pred)[0]

    if args[3] == 'mesh':
        adjacency_graph = AdjacencyGraph(cell_complex.graph, quiet=True)
        adjacency_graph.reachable = adjacency_graph.to_uids(indices_cells)
        adjacency_graph.non_reachable = np.setdiff1d(adjacency_graph.uid, adjacency_graph.reachable).tolist()
        adjacency_graph.save_surface_obj(args[2], cells=cell_complex.cells, engine='rendering')

    elif args[3] == 'cell':
        if len(indices_cells) > 0:
            cell_complex.save_obj(args[2], indices_cells=indices_cells, use_mtl=True)
    else:
        raise ValueError(f'Unexpected reconstruction type: {args[3]}')


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def multi_reconstruct_from_numpy(cfg: DictConfig):
    """
    Reconstruct from numpy and cell complex with multiprocessing.
    """
    # initialize logging
    logger = attach_to_log()

    # numpy filenames
    filenames_numpy = glob.glob(f'{cfg.output_dir}' + '/*.npy')
    args = []
    for filename_numpy in filenames_numpy:
        stem = Path(filename_numpy).stem
        filename_complex = Path(cfg.complex_dir) / (stem + '.cc')
        filename_output = Path(filename_numpy).with_suffix('.obj')
        if not filename_output.exists():
            args.append((filename_numpy, filename_complex, filename_output, cfg.reconstruct.type, cfg.reconstruct.seal))

    logger.info('Start reconstruction from numpy and cell complex')
    with multiprocessing.Pool(processes=cfg.num_workers if cfg.num_workers else multiprocessing.cpu_count()) as pool:
        # call with multiprocessing
        for _ in tqdm(pool.imap_unordered(reconstruct_from_numpy, args), desc='reconstruction', total=len(args)):
            pass

    # exit with a message
    logger.info('Done reconstruction from numpy and cell complex')


if __name__ == '__main__':
    multi_reconstruct_from_numpy()
