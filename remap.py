"""
Remap normalized instances to global CRS.
"""

import glob
from pathlib import Path
import multiprocessing
import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from utils import reverse_normalise_mesh, reverse_normalise_cloud, normalise_mesh


logger = logging.getLogger("trimesh")
logger.setLevel(logging.WARNING)


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def normalize_meshes(cfg: DictConfig):
    """
    Normalize meshes.

    cfg: DictConfig
        Hydra configuration
    """
    args = []
    input_filenames = glob.glob(f'{cfg.output_dir}/*.obj')
    output_dir = Path(cfg.output_dir) / 'normalized'
    output_dir.mkdir(exist_ok=True)
    for input_filename in input_filenames:
        base_filename = Path(input_filename).name
        reference_filename = (Path(cfg.reference_dir) / base_filename).with_suffix('.obj')
        output_filename = output_dir / base_filename
        args.append((input_filename, reference_filename, output_filename, 'scene', cfg.reconstruct.offset, True, False))
    print('start processing')
    with multiprocessing.Pool(processes=cfg.num_workers if cfg.num_workers else multiprocessing.cpu_count()) as pool:
        # call with multiprocessing
        for _ in tqdm(pool.imap_unordered(normalise_mesh, args), desc='Normalizing meshes', total=len(args)):
            pass


# normalize clouds as meshes
normalize_clouds = normalize_meshes


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def remap_meshes(cfg: DictConfig):
    """
    Remap normalized buildings to global CRS.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration
    """
    args = []
    input_filenames = glob.glob(cfg.output_dir + '/*.obj')
    output_dir = Path(cfg.output_dir) / 'global'
    output_dir.mkdir(exist_ok=True)
    for input_filename in input_filenames:
        base_filename = Path(input_filename).name
        reference_filename = Path(cfg.reference_dir) / base_filename
        output_filename = output_dir / base_filename
        args.append((input_filename, reference_filename, output_filename, 'scene', cfg.reconstruct.offset, cfg.reconstruct.scale, cfg.reconstruct.translate))
    with multiprocessing.Pool(processes=cfg.num_workers if cfg.num_workers else multiprocessing.cpu_count()) as pool:
        # call with multiprocessing
        for _ in tqdm(pool.imap_unordered(reverse_normalise_mesh, args), desc='Remapping meshes', total=len(args)):
            pass


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def remap_clouds(cfg: DictConfig):
    """
    Remap normalized point clouds to global CRS.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration
    """
    args = []
    input_filenames = glob.glob(f'{cfg.data_dir}/raw/test_cloud_normalised_ply/*.ply')
    output_dir = Path(cfg.output_dir) / 'global_clouds'
    output_dir.mkdir(exist_ok=True)
    for input_filename in input_filenames:
        base_filename = Path(input_filename).name
        reference_filename = (Path(cfg.reference_dir) / base_filename).with_suffix('.obj')
        output_filename = output_dir / base_filename
        args.append((input_filename, reference_filename, output_filename, 'scene', cfg.reconstruct.offset))
    print('start processing')
    with multiprocessing.Pool(processes=cfg.num_workers if cfg.num_workers else multiprocessing.cpu_count()) as pool:
        # call with multiprocessing
        for _ in tqdm(pool.imap_unordered(reverse_normalise_cloud, args), desc='Remapping clouds', total=len(args)):
            pass


if __name__ == '__main__':
    remap_meshes()
