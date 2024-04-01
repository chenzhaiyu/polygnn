"""
Evaluation of PolyGNN.
"""

import pickle
from pathlib import Path
import multiprocessing

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
from torchmetrics.classification import BinaryAccuracy
import torch.distributed as dist
from torch_geometric import compile

from network import PolyGNN
from dataset import CityDataset, TestOnlyDataset
from utils import init_device, Sampler, set_seed, attach_to_log, setup_runner


class PredictionSaver:
    """
    Asynchronous prediction saving.
    """
    def __init__(self, processes):
        self.pool = multiprocessing.Pool(processes=processes)

    @staticmethod
    def save(args):
        pred, name, cfg = args
        indices_cells = np.where(pred.cpu().numpy())[0]
        if len(indices_cells) > 0:
            complex_path = f'{cfg.complex_dir}/{name}.cc'
            with open(complex_path, 'rb') as handle:
                cell_complex = pickle.load(handle)
            output_path = f'{cfg.output_dir}/{name}.npy'
            output = np.zeros([cell_complex.num_cells], dtype=int)
            output[indices_cells] = 1

            if cfg.evaluate.seal:
                cells_boundary = cell_complex.cells_boundary()
                output[cells_boundary] = 0
            np.save(output_path, output)


def run_eval(rank, world_size, dataset_test, cfg):
    """
    Runner function for distributed inference of PolyGNN.
    """
    # set up runner
    setup_runner(rank, world_size, cfg.master_addr, cfg.master_port)

    # limit number of threads
    torch.set_num_threads(cfg.num_workers // world_size)

    # initialize logging
    logger = attach_to_log(filepath='./outputs/test.log')

    # indicate device
    logger.debug(f"Device activated: " + f"CUDA: {cfg.gpu_ids[rank]}")

    # initialize metric
    metric = BinaryAccuracy()

    # split test indices into `world_size` many chunks
    eval_indices = torch.arange(len(dataset_test))
    eval_indices = eval_indices.split(len(eval_indices) // world_size)[rank]
    dataloader_test = DataLoader(dataset_test[eval_indices], batch_size=cfg.batch_size // world_size,
                                 shuffle=cfg.shuffle, num_workers=cfg.num_workers // world_size,
                                 pin_memory=True, prefetch_factor=8)

    # initialize model
    model = PolyGNN(cfg)
    model.metric = metric
    model = model.to(rank)

    # distributed parallelization
    model = DistributedDataParallel(model, device_ids=[rank])

    # compile model for better performance
    compile(model, dynamic=True, fullgraph=True)

    # load from checkpoint
    map_location = f'cuda:{rank}'
    if rank == 0:
        logger.info(f'Resuming from {cfg.checkpoint_path}')
    state = torch.load(cfg.checkpoint_path, map_location=map_location)
    state_dict = state['state_dict']
    model.load_state_dict(state_dict, strict=False)

    # specify data attributes
    if cfg.sample.strategy == 'grid':
        points_suffix = f'_{cfg.sample.resolution}'
    elif cfg.sample.strategy == 'random':
        points_suffix = f'_{cfg.sample.length}'
    else:
        points_suffix = ''

    # start inference
    model.eval()
    pbar = tqdm(dataloader_test, desc=f'eval', disable=rank != 0)

    # initialize PredictionSaver instance
    prediction_saver = PredictionSaver(processes=cfg.num_workers // world_size)

    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(rank, f'points{points_suffix}', f'batch_points{points_suffix}', 'queries',
                             'edge_index', 'batch', 'y')
            outs = model(batch)
            outs = outs.argmax(dim=1)
            targets = batch.y

            # metric on current batch
            _accuracy = metric(outs, targets)
            pbar.set_postfix_str('acc={:.2f}'.format(_accuracy))

            # save prediction as numpy file
            if cfg.evaluate.save:
                Path(cfg.output_dir).mkdir(exist_ok=True)
                _, boundary_indices = torch.unique(batch.batch, return_counts=True)

                preds = torch.split(outs, split_size_or_sections=boundary_indices.tolist(), dim=0)
                names = batch.name

                # asynchronous file saving
                prediction_saver.pool.map(prediction_saver.save, zip(preds, names, [cfg] * len(preds)))

        # metric on all batches and all accelerators using custom accumulation
        accuracy = metric.compute()

        if rank == 0:
            logger.info(f"Evaluation accuracy: {accuracy}")

        # reset internal state such that metric ready for new data
        metric.reset()

    dist.barrier()

    dist.destroy_process_group()


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def test(cfg: DictConfig):
    """
    Test PolyGNN for reconstruction.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration
    """

    # initialize logger
    logger = attach_to_log(filepath='./outputs/test.log')

    # initialize device
    init_device(cfg.gpu_ids, register_freeze=cfg.gpu_freeze)
    logger.info(f"Device initialized: " + f"CUDA: {cfg.gpu_ids}")

    # fix randomness
    set_seed(cfg.seed)
    logger.info(f"Random seed set to {cfg.seed}")

    # initialize data sampler
    sampler = Sampler(strategy=cfg.sample.strategy, length=cfg.sample.length, ratio=cfg.sample.ratio,
                      resolutions=cfg.sample.resolutions, duplicate=cfg.sample.duplicate, seed=cfg.seed)
    transform = sampler.sample if cfg.sample.transform else None
    pre_transform = sampler.sample if cfg.sample.pre_transform else None

    # initialize dataset
    if cfg.dataset in {'munich', 'munich_perturb', 'munich_subsample', 'munich_truncate', 'munich_haswall', 'campus_ldbv'}:
        dataset = TestOnlyDataset(pre_transform=pre_transform, transform=transform, root=cfg.data_dir,
                                  split='test', num_workers=cfg.num_workers)
    else:
        dataset = CityDataset(pre_transform=pre_transform, transform=transform, root=cfg.data_dir,
                              split='test', num_workers=cfg.num_workers)

    world_size = len(cfg.gpu_ids)
    mp.spawn(run_eval, args=(world_size, dataset, cfg), nprocs=world_size, join=True)


if __name__ == '__main__':
    test()
