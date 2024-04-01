"""
Supervised training of PolyGNN.
"""

import os
from pathlib import Path

import wandb
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader
from torch_geometric import compile
from torchmetrics.classification import BinaryAccuracy

from network import PolyGNN, focal_loss, bce_loss
from dataset import CityDataset
from utils import init_device, Sampler, set_seed, attach_to_log, setup_runner


def run_train(rank, world_size, dataset_train, dataset_test, cfg):
    """
    Runner function for distributed training of PolyGNN.
    """
    # set up runner
    setup_runner(rank, world_size, cfg.master_addr, cfg.master_port)

    # limit number of threads
    torch.set_num_threads(cfg.num_workers // world_size)

    # initialize logging
    logger = attach_to_log(filepath='./outputs/train.log')
    if rank == 0:
        logger.info(f'Training PolyGNN on {cfg.dataset}')
        wandb_mode = 'online' if cfg.wandb else 'disabled'
        wandb.init(mode=wandb_mode, project='polygnn', entity='zhaiyu',
                   name=cfg.dataset+cfg.run_suffix, dir=cfg.wandb_dir)
        wandb.save('./outputs/.hydra/*')

    # indicate device
    logger.debug(f"Device activated: " + f"CUDA: {cfg.gpu_ids[rank]}")

    # split training indices into `world_size` many chunks
    train_indices = torch.arange(len(dataset_train))
    train_indices = train_indices.split(len(train_indices) // world_size)[rank]
    eval_indices = torch.arange(len(dataset_test))
    eval_indices = eval_indices.split(len(eval_indices) // world_size)[rank]

    # setup dataloaders
    dataloader_train = DataLoader(dataset_train[train_indices], batch_size=cfg.batch_size // world_size,
                                  shuffle=cfg.shuffle, num_workers=cfg.num_workers // world_size,
                                  pin_memory=True, prefetch_factor=8)
    dataloader_test = DataLoader(dataset_test[eval_indices], batch_size=cfg.batch_size // world_size,
                                 shuffle=cfg.shuffle, num_workers=cfg.num_workers // world_size,
                                 pin_memory=True, prefetch_factor=8)

    # initialize model
    model = PolyGNN(cfg).to(rank)

    # distributed parallelization
    model = DistributedDataParallel(model, device_ids=[rank])

    # compile model for better performance
    compile(model, dynamic=True, fullgraph=True)

    # freeze certain layers for fine-tuning
    if cfg.warm:
        for stage in cfg.freeze_stages:
            logger.info(f'Freezing stage: {stage}')
            for parameter in getattr(model, stage).parameters():
                parameter.requires_grad = False

    # initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.scheduler.base_lr, max_lr=cfg.scheduler.max_lr,
                                                  step_size_up=cfg.scheduler.step_size_up, mode=cfg.scheduler.mode,
                                                  cycle_momentum=False)
    # initialize metrics
    metric = BinaryAccuracy().to(rank)

    # warm start from checkpoint if available
    if cfg.warm:
        if rank == 0:
            logger.info(f'Resuming from {cfg.checkpoint_path}')
        map_location = f'cuda:{rank}'
        state = torch.load(cfg.checkpoint_path, map_location=map_location)
        state_dict = state['state_dict']
        model.load_state_dict(state_dict, strict=False)
        if cfg.warm_optimizer:
            try:
                optimizer.load_state_dict(state['optimizer'])
                if rank == 0:
                    logger.info(f'Optimizer loaded from checkpoint')
            except (KeyError, ValueError) as error:
                if rank == 0:
                    logger.warning(f'Optimizer not loaded from checkpoint: {error}')

        if cfg.warm_scheduler:
            try:
                scheduler.load_state_dict(state['scheduler'])
                if rank == 0:
                    logger.info(f'Scheduler loaded from checkpoint')
            except (KeyError, ValueError) as error:
                if rank == 0:
                    logger.warning(f'Scheduler not loaded from checkpoint: {error}')

        best_accuracy = state['accuracy']
        if state['epoch'] > cfg.num_epochs:
            if rank == 0:
                logger.info(f'Expected epoch reached from checkpoint')
            return
        epoch_generator = range(state['epoch'] + 1, cfg.num_epochs)
    else:
        best_accuracy = 0
        epoch_generator = range(cfg.num_epochs)

    # initialize loss function
    if cfg.loss == 'focal':
        loss_func = focal_loss
    elif cfg.loss == 'bce':
        loss_func = bce_loss
    else:
        raise ValueError(f'Unexpected loss function: {cfg.loss}')

    # specify data attributes
    if cfg.sample.strategy == 'grid':
        points_suffix = f'_{cfg.sample.resolution}'
    elif cfg.sample.strategy == 'random':
        points_suffix = f'_{cfg.sample.length}'
    else:
        points_suffix = ''

    # start training
    for i in epoch_generator:
        model.train()
        pbar = tqdm(dataloader_train, desc=f'epoch {i}', disable=rank != 0)

        if rank == 0:
            wandb.log({"epoch": i})

        for batch in pbar:
            optimizer.zero_grad()
            batch = batch.to(rank, f'points{points_suffix}', f'batch_points{points_suffix}', 'queries', 'edge_index',
                             'batch', 'y')
            outs = model(batch)
            targets = batch.y
            loss, accuracy, ratio, _, _ = loss_func(outs, targets)

            if rank == 0:
                wandb.log({"loss": loss})
                wandb.log({"train_accuracy": accuracy})
                wandb.log({"ratio:": ratio})
                wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})
                pbar.set_postfix_str('loss={:.2f}, acc={:.2f}, ratio={:.2f}'.format(loss, accuracy, ratio))

            loss.backward()
            optimizer.step()
            scheduler.step()

        dist.barrier()

        # validate and save checkpoint with DDP
        if cfg.validate and i % cfg.save_interval == 0:
            model.metric = metric
            model = model.to(rank)
            model.eval()

            pbar = tqdm(dataloader_test, desc=f'eval', disable=rank != 0)
            with torch.no_grad():
                for batch in pbar:
                    batch = batch.to(rank, f'points{points_suffix}', f'batch_points{points_suffix}', 'queries',
                                     'edge_index', 'batch', 'y')
                    outs = model(batch)
                    outs = outs.argmax(dim=1)
                    targets = batch.y

                    # metric on current batch
                    accuracy = metric(outs, targets)
                    if rank == 0:
                        pbar.set_postfix_str('acc={:.2f}'.format(accuracy))

                # metrics on all batches and all accelerators using custom accumulation
                accuracy = metric.compute()

                dist.barrier()

                if rank == 0:
                    logger.info(f'Evaluation accuracy: {accuracy:.4f}')
                    wandb.log({"eval_accuracy": accuracy})
                    checkpoint_path = os.path.join(cfg.checkpoint_dir, f'model_epoch{i}.pth')
                    logger.info(f'Saving checkpoint to {checkpoint_path}.')
                    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    state = {
                        'epoch': i,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'accuracy': accuracy,
                    }
                    # Cannot pickle 'WeakMethod' object when saving state_dict for CyclicLr
                    # https://github.com/pytorch/pytorch/pull/91400
                    torch.save(state, checkpoint_path)
                    if accuracy > best_accuracy:
                        logger.info(f'Saving checkpoint to {cfg.checkpoint_path}.')
                        torch.save(state, cfg.checkpoint_path)
                        wandb.save(cfg.checkpoint_path)
                        best_accuracy = accuracy

                # reset internal state such that metric ready for new data
                metric.reset()

        dist.barrier()

    dist.destroy_process_group()


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def train(cfg: DictConfig):
    """
    Train PolyGNN.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration
    """
    logger = attach_to_log(filepath='./outputs/train.log')

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
    dataset_train = CityDataset(pre_transform=pre_transform, transform=transform, root=cfg.data_dir,
                                split='train', num_workers=cfg.num_workers, num_queries=cfg.num_queries)
    dataset_test = CityDataset(pre_transform=pre_transform, transform=transform, root=cfg.data_dir,
                               split='test', num_workers=cfg.num_workers, num_queries=cfg.num_queries)

    world_size = len(cfg.gpu_ids)
    mp.spawn(run_train, args=(world_size, dataset_train, dataset_test, cfg), nprocs=world_size, join=True)


if __name__ == '__main__':
    train()
