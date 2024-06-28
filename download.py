"""
Download datasets and/or models from public urls.
"""

import os
import tarfile
import urllib.request

from tqdm import tqdm
import hydra
from omegaconf import DictConfig


def my_hook(t):
    """
    Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py

    Example
    -------
        with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def download(cfg: DictConfig):
    """
    Download datasets and/or models from public urls.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration
    """
    data_url, checkpoint_url = cfg.url_dataset['default'], cfg.url_checkpoint['default']
    data_dir, checkpoint_dir = cfg.data_dir, cfg.checkpoint_dir
    data_file, checkpoint_file = (os.path.join(data_dir, f'{cfg.dataset}.tar.gz'), f'{cfg.checkpoint_path}')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if data_url is not None:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=data_file) as t:
            urllib.request.urlretrieve(data_url, filename=data_file, reporthook=my_hook(t), data=None)
        with tarfile.open(data_file, 'r:gz') as tar:
            tar.extractall(data_dir)
        os.remove(data_file)

    if checkpoint_url is not None:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=checkpoint_file) as t:
            urllib.request.urlretrieve(checkpoint_url, filename=checkpoint_file, reporthook=my_hook(t), data=None)


if __name__ == '__main__':
    download()
