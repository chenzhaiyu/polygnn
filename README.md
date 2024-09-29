# PolyGNN

-----------
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://raw.githubusercontent.com/chenzhaiyu/polygnn/main/LICENSE)

PolyGNN is the implementation of the paper [*PolyGNN: Polyhedron-based Graph Neural Network for 3D Building Reconstruction from Point Clouds*](https://arxiv.org/abs/2307.08636). 
> [!NOTE]  
> This repository is under development and may differ from the arXiv manuscript.

<p align="center">
<img src="https://raw.githubusercontent.com/chenzhaiyu/polygnn/master/docs/architecture.png" width="680"/>
</p>

## 🛠️ Setup

### Repository

Clone the repository:

```bash
git clone https://github.com/chenzhaiyu/polygnn && cd polygnn
```

### All-in-one installation

Create a conda environment with all dependencies:

```bash
conda env create -f environment.yml && conda activate polygnn
```

### Manual installation

Alternatively, manual installation is straightforward.

Create a conda environment with [mamba](https://github.com/mamba-org/mamba) for faster parsing:

```bash
conda create --name polygnn python=3.10 && conda activate polygnn
conda install mamba -c conda-forge
```

Install the required dependencies:

```
mamba install pytorch torchvision sage=10.0 pytorch-cuda=11.7 pyg=2.3 pytorch-scatter pytorch-sparse pytorch-cluster torchmetrics rtree -c pyg -c pytorch -c nvidia -c conda-forge
pip install abspy hydra-core hydra-colorlog omegaconf trimesh tqdm wandb plyfile
```

## 🚀 Usage

### Quick start

Download the mini data and pretrained weights:

```python
python download.py dataset=mini
```
In case you encounter issues (e.g., Google Drive download limits), you can manually download the data and weights [here](https://drive.google.com/drive/folders/1fAwvhGtOgS8f4IldE1J4v5s0438WM24b?usp=sharing), then extract them into `./checkpoints/mini` and `./data/mini`, respectively.
The mini data contains 200 random instances (~0.07% of the full dataset).

Train PolyGNN on the mini dataset:

```python
python train.py dataset=mini
```
The first time you launch training, the data will be automatically preprocessed.

Evaluate PolyGNN's performance:

```python
python test.py dataset=mini evaluate.save=true
```

Generate meshes from graph predictions:

```python
python reconstruct.py dataset=mini reconstruct.type=mesh
```

Remap meshes to their original CRS:

```python
python remap.py dataset=mini
```

Derive reconstruction performance statistics:

```python
python stats.py dataset=mini
```

Check available configurations:
```python
# training
python train.py --cfg job

# evaluation
python test.py --cfg job
```
Alternatively, review the configuration file: `conf/config.yaml`.


### Custom data

PolyGNN requires polyhedron-based graphs as input. To prepare this from your own point clouds:
1. Extract planar primitives using tools such as [Easy3D](https://github.com/LiangliangNan/Easy3D) or [GoCoPP](https://github.com/Ylannl/GoCoPP), preferably in [VertexGroup](https://abspy.readthedocs.io/en/latest/vertexgroup.html) format.
2. To train, instantiate your dataset with the [`CityDataset`](https://github.com/chenzhaiyu/polygnn/blob/67addd77a6be1d100448e3bd7523babfa063d0dd/dataset.py#L157) class, or use the [`TestOnlyDataset`](https://github.com/chenzhaiyu/polygnn/blob/67addd77a6be1d100448e3bd7523babfa063d0dd/dataset.py#L276) class for testing-only purposes.

## 👷 TODOs

- [x] Demo with mini data and pretrained weights
- [x] Short tutorial for getting started
- [ ] Host the entire dataset

## 🎓 Citation

If you use PolyGNN in a scientific work, please consider citing the paper:

```bibtex
@article{chen2023polygnn,
  title={PolyGNN: polyhedron-based graph neural network for 3D building reconstruction from point clouds},
  author={Chen, Zhaiyu and Shi, Yilei and Nan, Liangliang and Xiong, Zhitong and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:2307.08636},
  year={2023}
}
```
