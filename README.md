# PolyGNN

-----------

***PolyGNN*** is an implementation of the paper [*PolyGNN: Polyhedron-based Graph Neural Network for 3D Building Reconstruction from Point Clouds*](https://arxiv.org/abs/2307.08636). 
> [!NOTE]  
> This repository is undergoing revisions and may differ from the state of the arXiv manuscript.

<p align="center">
<img src="https://raw.githubusercontent.com/chenzhaiyu/polygnn/master/docs/architecture.jpg" width="680"/>
</p>

## Dependencies

### All-in-one installation

Create a conda environment with all dependencies installed:

```bash
git clone https://github.com/chenzhaiyu/polygnn && cd polygnn
conda env create -f environment.yml && conda activate polygnn
```

### Manual installation

Create a conda environment and enter it:

```bash
conda create --name polygnn python=3.10 && conda activate polygnn
```

Install [mamba](https://github.com/mamba-org/mamba) for faster package parsing and installation:
```bash
conda install mamba -c conda-forge
```

Install the main dependencies:
```
mamba install pytorch torchvision sage=10.0 pytorch-cuda=11.7 pyg=2.3 pytorch-scatter pytorch-sparse pytorch-cluster torchmetrics rtree -c pyg -c pytorch -c nvidia -c conda-forge
pip install abspy hydra-core hydra-colorlog omegaconf trimesh tqdm wandb plyfile
```

## Usage

Download data and weights:
```python
python download.py dataset=munich
```

Train PolyGNN:
```python
python train.py dataset=munich
```

Evaluate PolyGNN:
```python
python test.py dataset=munich evaluate.save=true
```

Reconstruct meshes from predictions:
```python
python reconstruct.py dataset=munich reconstruct.type=mesh
```

Remap meshes to original CRS:
```python
python remap.py dataset=munich
```

Generate statistics:
```python
python stats.py dataset=munich
```

## TODOs

- [ ] Host data and weights
- [ ] Short tutorial on getting started
- [ ] Scripts for data generation and manipulation

## License

[MIT](https://raw.githubusercontent.com/chenzhaiyu/polygnn/main/LICENSE)


## Citation

If you use *PolyGNN* in a scientific work, please consider citing the paper:

```bibtex
@article{chen2023polygnn,
  title={PolyGNN: polyhedron-based graph neural network for 3D building reconstruction from point clouds},
  author={Chen, Zhaiyu and Shi, Yilei and Nan, Liangliang and Xiong, Zhitong and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:2307.08636},
  year={2023}
}
```
