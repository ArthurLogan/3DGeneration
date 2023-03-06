# 3D Generation

三维模型生成相关的文章复现。

- [ ] [3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models](https://arxiv.org/abs/2301.11445)

三维模型编码方法。
- [ ] [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828)
- [ ] [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103)
- [ ] [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

## 数据集

Occupancy中预处理好的ShapeNet数据集，需要修改config/train.json中的数据集路径。

```bash
python train.py
```

## 相关库

通过Anaconda3维护环境，需要安装conda环境后，通过如下命令安装环境。

```bash
conda env create -f environment.yml
```
