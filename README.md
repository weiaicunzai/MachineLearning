# MoCoV1 Pytorch Implementation



This repository implenments [MoCo v1](https://arxiv.org/abs/1911.05722) using Pytorch.
This repository is only for learning purpose since I have only implemented the core idea of MoCo v1.


## Dataset
I have used the ``torchvision.datasets.Caltech256`` dataset to train MoCo. 
However, you might encounter ``urllib.error.HTTPError: HTTP Error 404: Not Found`` error when trying to use ``download=True`` to download it.
It is because that the url of ``Caltech256`` dataset has changed, the older ``torchvision`` version (in my case, ``0.8.2``) has trouble downloading the dataset.


## Training command

It at least requires 2 gpus to run, since the Shuffle BN operation introduced in the paper requires at least two GPUs to run.


```
python -m torch.distributed.launch --nproc_per_node=2  train_moco.py

```