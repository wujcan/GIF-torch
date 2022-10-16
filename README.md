# GIF-Torch
This is the PyTorch implementation for our Graph-oriented Influence Function.


## Environment Requirement
The code runs well under python 3.6.10. The required packages are as follows:

- pytorch == 1.9.0+cu111
- torch-geometric == 2.0.3
- torchvision == 0.10.0+cu111

## Quick Start

For regular unlearning task (e.g., Edge Unlearning with ratio 0.05 on Cora using GCN)
```bash
python main.py --dataset_name cora --target_model GCN --is_train_target_model True --exp Unlearn --method GIF --is_use_node_feature True --num_runs 10 --unlearn_task edge --unlearn_ratio 0.05 --iteration 100 --scale 500
```

For adversarial attack experiments (e.g., ratio=0.5 on Cora using GCN)
```bash
python main.py --dataset_name cora --target_model GCN --is_train_target_model True --exp Attack --method GIF --is_use_node_feature True --num_runs 10 --unlearn_task edge --unlearn_ratio 0.5 --iteration 100 --scale 500
```