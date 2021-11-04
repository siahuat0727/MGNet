# Multi-Glimpse Network

Multi-Glimpse Network: A Robust and Efficient Classification Architecture based on Recurrent Downsampled Attention

[arXiv](https://arxiv.org/abs/2111.02018)

## Requirements

Python ≥ 3.8

## Installation

For example, venv + pip:
```bash
$ python3 -m venv env
$ source env/bin/activate
(env) $ python3 -m pip install -r requirements.txt
```

## Evaluation

### Accuracy on clean images

1. Create ImageNet100 from ImageNet (using symbolic links).

```bash
$ python3 tools/create_imagenet100.py tools/imagenet100.txt \
    /path/to/ImageNet /path/to/ImageNet100
```

2. Download checkpoints from [Google Drive](https://drive.google.com/drive/folders/1CeRILtCRKkJsNbxsUhO07Ni5b-gIDD7f?usp=sharing).

3. Test accuracy.

```bash
$ export dataset="--train_dir /path/to/ImageNet100/train \
    --val_dir /path/to/ImageNet100/val \
    --dataset imagenet --num_class 100"
# Baseline
$ python3 main.py $dataset --test --n_iter 1 --scale 1.0  --model resnet18 \
    --checkpoint resnet18_baseline
# Ours
$ python3 main.py $dataset --test --n_iter 4 --scale 2.33 --model resnet18 \
    --checkpoint resnet18_ours --alpha 0.6 --s 0.02
```

Add the flag `--flop_count` to count the approximate FLOPs for the inference of an image. (using [fvcore](https://github.com/facebookresearch/fvcore))

<div style="page-break-after: always;"></div>

### Accuracy on adversarial attacks (PGD)

1. Test adversarial accuracy.

```bash
# Baseline
$ python3 main.py $dataset --test --n_iter 1 --scale 1.0  --adv --step_k 10 \
    --model resnet18 --checkpoint resnet18_baseline
# Ours
$ python3 main.py $dataset --test --n_iter 4 --scale 2.33 --adv --step_k 10 \
    --model resnet18 --checkpoint resnet18_ours --alpha 0.6 --s 0.02
```

### Accuracy on common corruptions

1. Create ImageNet100-C from ImageNet-C (using symbolic links).

```bash
$ python3 tools/create_imagenet100c.py  \
    tools/imagenet100.txt  /path/to/ImageNet-C/ /path/to/ImageNet100-C/
```

2. Test for a single corruption.

```bash
$ export dataset="--train_dir /path/to/ImageNet100/train \
    --val_dir /path/to/ImageNet100-C/pixelate/5 \
    --dataset imagenet --num_class 100"
# Baseline
$ python3 main.py $dataset --test --n_iter 1 --scale 1.0  --model resnet18 \
    --checkpoint resnet18_baseline
# Ours
$ python3 main.py $dataset --test --n_iter 4 --scale 2.33 --model resnet18 \
    --checkpoint resnet18_ours --alpha 0.6 --s 0.02
```

3. A simple script to test all corruptions and collect results.

```bash
# Modify tools/eval_imagenet100c.py and run it to generate script
$ python3 tools/eval_imagenet100c.py /home2/ImageNet100-C/ > run.sh
# Evaluate
$ bash run.sh
# Collect results
$ python3 tools/collect_imagenet100c.py
```


<div style="page-break-after: always;"></div>

## Training


```bash
$ export dataset="--train_dir /path/to/ImageNet100/train \
    --val_dir /path/to/ImageNet100/val \
    --dataset imagenet --num_class 100"
# Baseline
$ python3 main.py $dataset --epochs 400 --n_iter 1 --scale 1.0 \
    --model resnet18 --gpu 0,1,2,3
# Ours
$ python3 main.py $dataset --epochs 400 --n_iter 4 --scale 2.33 \
    --model resnet18 --alpha 0.6 --s 0.02  --gpu 0,1,2,3
```

Check tensorboard for the logs. (When training with multiple gpus, the log value may be scaled by the number of gpus except for the validation accuracy)

```bash
tensorboard  --logdir=logs
```



Note that we left our exploration in the code for further study, e.g., self-supervised spatial guidance, dynamic gradient re-scaling operation.
