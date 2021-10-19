import sys
import glob

ckpts = [
    "checkpoints/resnet18_baseline",
    "checkpoints/resnet18_ours",
]

settings = [
    "--scale 1.0 --n_iter 1",
    "--scale 2.33 --s 0.02 --alpha 0.6 --n_iter 4",
]

exps = ['ori', 'ours']


assert len(sys.argv) == 2, f'Usage: python3 {sys.argv[0]} /path/to/ImageNet100-C/'
for dataset in sorted(glob.glob(f'{sys.argv[1]}/*/?')):
    name = '_'.join(dataset.split('/')[-2:])
    for exp, ckpt, setting in zip(exps, ckpts, settings):
        print(f'python3 main.py {setting} --tmp --model resnet18 --dataset imagenet --num_class 100 --val_dir {dataset} --test --gpu 0,  --checkpoint {ckpt} &> {name}_{exp}.out')
