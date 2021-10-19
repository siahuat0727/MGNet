import os
import sys
assert len(sys.argv) == 4, f'Usage: python3 {sys.argv[0]} /path/to/imagenet100.txt /path/to/imagenet /path/to/imagenet100'

with open(sys.argv[1]) as f:
    cls = [line.strip() for line in f]
assert len(cls) == 100, cls

imagenet = sys.argv[2]
imagenet100 = sys.argv[3]

os.system(f'mkdir {imagenet100}')
os.system(f'mkdir {imagenet100}/train')
os.system(f'mkdir {imagenet100}/val')
for c in cls:
    os.system(f'ln -s {imagenet}/train/{c} {imagenet100}/train/{c}')
    os.system(f'ln -s {imagenet}/val/{c} {imagenet100}/val/{c}')
