import os
from os.path import join
import sys
import glob
assert len(sys.argv) == 4, f'Usage: python3 {sys.argv[0]} /path/to/imagenet100.txt /path/to/imagenet-c/ /path/to/imagenet100-c'

with open(sys.argv[1]) as f:
    cls = [line.strip() for line in f]
assert len(cls) == 100, cls

imagenet100_c = sys.argv[3]

for ctype_path in glob.glob(f'{sys.argv[2]}/*/?'):
    ctype = '/'.join(ctype_path.split('/')[-2:])
    path = join(imagenet100_c, ctype)
    ret = os.system(f'mkdir -p {path}')
    assert ret == 0
    for c in cls:
        os.system(f"ln -s {join(ctype_path, c)} {join(path, c)}")
