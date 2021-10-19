import re
import glob
from collections import defaultdict
from tabulate import tabulate

def mean(a):
    return sum(a) / len(a)

def avg_list_of_list(l):
    return list(map(mean, zip(*l)))


table = defaultdict(list)

for path in sorted(glob.glob('*.out')):
    title = path[:path.rfind('_')]
    with open(path) as f:
        s = f.read()
    res = re.findall(r'val_[1234]\':\ tensor\(\d+\.\d+', s)
    res = [float(r[15:]) for r in res]
    table[title].extend(res)


table2 = defaultdict(list)
for k in table.keys():
    table2[k.split('_')[0]].append(table[k])

for k in table2.keys():
    table2[k] = avg_list_of_list(table2[k])

table2 = {
    t: table2[t]
    for t in ['gaussian' ,'shot' ,'impulse' ,'defocus' ,'glass' ,'motion' ,'zoom' ,'snow' ,'frost' ,'fog' ,'brightness' ,'contrast' ,'elastic' ,'pixelate' ,'jpeg']
}

table2['Average'] = [
    sum(vs)/len(vs)
    for vs in zip(*table2.values())
]

for i in range(len(table2['Average'])):
    print('\t'.join(
        f'{100*table2[t][i]:.2f}'
        for t in ['Average', 'gaussian' ,'shot' ,'impulse' ,'defocus' ,'glass' ,'motion' ,'zoom' ,'snow' ,'frost' ,'fog' ,'brightness' ,'contrast' ,'elastic' ,'pixelate' ,'jpeg']
    ))

for k in table2.keys():
    max_ = max(table2[k])
    table2[k] = [
            f'**{v:.4f}**' if v == max_ else f'{v:.4f}'
        for v in table2[k]
    ]

table2 = [
    [k, *v]
    for k, v in table2.items()
]

header = ['Corrupt type', 'Baseline'] + [f'Ours_{i}' for i in range(1,5)]
print(tabulate(table2, headers=header, tablefmt="github"))
