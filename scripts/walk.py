import os
import random

edges = {}

for i in os.listdir('edges'):
    if '=' not in i: continue
    src, dst = i.rsplit('.', 1)[0].split('=', 1)
    edges.setdefault(src, set()).add((src, dst))

seen = set()

src = random.choice(edges.keys())
for i in range(1000):
    print src
    dsts = edges.get(src, set()).difference(seen)
    if not dsts:
        src = random.choice(edges.keys())
    else:
        dst = random.choice(list(dsts))[1]
        print '='.join([src, dst])
        src = dst
