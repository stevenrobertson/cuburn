#!/usr/bin/python2

import os
import sys
import random

if len(sys.argv) > 1:
    random.seed(sys.argv[1])

edges = {}

for i in os.listdir('edges'):
    if '=' not in i: continue
    src, dst = i.rsplit('.', 1)[0].split('=', 1)
    edges.setdefault(src, set()).add((src, dst))

seen = set()

src = random.choice(edges.keys())
for i in range(1000):
    print 'edges/' + src
    dsts = edges.get(src, set()).difference(seen)
    if not dsts:
        src = random.choice(edges.keys())
    else:
        dst = random.choice(list(dsts))[1]
        print 'edges/' + '='.join([src, dst])
        src = dst
