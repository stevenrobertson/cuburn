import os, sys
from os.path import isdir, join

dir = sys.argv[1] if len(sys.argv) > 1 else '.'

print 'digraph {'
for i in os.listdir(dir):
    if not isdir(join(dir, i)):
        i = i.rsplit('.', 1)[0]
    parts = i.split('=')
    print ' -> '.join(parts[:2])
    # TODO: add label (optional section 3)
print '}'
