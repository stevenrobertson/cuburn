import os, sys

print 'digraph {'
for i in os.listdir(sys.argv[1] if len(sys.argv) > 1 else '.'):
    parts = i.rsplit('.', 1)[0].split('=')
    print ' -> '.join(parts[:2])
    # TODO: add label (optional section 3)
print '}'
