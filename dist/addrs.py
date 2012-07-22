ip = '10.1.10.4'
port = 12615
names = 'tasks tasks_loprio workers responses'.split()
addrs = dict((k, 'tcp://%s:%d' % (ip, port+i)) for i, k in enumerate(names))
