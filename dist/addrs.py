ip = '127.0.0.1'
port = 12615
names = 'tasks tasks_loprio workers responses'.split()
addrs = dict((k, 'tcp://%s:%d' % (ip, port+i)) for i, k in enumerate(names))
