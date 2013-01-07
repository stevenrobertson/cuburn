#!/usr/bin/env python2
import os
import uuid
import weakref
import numpy as np
import cPickle as pickle

import gevent
from gevent import spawn, queue, coros
import zmq.green as zmq

import _importhack
from cuburn import profile, output
from cuburn.genome import db, util

from messages import *

# TODO: remove this dependency (loading the output module to get the suffix
# requires a compiler / default instance)
import pycuda.autoinit

class RenderClient(object):
    def __init__(self, task_addr, rsp_addr, ctx=None, start=True):
        ctx = zmq.Context() if ctx is None else ctx
        self.tsock = ctx.socket(zmq.REQ)
        self.tsock.connect(task_addr)

        self.cid = uuid.uuid1().hex
        self.rsock = ctx.socket(zmq.DEALER)
        self.rsock.setsockopt(zmq.IDENTITY, self.cid)
        self.rsock.connect(rsp_addr)

        self.tq = queue.Queue(0)

        self.taskmap = weakref.WeakValueDictionary()
        if start: self.start()

    def put(self, task, rq=None):
        """
        Add a task to the render queue. Ths method blocks. Returns the
        queue to which the response will be sent.
        """
        rq = queue.Queue() if rq is None else rq
        self.tq.put((task, rq))
        return rq

    def start(self):
        spawn(self._deal_tasks)
        spawn(self._deal_rsps)

    def _deal_tasks(self):
        for task, rq in self.tq:
            rid = uuid.uuid1().hex
            self.taskmap[rid] = rq
            atask = AddressedTask([self.cid, rid], task)
            self.tsock.send_pyobj(atask)
            # Wait for an (empty) response. This ratelimits tasks.
            self.tsock.recv()

    def _deal_rsps(self):
        while True:
            rsp = self.rsock.recv_multipart(copy=False)
            rq = self.taskmap.get(rsp[0].bytes, None)
            if rq: rq.put((rsp[1].bytes, rsp[2].bytes.split('\0'), rsp[3:]))

# Time (in seconds) before a job times out
# TODO: replace timeout mechanism with polling?
TIMEOUT=4800

# Max. queue length before request considered lost, as a multiple of the
# number of in-flight requests
QUEUE_LENGTH_FACTOR=2

RETRIES=2

def iter_genomes(prof, outpath, gpaths):
    """
    Walk a list of genome paths, yielding them in an order suitable for
    the `genomes` argument of `create_jobs()`.
    """
    gdb = db.connect('.')

    for gpath in gpaths:
        try:
            gnm, basename = gdb.get_anim(gpath)
        except IOError:
            continue
        odir = os.path.join(outpath, basename)
        if (os.path.isfile(os.path.join(odir, 'COMPLETE')) or
            os.path.isfile(os.path.join(outpath, 'ref', basename+'.ts'))):
            continue
        gprof = profile.wrap(prof, gnm)
        ghash = util.hash(gnm)
        times = list(profile.enumerate_times(gprof))
        if not os.path.isdir(odir):
            os.makedirs(odir)
        with open(os.path.join(odir, 'NFRAMES'), 'w') as fp:
            fp.write(str(len(times)))
        outmod = output.get_output_for_profile(gprof)
        for i, t in times:
            opath = os.path.join(odir, '%05d' % i)
            if not os.path.isfile(opath + outmod.suffix):
                yield Task(opath, ghash, prof, gnm, t)

def get_result(cli, task, rq):
    try:
        log, names, bufs = rq.get(timeout=TIMEOUT)
    except queue.Empty:
        cli.put(task, rq)
        print '>>', task.id
        log, names, bufs = rq.get()

    with open(task.id + '.log', 'wb') as fp:
        fp.write(log)

    for name in reversed(names):
        buf = bufs.pop()
        with open(task.id + name, 'wb') as fp:
            fp.write(buffer(buf))
    print '< ', task.id

def main(addrs):
    parser = profile.add_args()
    parser.add_argument('genomes', nargs='+')
    args = parser.parse_args()
    prof_name, prof = profile.get_from_args(args)

    cli = RenderClient(addrs['tasks_loprio'], addrs['responses'])

    gen = iter_genomes(prof, 'out/%s' % prof_name, args.genomes)
    try:
        for task in gen:
            rq = cli.put(task)
            print ' >', task.id
            spawn(get_result, cli, task, rq)
    except KeyboardInterrupt:
        print 'Interrupt received, flushing'

    while cli.taskmap:
        for k, v in cli.taskmap.items():
            if not v.getters:
                cli.taskmap.pop(k)
        print 'Still waiting on %d tasks...' % len(cli.taskmap)
        gevent.sleep(3)

if __name__ == "__main__":
    import addrs
    main(addrs.addrs)
