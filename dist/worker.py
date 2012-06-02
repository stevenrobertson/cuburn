#!/usr/bin/env python2
import sys
from cStringIO import StringIO

import zmq
import pycuda.driver as cuda
cuda.init()

import _importhack
from cuburn import render, profile, output
from cuburn.genome import convert, db, use

from messages import *

class PrecompiledRenderer(render.Renderer):
    def compile(self, gnm):
        return self.packer, None, self.cubin
    def __init__(self, gnm, gprof, packer, cubin):
        self.packer, self.cubin = packer, cubin
        super(PrecompiledRenderer, self).__init__(gnm, gprof)

def main(card_num, worker_addr):
    dev = cuda.Device(card_num)
    ctx = dev.make_context(cuda.ctx_flags.SCHED_BLOCKING_SYNC)
    rmgr = render.RenderManager()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(worker_addr)

    # Start the request loop with an empty job
    sock.send('')
    hash = None
    while True:
        addr, task, cubin, packer = sock.recv_pyobj()
        gprof = profile.wrap(task.profile, task.anim)
        if hash != task.hash:
            rdr = PrecompiledRenderer(task.anim, gprof, packer, cubin)
        buf = rmgr.queue_frame(rdr, task.anim, gprof, task.time)
        ofile = StringIO()
        output.PILOutput.save(buf, ofile, task.id[-3:])
        ofile.seek(0)
        sock.send_multipart(addr + [ofile.read()])
        hash = task.hash
        print 'Rendered', task.id

if __name__ == "__main__":
    import addrs
    main(int(sys.argv[1]), addrs.addrs['workers'])
