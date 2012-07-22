#!/usr/bin/env python2
import sys
import socket
from cStringIO import StringIO

import gevent
from gevent import spawn, queue
import zmq.green as zmq
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

def main(worker_addr):
    rmgr = render.RenderManager()

    ctx = zmq.Context()
    in_queue = queue.Queue(0)
    out_queue = queue.Queue(0)

    def request_loop():
        sock = ctx.socket(zmq.REQ)
        sock.connect(worker_addr)

        # Start the request loop with an empty job
        sock.send('')

        hash = None
        while True:
            log = [('worker', socket.gethostname() + ':' +
                    cuda.Context.get_current().get_device().pci_bus_id())]
            addr, task, cubin, packer = sock.recv_pyobj()
            gprof = profile.wrap(task.profile, task.anim)
            if hash != task.hash:
                rdr = PrecompiledRenderer(task.anim, gprof, packer, cubin)
            for t in task.times:
                evt, buf = rmgr.queue_frame(rdr, task.anim, gprof, t)
                while not evt.query():
                    gevent.sleep(0.01)
                out, frame_log = rdr.out.encode(buf)
                log += frame_log
                print 'Rendered', task.id, 'in', int(evt.time()), 'ms'
            final_out, final_log = rdr.out.encode(None)
            assert not (out and final_out), 'Got output from two sources!'
            out = out or final_out
            log += final_log
            log = '\0'.join([k + ' ' + v for k, v in log])

            suffixes, files = zip(*[(k, v.read())
                                    for k, v in sorted(out.items())])
            # TODO: reduce copies, generally spruce up the memory usage here
            sock.send_multipart(addr + [log, '\0'.join(suffixes)] + list(files))

    # Spawn two request loops to take advantage of CUDA pipelining.
    spawn(request_loop)
    request_loop()

if __name__ == "__main__":
    import addrs
    dev = cuda.Device(int(sys.argv[1]))
    cuctx = dev.make_context(cuda.ctx_flags.SCHED_BLOCKING_SYNC)
    try:
        main(addrs.addrs['workers'])
    finally:
        cuda.Context.pop()
