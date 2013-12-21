#!/usr/bin/env python2
from itertools import takewhile

import gevent
from gevent import spawn, queue, event
import zmq.green as zmq
import cPickle as pickle

import _importhack
from cuburn.render import Renderer

from messages import *

ctx = zmq.Context()

def setup_task_listeners(addrs, tq, rq):
    hisock = ctx.socket(zmq.REP)
    losock = ctx.socket(zmq.REP)
    hisock.bind(addrs['tasks'])
    losock.bind(addrs['tasks_loprio'])

    loevt = event.Event()
    loevt.set()

    @spawn
    def listen_hi():
        while True:
            if not hisock.poll(timeout=0):
                # No messages pending. Set loevt, allowing messages from
                # losock to be added to the queue.
                loevt.set()
            task = hisock.recv_pyobj()
            loevt.clear() # Got message; pause listen_lo().
            tq.put(task)
            hisock.send('')

    @spawn
    def listen_lo():
        while True:
            loevt.wait()
            tq.put(losock.recv_pyobj())
            losock.send('')

def setup_worker_listener(addrs, tq, rq):
    wsock = ctx.socket(zmq.ROUTER)
    wsock.bind(addrs['workers'])

    readyq = queue.Queue()

    compcache = {}

    @spawn
    def send_work():
        for addr, task in tq:
            print ' >', ' '.join(addr)
            if task.hash not in compcache:
                try:
                    rsp = Renderer.compile(task.anim, arch='sm_35')
                except:
                    # Store exceptions, so that we don't try to endlessly
                    # recompile bad genomes
                    import traceback
                    rsp = traceback.format_exc()
                    print 'Error while compiling task:', rsp
                compcache[task.hash] = rsp
            else:
                rsp = compcache[task.hash]
            if isinstance(rsp, basestring):
                continue
            packer, lib, cubin = rsp
            ctask = FullTask(addr, task, cubin, packer)
            worker_addr = readyq.get()
            wsock.send_multipart([worker_addr, '', pickle.dumps(ctask)])

    @spawn
    def read_rsps():
        while True:
            rsp = wsock.recv_multipart(copy=False)
            if rsp[2].bytes != '':
                print '< ', rsp[2].bytes, rsp[3].bytes
                rq.put(rsp[2:])
            readyq.put(rsp[0])

def setup_responder(addrs, rq):
    rsock = ctx.socket(zmq.ROUTER)
    rsock.bind(addrs['responses'])

    @spawn
    def send_responses():
        for rsp in rq:
            rsock.send_multipart(rsp)
    return send_responses

def main(addrs):
    # Channel holding (addr, task) pairs.
    tq = queue.Queue(0)
    # Queue holding response messages (as a list of raw zmq frames).
    rq = queue.Queue()

    setup_task_listeners(addrs, tq, rq)
    setup_worker_listener(addrs, tq, rq)
    # TODO: Will switch to a Nanny central wait loop
    setup_responder(addrs, rq).join()

if __name__ == "__main__":
    import addrs
    main(addrs.addrs)
