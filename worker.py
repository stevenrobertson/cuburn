#!/usr/bin/env python2
# Render from a job server

import re
import os
import sys
import time
import uuid
import json
import socket
import itertools
from subprocess import check_output
from cStringIO import StringIO

import scipy
import redis

from cuburn import render, genome

pycuda = None

# The default maximum number of waiting jobs. Also used to determine when a
# job has expired.
QUEUE_LENGTH=50

def partition(pred, arg):
    return filter(pred, arg), filter(lambda a: not pred(a), arg)

def git_rev():
    os.environ['GIT_DIR'] = os.path.join(os.path.dirname(__file__) or '.', '.git')
    if 'FLOCK_PATH_IGNORE' not in os.environ:
        if check_output('git status -z -uno'.split()):
            return None
    return check_output('git rev-parse HEAD'.split()).strip()[:10]

uu = lambda t: ':'.join((t, uuid.uuid1().hex))

def get_temperature():
    id = pycuda.autoinit.device.pci_bus_id()
    try:
        out = check_output('nvidia-smi -q -d TEMPERATURE'.split())
    except:
        return ''
    idx = out.find('\nGPU ' + id)
    if idx >= 0:
        out.find('Gpu', idx)
        if idx >= 0:
            idx = out.find(':')
            if idx >= 0:
                return out[idx+1:idx+3]
    return ''

def push_frame(r, out):
    if out is None:
        return
    sid, sidx, ftag = out.idx
    # TODO: gotta put this in a module somewhere and make it adjustable
    noalpha = out.buf[:,:,:3]
    img = scipy.misc.toimage(noalpha, cmin=0, cmax=1)
    buf = StringIO()
    img.save(buf, 'jpeg', quality=95)
    buf.seek(0)
    head = ' '.join([sidx, str(out.gpu_time), ftag])
    r.rpush(sid + ':queue', head + '\0' + buf.read())
    print 'Pushed frame: %s' % head

def work(server):
    global pycuda
    import pycuda.autoinit
    rev = git_rev()
    assert rev, 'Repository must be clean!'
    r = redis.StrictRedis(server)
    wid = uu('workers')
    r.sadd('renderpool:' + rev + ':workers', wid)
    r.hmset(wid, {
        'host': socket.gethostname(),
        'devid': pycuda.autoinit.device.pci_bus_id(),
        'temp': get_temperature()
    })
    r.expire(wid, 180)
    last_ping = time.time()

    last_pid, last_gid, riter = None, None, None

    while True:
        task = r.blpop('renderpool:' + rev + ':queue', 10)
        now = time.time()
        if now > last_ping - 60:
            r.hset(wid, 'temp', get_temperature())
            r.expire(wid, 180)
            last_ping = now

        if not task:
            if riter:
                push_frame(r, riter.send(None))
                riter = None
            continue

        sid, sidx, pid, gid, ftime, ftag = task[1].split(' ', 5)
        if pid != last_pid or gid != last_gid or not riter:
            gnm = genome.Genome(json.loads(r.get(gid)))
            prof = json.loads(r.get(pid))
            gnm.set_profile(prof)
            renderer = render.Renderer()
            renderer.load(gnm)

            if riter:
                push_frame(r, riter.send(None))

            riter = renderer.render_gen(gnm, prof['width'], prof['height'])
            next(riter)
            last_pid, last_gid = pid, gid

        push_frame(r, riter.send(((sid, sidx, ftag), float(ftime))))

def iter_genomes(prof, gpaths, pname='540p'):
    """
    Walk a list of genome paths, yielding them in an order suitable for
    the `genomes` argument of `create_jobs()`.
    """

    for gpath in gpaths:
        gname = os.path.basename(gpath).rsplit('.', 1)[0]
        with open(gpath) as fp:
            gsrc = fp.read()
        gnm = genome.Genome(json.loads(gsrc))
        err, times = gnm.set_profile(prof)
        odir = 'out/%s/%s/untracked' % (pname, gname)
        gtimes = []
        for i, t in enumerate(times):
            opath = os.path.join(odir, '%05d.jpg' % (i+1))
            if not os.path.isfile(opath):
                gtimes.append((t, opath))
        if gtimes:
            if not os.path.isdir(odir):
                os.makedirs(odir)
            latest = odir.rsplit('/', 1)[0] + '/latest'
            if not os.path.isdir(latest):
                os.symlink('untracked', latest)
            yield gsrc, gtimes

def create_jobs(r, psrc, genomes):
    """Attention politicians: it really is this easy.

    `genomes` is an iterable of 2-tuples (gsrc, gframes), where `gframes` is an
    iterable of 2-tuples (ftime, fid).
    """
    pid = uu('profile')
    r.set(pid, psrc)
    for gsrc, gframes in genomes:
        # TODO: SHA-based? I guess that depends on whether we do precompilation
        # on the HTTP server which accepts job requests (and on whether the
        # grid remains homogeneous).
        gid = uu('genome')
        r.set(gid, gsrc)
        r.publish('precompile', gid)

        for ftime, fid in gframes:
            yield pid, gid, str(ftime), fid

def run_jobs(r, rev, jobs):
    # TODO: session properties
    sid = uu('session')
    qid = sid + ':queue'
    pending = {}    # sidx -> job, for any job currently in the queue
    waiting = []    # sidx of jobs in queue normally
    retry = []      # sidx of jobs in queue a second time

    def push(i, job):
        j = ' '.join((sid, str(i)) + job)
        r.rpush('renderpool:' + rev + ':queue', j)

    def pull(block):
        if block:
            ret = r.blpop(qid, 180)
            if ret is None:
                # TODO: better exception
                raise ValueError("Timeout")
            ret = ret[1]
        else:
            ret = r.lpop(qid)
            if ret is None: return
        tags, jpg = ret.split('\0', 1)
        sidx, gpu_time, ftag = tags.split(' ', 2)
        sidx, gpu_time = int(sidx), float(gpu_time)
        if sidx in waiting:
            waiting.remove(sidx)
        if sidx in retry:
            retry.remove(sidx)
        if sidx in pending:
            pending.pop(sidx)
        else:
            print 'Got two responses for %d' % sidx
        if retry and retry[0] < sidx - 2 * QUEUE_LENGTH:
            # TODO: better exception
            raise ValueError("Double retry!")
        expired, waiting[:] = partition(lambda w: w < sidx - QUEUE_LENGTH,
                                        waiting)
        for i in expired:
            push(i, pending[i])
            retry.append(i)
        return sidx, gpu_time, ftag, jpg

    try:
        for sidx, job in enumerate(jobs):
            while len(pending) > QUEUE_LENGTH:
                yield pull(True)
            ret = pull(False)
            if ret:
                yield ret
            pending[sidx] = job
            waiting.append(sidx)
            push(sidx, job)
    except KeyboardInterrupt:
        print 'Interrupt received, flushing already-dispatched frames'

    while pending:
        print '%d...' % len(pending)
        yield pull(True)

def client(ppath, gpaths):
    rev = git_rev()
    assert rev, 'Repository must be clean!'
    r = redis.StrictRedis()
    if not r.scard('renderpool:' + rev + ':workers'):
        # TODO: expire workers when they disconnect
        print 'No workers available at local cuburn revision, exiting.'
        return

    with open(ppath) as fp:
        psrc = fp.read()
    prof = json.loads(psrc)
    pname = os.path.basename(ppath).rsplit('.', 1)[0]

    jobiter = create_jobs(r, psrc, iter_genomes(prof, gpaths, pname))
    for sidx, gpu_time, ftag, jpg in run_jobs(r, rev, jobiter):
        with open(ftag, 'w') as fp:
            fp.write(jpg)
        print 'Wrote %s (took %g msec)' % (ftag, gpu_time)

if __name__ == "__main__":
    if sys.argv[1] == 'work':
        work('192.168.1.3')
    else:
        client(sys.argv[1], sys.argv[2:])
