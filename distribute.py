#!/usr/bin/env python2

import os
import sys
import socket
import argparse
import subprocess
import traceback

import gevent
import gevent.event
import gevent.queue
import gevent.pool
from gevent import monkey
monkey.patch_all()

import json
import warnings
from subprocess import Popen
from itertools import ifilter
from collections import namedtuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cuburn import render, filters, output, profile
from cuburn.genome import convert, use, db

ready_str = 'worker ready'
closing_encoder_str = 'closing encoder'
output_file_str = 'here is a file for you'
done_str = 'we done here'

def write_str(out, val):
  out.write(np.array([len(val)], '>u4').tostring())
  out.write(val)
  out.flush()

def write_filelike(out, filelike):
  filelike.seek(0, 2)
  out.write(np.array([filelike.tell()], '>u8').tostring())
  filelike.seek(0)
  buf = filelike.read(1024 * 1024)
  while buf:
    out.write(buf)
    buf = filelike.read(1024 * 1024)
  out.flush()

def read_str(infp):
  sz_buf = infp.read(4)
  assert len(sz_buf) == 4, 'Incomplete read of str size'
  assert sz_buf[0] == '\0', 'No str should be that big'
  sz = np.frombuffer(sz_buf, '>u4')[0]
  msg = infp.read(sz)
  assert len(msg) == sz, 'Incomplete read, expected %d got %d' % (sz, len(msg))
  return msg

def copy_filelike(infp, dst):
  sz_buf = infp.read(8)
  assert len(sz_buf) == 8, 'Incomplete read of filelike size'
  assert sz_buf[0] == '\0', 'No filelike should be that big'
  sz = np.frombuffer(sz_buf, '>u8')[0]
  recvd = 0
  while recvd < sz:
    # uh... why is int needed here?
    chunk_sz = int(min(1024 * 1024, sz - recvd))
    chunk = infp.read(chunk_sz)
    assert len(chunk) == chunk_sz, (
        'Incomplete chunk, expected %d (%s)got %d' % (sz, `sz_buf`, len(chunk)))
    recvd += len(chunk)

def work(args):
  addr = socket.gethostname().split('.')[0] + '/' + str(args.device)
  write_str(sys.stdout, ready_str)

  import pycuda.driver as cuda
  cuda.init()
  dev = cuda.Device(args.device)
  cuctx = dev.make_context(flags=cuda.ctx_flags.SCHED_BLOCKING_SYNC)

  try:
    job_text = read_str(sys.stdin)
    if job_text == done_str:
      return
    job_desc = json.loads(job_text)
    prof, gnm, times, name = map(job_desc.get, 'profile genome times name'.split())
    gprof = profile.wrap(prof, gnm)

    rmgr = render.RenderManager()
    arch = 'sm_{}{}'.format(
        dev.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR),
        dev.get_attribute(cuda.device_attribute.COMPUTE_CAPABILITY_MINOR))
    rdr = render.Renderer(gnm, gprof, arch=arch)
    last_render_time_ms = 0

    def save(buf):
      out, log = rdr.out.encode(buf)
      for suffix, file_like in out.items():
        write_str(sys.stdout, output_file_str)
        write_str(sys.stdout, suffix)
        write_filelike(sys.stdout, file_like)
        if getattr(file_like, 'close', None):
          file_like.close()

    evt = buf = next_evt = next_buf = None
    for idx, t in enumerate(list(times) + [None]):
      evt, buf = next_evt, next_buf
      if t is not None:
        next_evt, next_buf = rmgr.queue_frame(rdr, gnm, gprof, t)
      if not evt: continue
      if last_render_time_ms > 2000:
        while not evt.query():
          gevent.sleep(0.2)
      else:
        evt.synchronize()
      last_render_time_ms = evt.time()
      print >> sys.stderr, '%30s: %s (%3d/%3d), %dms' % (
          addr, name, idx, len(times), last_render_time_ms)
      sys.stderr.flush()

      save(buf)
    write_str(sys.stdout, closing_encoder_str)
    save(None)
    write_str(sys.stdout, done_str)
  finally:
    cuda.Context.pop()

Job = namedtuple('Job', 'genome name times retry_count')

def dispatch(args):
  pname, prof = profile.get_from_args(args)

  workers = args.worker
  if not workers:
    try:
      with open(os.path.expanduser('~/.cuburn-workers')) as fp:
        workers = filter(None, fp.read().split())
    except:
      traceback.print_exc()
      pass
  if not workers:
    print >> sys.stderr, ('No workers defined. Pass --workers or set up '
                          '~/.cuburn-workers with one worker per line.')
    sys.exit(1)

  gdb = db.connect(args.genomedb)

  job_queue = gevent.queue.JoinableQueue(5)
  active_job_group = gevent.pool.Group()
  def fill_jobs():
    for oid in args.flames:
      ids = [oid]
      if oid[0] == '@':
        with open(oid[1:]) as fp:
          ids = fp.read().split('\n')
      for id in ids:
        gnm, basename = gdb.get_anim(id)
        gprof = profile.wrap(prof, gnm)
        for name, times in profile.enumerate_jobs(gprof, basename, args,
                                                  resume=True):
          job_queue.put(Job(gnm, name, times, 0))
  job_filler = gevent.spawn(fill_jobs)

  def connect_to_worker(addr):
    host, device = addr.split('/')
    if host == 'localhost':
      distribute_path = os.path.expanduser('~/.cuburn_dist/distribute.py')
      args = [distribute_path, 'work', '--device', str(device)]
      subp = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
      assert read_str(subp.stdout) == ready_str
    else:
      connect_timeout = 5
      while True:
        try:
          subp = subprocess.Popen(
              ['ssh', host, '.cuburn_dist/distribute.py', 'work',
               '--device', str(device)],
              stdin=subprocess.PIPE, stdout=subprocess.PIPE)
          assert read_str(subp.stdout) == ready_str
          break
        except:
          traceback.print_exc()
          gevent.sleep(connect_timeout)
          connect_timeout *= 2
    return subp

  exiting = False
  worker_failure_counts = {}
  def run_job(addr):
    worker = connect_to_worker(addr)
    job = job_queue.get()
    evt = gevent.event.Event()
    def _run_job():
      try:
        if job is None:
          write_str(worker.stdin, done_str)
          worker.stdin.close()
          return
        job_desc = dict(profile=prof, genome=job.genome, times=list(job.times),
                        name=job.name)
        write_str(worker.stdin, json.dumps(job_desc))
        worker.stdin.close()
        while True:
          msg_name = read_str(worker.stdout)
          if msg_name == closing_encoder_str:
            evt.set()
          elif msg_name == output_file_str:
            filename = job.name + read_str(worker.stdout)
            with open(filename + '.tmp', 'w') as fp:
              copy_filelike(worker.stdout, fp)
            os.rename(filename + '.tmp', filename)
          else:
            assert msg_name == done_str, 'no known event ' + msg_name
            break
        worker_failure_counts[addr] = 0
      except:
        print >> sys.stderr, traceback.format_exc()
        worker_failure_counts[addr] = worker_failure_counts.get(addr, 0) + 1
        if job.retry_count < 3:
          job_queue.put(Job(job.genome, job.name, job.times, job.retry_count + 1))
      finally:
        job_queue.task_done()
        evt.set()
    greenlet = gevent.spawn(_run_job)
    active_job_group.add(greenlet)
    return greenlet, evt

  def run_worker(addr):
    while worker_failure_counts.get(addr) < 4 and not exiting:
      greenlet, evt = run_job(addr)
      evt.wait()

  worker_group = gevent.pool.Group()
  for addr in workers:
    worker_group.spawn(run_worker, addr)
  job_filler.join()

  # Flush all outstanding jobs and, possibly, retries
  while job_queue.join():
    active_job_group.join()
    if job_queue.empty():
      break

  # Close the remaining workers
  exiting = True
  map(job_queue.put, [None] * len(worker_group))
  worker_group.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Render fractal flames on multiple GPUs.')

    cmd_parser = parser.add_subparsers()
    dispatch_parser = cmd_parser.add_parser(
        'dispatch', help='Dispatch tasks to workers.')
    dispatch_parser.add_argument('flames', metavar='ID', type=str, nargs='+',
        #fromfile_prefix_chars='@',
        help='Flames to render (prefix playlist with @)')
    dispatch_parser.add_argument('--worker', metavar='ADDRESS', nargs='*',
        help='Worker address (in the form "host/device_id")')
    dispatch_parser.add_argument('-d', '--genomedb', metavar='PATH', type=str,
        help="Path to genome database (file or directory, default '.')",
        default='.')
    profile.add_args(dispatch_parser)
    dispatch_parser.set_defaults(func=dispatch)

    worker_parser = cmd_parser.add_parser(
        'work', help='Perform a task (controlled by a dispatcher).')
    worker_parser.add_argument('--device', metavar='NUM', type=int,
        help='GPU device number to use, 0-indexed.')
    worker_parser.set_defaults(func=work)

    args = parser.parse_args()
    args.func(args)
