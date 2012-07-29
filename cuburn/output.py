import os
import tempfile
from cStringIO import StringIO
from subprocess import Popen, PIPE
import numpy as np
from numpy import float32 as f32, int32 as i32

import pycuda.driver as cuda

from code.util import ClsMod, launch
from code.output import pixfmtlib

import scipy.misc

if not hasattr(scipy.misc, 'toimage'):
    raise ImportError("Could not find scipy.misc.toimage. "
                      "Are scipy and PIL installed?")

try:
    import gevent
except ImportError:
    gevent = None

def launchC(name, mod, stream, dim, fb, *args):
    launch(name, mod, stream,
            (32, 8, 1), (int(np.ceil(dim.w/32.)), int(np.ceil(dim.h/8.))),
            fb.d_back, fb.d_front,
            i32(fb.gutter), i32(dim.w), i32(dim.astride), i32(dim.h),
            *args)

class Output(object):
    def convert(self, fb, gnm, dim, stream=None):
        """
        Convert a filtered buffer to whatever output format is needed by the
        writer.

        This function is intended for use by the Renderer, and should not be
        called by clients. It does not modify its instance.
        """
        raise NotImplementedError()

    def copy(self, fb, dim, pool, stream=None):
        """
        Schedule a copy from the device buffer to host memory, returning the
        target buffer(s).

        This function is intended for use by the Renderer, and should not be
        called by clients. It does not modify its instance.
        """
        raise NotImplementedError()

    def encode(self, host_frame):
        """
        Push `host_frame` (as returned from `Output.copy`) into the encoding
        pipeline, and return any completed media segments. If `host_frame` is
        None, flush the encoding pipeline.

        The return value is a 2-tuple `(media, logs)`. `media` is a dictionary
        mapping channel names (appropriate for use as file suffixes) to
        file-like objects containing the encoded media segments. `logs` is a
        dictionary containing log entries. Either or both entries can be empty
        at any time (and will typically be either populated on each frame
        except the flush, for non-temporal codecs, or will be empty on all
        frames except the flush, for temporal codecs.)

        Media segments are discretely decodeable chunks of content. The
        mapping of media segments to individual frames is not specified.
        """
        raise NotImplementedError()

    @property
    def suffix(self):
        """
        Return the file suffix that will be used. If more than one suffix will
        be used, the value returned is the one considered to be "primary".
        """
        raise NotImplementedError()


class PILOutput(Output, ClsMod):
    lib = pixfmtlib

    def __init__(self, codec='jpeg', quality=100, alpha=False):
        super(PILOutput, self).__init__()
        self.type, self.quality, self.alpha = codec, quality, alpha

    def convert(self, fb, gnm, dim, stream=None):
        launchC('f32_to_rgba_u8', self.mod, stream, dim, fb,
                fb.d_rb, fb.d_seeds)

    def copy(self, fb, dim, pool, stream=None):
        h_out = pool.allocate((dim.h, dim.w, 4), 'u1')
        cuda.memcpy_dtoh_async(h_out, fb.d_back, stream)
        return h_out

    def _convert_buf(self, buf):
        out = StringIO()
        img = scipy.misc.toimage(buf, cmin=0, cmax=1)
        img.save(out, self.type, quality=self.quality)
        out.seek(0)
        return out

    def encode(self, buf):
        if buf is None: return {}, []
        if self.type == 'jpeg':
            out = self._convert_buf(buf[:,:,:3])
            if self.alpha:
                alpha = self._convert_buf(buf[:,:,3])
                return {'_color.jpg': out, '_alpha.jpg': alpha}, []
            return {'.jpg': out}, {}
        return {'.'+self.type: self._convert_buf(buf)}, []

    @property
    def suffix(self):
        if self.type == 'jpeg':
            if self.alpha: return '_color.jpg'
            return '.jpg'
        return '.'+self.type

class X264Output(Output, ClsMod):
    lib = pixfmtlib

    profiles = (
      { 'normal': '--profile high444 --level 4.2'
      , '': ''
      })
    base = ('x264 --no-progress --input-depth 16 --sync-lookahead 0 '
            '--rc-lookahead 5 --muxer raw -o - - --log-level debug ')

    def __init__(self, profile='normal', csp='i444', crf=15,
                 x264opts='', alpha=False):
        super(X264Output, self).__init__()
        self.args = ' '.join([self.base, self.profiles[profile],
                              '--crf', str(crf), x264opts]).split()
        self.alpha = alpha
        self.csp = csp
        self.framesize = None
        self.zeros = None
        self.subp = None
        self.outf = None
        self.asubp = None
        self.aoutf = None

    def convert(self, fb, gnm, dim, stream=None):
        launchC('f32_to_rgba_u16', self.mod, stream, dim, fb,
                fb.d_rb, fb.d_seeds)

    def copy(self, fb, dim, pool, stream=None):
        h_out = pool.allocate((dim.h, dim.w, 4), 'u2')
        cuda.memcpy_dtoh_async(h_out, fb.d_back, stream)
        return h_out

    def _spawn_sub(self, framesize, alpha):
        res = '%dx%d' % (framesize[1], framesize[0])
        csp = 'yv12' if alpha else 'rgb'
        extras = ['--input-csp', csp, '--demuxer', 'raw', '--input-res', res]
        outf = tempfile.TemporaryFile(bufsize=0)
        if alpha:
            extras += ['--output-csp', 'i420', '--chroma-qp-offset', '24']
        else:
            extras += ['--output-csp', self.csp]
        subp = Popen(self.args + extras, stdin=PIPE, stderr=PIPE, stdout=outf)
        return outf, subp

    def _spawn(self, framesize):
        self.framesize = framesize
        self.outf, self.subp = self._spawn_sub(framesize, False)
        if self.alpha:
            self.aoutf, self.asubp = self._spawn_sub(framesize, True)
            bufsz = framesize[0] * framesize[1] / 2
            self.zeros = np.empty(bufsz, dtype='u2')
            self.zeros.fill(32767)

    def _flush_sub(self, subp):
        if gevent is not None:
            # Use non-blocking poll to allow applications to continue
            # rendering in other coros
            subp.stdin.close()
            log = ''
            while subp.poll() is None:
                log += subp.stderr.read()
                gevent.sleep(0.1)
            log += subp.stderr.read()
        else:
            (stdout, log) = subp.communicate()
        if subp.returncode:
            raise IOError("x264 exited with an error")
        return log

    def _flush(self):
        if self.subp is None:
            return {}, []
        log = self._flush_sub(self.subp)
        self.outf.seek(0)
        self.subp = None
        if self.alpha:
            alog = self._flush_sub(self.asubp)
            self.aoutf.seek(0)
            self.asubp = None
            return ({'_color.h264': self.outf, '_alpha.h264': self.aoutf},
                    [('x264_color', log), ('x264_alpha', alog)])
        return {'.h264': self.outf}, [('x264_color', log)]

    def _write(self, buf, subp):
        try:
            subp.stdin.write(buffer(buf))
        except IOError, e:
            print 'Exception while writing. Log:'
            print subp.stderr.read()
            raise e

    def encode(self, buf):
        out = ({}, [])
        if buf is None or self.framesize != buf.shape[:2]:
            out = self._flush()
        if buf is None:
            return out
        if self.subp is None:
            self._spawn(buf.shape[:2])
        self._write(np.delete(buf, 3, axis=2), self.subp)
        if self.alpha:
            self._write(buf[:,:,3].tostring(), self.asubp)
            self._write(buffer(self.zeros), self.asubp)
        return out

    @property
    def suffix(self):
        if self.alpha: return '_color.h264'
        return '.h264'

def get_output_for_profile(gprof):
    opts = dict(gprof.output._val)
    handler = opts.pop('type', 'jpeg')
    if handler in ('jpeg', 'png', 'tiff'):
        return PILOutput(codec=handler, **opts)
    elif handler == 'x264':
        return X264Output(**opts)
    raise ValueError('Invalid output type "%s".' % handler)
