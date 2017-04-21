import io
import os
import tempfile
from cStringIO import StringIO
from subprocess import Popen, PIPE
import numpy as np
from numpy import float32 as f32, int32 as i32

import pycuda.driver as cuda

from code.util import ClsMod, launch
from code.output import pixfmtlib


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
        import scipy.misc
        if not hasattr(scipy.misc, 'toimage'):
            raise ImportError("Could not find scipy.misc.toimage. "
                              "Are scipy and PIL installed?")

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
        import scipy.misc
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

class TiffOutput(Output, ClsMod):
    lib = pixfmtlib

    def __init__(self, alpha=False):
        import tifffile
        if 'filename' in tifffile.TiffWriter.__init__.__func__.func_doc:
            raise EnvironmentError('tifffile version too old!')
        super(TiffOutput, self).__init__()
        self.alpha = alpha

    def convert(self, fb, gnm, dim, stream=None):
        launchC('f32_to_rgba_u16', self.mod, stream, dim, fb,
                fb.d_rb, fb.d_seeds)

    def copy(self, fb, dim, pool, stream=None):
        h_out = pool.allocate((dim.h, dim.w, 4), 'u2')
        cuda.memcpy_dtoh_async(h_out, fb.d_back, stream)
        return h_out

    def encode(self, buf):
        import tifffile

        if buf is None: return {}, []
        if not self.alpha:
            buf = buf[:,:,:3]
        out = io.BytesIO()
        tifffile.imsave(out, buf)
        out.seek(0)
        return {'.tiff': out}, []

    @property
    def suffix(self):
        return '.tiff'


class ProResOutput(Output, ClsMod):
    lib = pixfmtlib

    def __init__(self, fps=24):
        super(ProResOutput, self).__init__()
        self.fps = fps
        self._outf = None
        self._subp = None
        self._dim = None

    def convert(self, fb, gnm, dim, stream=None):
        self._dim = dim
        launchC('f32_to_yuv444p12', self.mod, stream, dim, fb,
                fb.d_rb, fb.d_seeds)

    def copy(self, fb, dim, pool, stream=None):
        h_out = pool.allocate((3, dim.h, dim.w), 'u2')
        cuda.memcpy_dtoh_async(h_out, fb.d_back, stream)
        return h_out

    def _spawn(self):
        self._outf = tempfile.NamedTemporaryFile(bufsize=0, suffix='mov')
        cmd = ('ffmpeg -loglevel panic -f rawvideo -pix_fmt yuv444p12le '
               '-s {w}x{h} -r {fps} -i - -c:v prores -f mov -y {fn}').format(
                       w=self._dim.w, h=self._dim.h, fps=self.fps,
                       fn=self._outf.name)
        self._subp = Popen(cmd.split(), stdin=PIPE)

    def _flush(self):
        if not self._subp:
            return {}, []
        self._subp.stdin.close()
        self._subp.wait()
        if self._subp.returncode:
            raise IOError("ffmpeg exited with an error")
        # get a new handle, delete the named file
        outf = open(self._outf.name)
        self._outf.close()
        self._outf, self._subp = None, None
        return {'.mov': outf}, []

    def encode(self, host_frame):
        if host_frame is None:
            return self._flush()
        if not self._subp:
            self._spawn()
        self._subp.stdin.write(buffer(host_frame))
        return {}, []

    @property
    def suffix(self):
        return '.mov'


class X264Output(Output, ClsMod):
    lib = pixfmtlib

    profiles = (
      { 'normal': '--profile high444 --level 4.2'
      , '': ''
      })
    base = ('--no-progress --input-depth 16 --sync-lookahead 0 '
            '--rc-lookahead 5 --muxer raw -o - - --log-level debug')

    def __init__(self, profile='normal', csp='i444', crf=15,
                 command='x264', x264opts='', alpha=False):
        super(X264Output, self).__init__()
        self.args = ' '.join([command, self.base, self.profiles[profile],
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

class VPxOutput(Output, ClsMod):
    lib = pixfmtlib

    base = ('vpxenc --end-usage=3 -p 1 -q --cpu-used=-8 --lag-in-frames=5 '
            '--min-q=2 --disable-kf --arnr-maxframes=3 -o - -')

    def __init__(self, codec='vp9', fps=24, crf=15, pix_fmt='yuv420p'):
        super(VPxOutput, self).__init__()
        self.codec = codec
        self.pix_fmt = pix_fmt

        self.dim = None
        self.subp = None
        self.outf = None

        self.args = self.base.split()
        if pix_fmt == 'yuv420p':
            self.out_filter = 'f32_to_yuv444p'
        else:
            assert codec == 'vp9'
            if pix_fmt == 'yuv444p':
                self.out_filter = 'f32_to_yuv444p'
                self.args += ['--profile=1', '--i444']
            elif pix_fmt == 'yuv420p10':
                assert codec == 'vp9'
                self.out_filter = 'f32_to_yuv420p10'
                self.args += ['-b', '10', '--input-bit-depth=10', '--profile=2']
            elif pix_fmt == 'yuv444p10':
                assert codec == 'vp9'
                self.out_filter = 'f32_to_yuv444p10'
                self.args += ['-b', '10', '--input-bit-depth=10',
                              '--profile=3', '--i444']
            elif pix_fmt == 'yuv444p12':
                assert codec == 'vp9'
                self.out_filter = 'f32_to_yuv444p12'
                self.args += ['-b', '12', '--input-bit-depth=12',
                              '--profile=3', '--i444']
            else:
                raise ValueError('Invalid pix_fmt: ' + pix_fmt)
        self.args += ['--codec=' + codec, '--cq-level=' + str(crf), '--fps=%d/1' % fps]
        if codec == 'vp9':
            self.args += ['-t', '4']

    def convert(self, fb, gnm, dim, stream=None):
        self.dim = dim
        launchC(self.out_filter, self.mod, stream, dim, fb,
                fb.d_rb, fb.d_seeds)

    def copy(self, fb, dim, pool, stream=None):
        fmt = 'u1'
        if self.pix_fmt in ('yuv444p10', 'yuv420p10', 'yuv444p12'):
            fmt = 'u2'
        dims =  (3, dim.h, dim.w)
        if self.pix_fmt == 'yuv420p10':
            dims = (dim.h * dim.w * 6 / 4,)
        h_out = pool.allocate(dims, fmt)
        cuda.memcpy_dtoh_async(h_out, fb.d_back, stream)
        return h_out

    def _spawn(self):
        extras = ['-w', self.dim.w, '-h', self.dim.h]
        num_columns = int(max(0, min(3, np.log2(self.dim.w) - 8.9)))
        if num_columns:
            extras.append('--tile-columns=%d' % num_columns)

        self.outf = tempfile.TemporaryFile(bufsize=0)
        self.subp = Popen(map(str, self.args + extras),
                          stdin=PIPE, stderr=PIPE, stdout=self.outf)

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
            raise IOError("vpxenc exited with an error")
        return log

    def _flush(self):
        if self.subp is None:
            return {}, []
        log = self._flush_sub(self.subp)
        self.outf.seek(0)
        self.subp = None
        return {'.webm': self.outf}, [('webm', log)]

    def _write(self, buf, subp):
        try:
            subp.stdin.write(buffer(buf))
        except IOError, e:
            print 'Exception while writing. Log:'
            print subp.stderr.read()
            raise e

    def encode(self, buf):
        out = ({}, [])
        if buf is None:
            return self._flush()
        if self.subp is None:
            self._spawn()
        if self.pix_fmt == 'yuv420p':
            # Perform terrible chroma subsampling
            self._write(buf[0].tostring(), self.subp)
            self._write(buf[1,::2,::2].tostring(), self.subp)
            self._write(buf[2,::2,::2].tostring(), self.subp)
        else:
            self._write(buf, self.subp)
        return out

    @property
    def suffix(self):
        return '.webm'


def get_output_for_profile(gprof):
    opts = dict(gprof.output._val)
    handler = opts.pop('type', 'jpeg')
    if handler in ('jpeg', 'png'):
        return PILOutput(codec=handler, **opts)
    elif handler == 'tiff':
        return TiffOutput(**opts)
    elif handler == 'x264':
        return X264Output(**opts)
    elif handler == 'vp8':
        return VPxOutput(codec='vp8', fps=gprof.fps, **opts)
    elif handler == 'vp9':
        return VPxOutput(codec='vp9', fps=gprof.fps, **opts)
    elif handler == 'prores':
        return ProResOutput(fps=gprof.fps, **opts)
    raise ValueError('Invalid output type "%s".' % handler)
