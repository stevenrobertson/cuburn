#!/usr/bin/env python2
import random, os, subprocess, sys

class Shower(object):
    def __init__(self):
        self.nodes = {}
        self.edges_by_src = {}
        self.had_webm = False
        self.y4m_header = None

    def walk_dir(self, dir):
        for i in os.listdir(dir):
            fn = os.path.join(dir, i)
            if not i.endswith('.h264') and not i.endswith('.webm'): continue
            if i.startswith('latest'): continue
            path = i.rsplit('.', 1)[0].rsplit('_', 1)[0]
            if '=' in i:
                src, dst = path.split('=')
                self.edges_by_src.setdefault(src, set()).add((dst, fn))
            else:
                self.nodes[path] = fn
            if i.endswith('.webm'):
                self.had_webm = True

    def output(self, path):
        sys.stderr.write(path)
        if self.had_webm:
            self._output_y4m(path)
        else:
            with open(path) as fp:
                self._output_raw(fp)

    def _output_y4m(self, path):
        subp = subprocess.Popen(['ffmpeg', '-r', '24', '-i', path, '-f', 'yuv4mpegpipe', '-pix_fmt', 'yuv420p', '-'],
                stdout=subprocess.PIPE)
        y4m_header = subp.stdout.readline()
        if self.y4m_header is None:
            self.y4m_header = y4m_header
            sys.stdout.write(y4m_header)
        elif self.y4m_header != y4m_header:
            raise ValueError('Different Y4M headers: %s vs %s', repr(y4m_header), repr(self.y4m_header))
        self._output_raw(subp.stdout)

    def _output_raw(self, fp):
        for chunk in iter(lambda: fp.read(1024 * 1024), ''):
            sys.stdout.write(chunk)

    def run_for(self, n):
        src = None
        for i in range(n):
            if src not in self.nodes:
                src = random.choice(self.nodes.keys())
            self.output(self.nodes[src])
            if src in self.edges_by_src:
                src, path = random.choice(list(self.edges_by_src[src]))
                self.output(path)
            else:
                src = None

def main():
    shower = Shower()
    while True:
        shower.walk_dir(sys.argv[1])
        shower.run_for(1000)

if __name__ == '__main__':
    main()
