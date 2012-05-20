import os
import json

import convert

class GenomeDB(object):
    """
    Abstract base class for accessing genomes by ID. This is likely to be
    extended in the future.
    """
    def __init__(self):
        self.stashed = {}
    def _get(self, id):
        raise NotImplementedError()
    def get(self, id):
        if id in self.stashed:
            return self.stashed[id]
        return self._get(id)
    def stash(self, id, gnm):
        self.stashed[id] = gnm

    def get_anim(self, name, half=False):
        """
        Given the identifier of any type of genome that can be converted to an
        animation, do so.

        Returns `(gnm, basename)`, where gnm is the animation genome as a
        plain Python dict and basename is probably a suitable name for output
        files.
        """
        basename = os.path.basename(name)
        split = basename.rsplit('.', 1)
        head, ext = split[0], split[1] if len(split) == 2 else ''
        if ext in ('json', 'flam3', 'flame'):
            basename = head

        if os.path.isfile(name) and ext in ('flam3', 'flame'):
            with open(name) as fp:
                gnm_str = fp.read()
            flames = convert.XMLGenomeParser.parse(gnm_str)
            if len(flames) != 1:
                warnings.warn(
                        '%d flames in file, only using one.' % len(flames))
            gnm = convert.flam3_to_node(flames[0])
        else:
            gnm = self.get(name)

        if gnm['type'] == 'node':
            gnm = convert.node_to_anim(gnm, half=half)
        elif gnm['type'] == 'edge':
            gnm = convert.edge_to_anim(self, gnm)
        assert gnm['type'] == 'animation', 'Unrecognized genome type.'

        return gnm, basename

class OneFileDB(GenomeDB):
    def __init__(self, dct):
        assert dct.get('type') == 'onefiledb', "Doesn't look like a OneFileDB."
        self.dct = dct

    @classmethod
    def read(cls, path):
        with open(path) as fp:
            return cls(json.load(fp))

    def get(self, id):
        return self.dct[id]

class FilesystemDB(GenomeDB):
    def __init__(self, path):
        self.path = path

    def get(self, id):
        if not id.endswith('.json'):
            id += '.json'
        with open(os.path.join(self.path, id)) as fp:
            return json.load(fp)

def connect(path):
    if os.path.isfile(path):
        return OneFileDB.read(path)
    return FilesystemDB(path)
