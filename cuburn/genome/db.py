import os
import json

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

class OneFileDB(object):
    def __init__(self, dct):
        assert dct.get('type') == 'onefiledb', "Doesn't look like a OneFileDB."
        self.dct = dct

    @classmethod
    def read(cls, path):
        with open(path) as fp:
            return cls(json.load(fp))

    def get(self, id):
        return self.dct[id]

class FilesystemDB(object):
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
