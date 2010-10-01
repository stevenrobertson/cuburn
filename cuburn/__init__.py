
from collections import namedtuple

Flag = namedtuple('Flag', 'level desc')

class DebugSettings(object):
    """
    Container for default debug settings.
    """
    def __init__(self, items):
        self.items = items
        self.values = {}
        self.level = 1
    def __getattr__(self, name):
        if name not in self.items:
            raise KeyError("Unknown debug flag name!")
        if name in self.values:
            return self.values[name]
        return (self.items[name].level <= self.level)
    def format_help(self):
        name_len = min(30, max(map(len, self.items.keys())))
        fmt = '%-' + name_len + 's %d %s'
        return '\n'.join([fmt % (k, v.level, v.desc)
                          for k, v in self.items.items()])

debug_flags = dict(
    count_writes = Flag(3,  "Count the number of points written per thread "
                            "when doing iterations."),
    count_rounds = Flag(3,  "Count the number of times the iteration loop "
                            "runs per thread when doing iterations.")
    )

