import ctypes
import tempita

def ppr_ptx(src):
    # TODO: Add variable realignment
    indent = 0
    out = []
    for line in [l.strip() for l in src.split('\n')]:
        if not line:
            continue
        if len(line.split()) == 1 and line.endswith(':'):
            out.append(line)
            continue
        if '}' in line and '{' not in line:
            indent -= 1
        if line.startswith('@'):
            out.append(' ' * ((indent - 1) * 4) + line)
        else:
            out.append(' ' * (indent * 4) + line)
        if '{' in line and '}' not in line:
            indent += 1
    return '\n'.join(out)

def multisub(tmpl, subs):
    while '{{' in tmpl:
        tmpl = tempita.Template(tmpl).substitute(subs)
    return tmpl

class PTXAssembler(object):
    """
    Assembles PTX fragments into a module.
    """

    def __init__(self, ctx, entries, build_tests=False):
        self.assemble(ctx, entries, build_tests)

    def deporder(self, unsorted_instances, instance_map, ctx):
        """
        Do a DFS on PTXFragment dependencies, and return an ordered list of
        instances where no fragment depends on any before it in the list.

        `unsorted_instances` is the list of instances to sort.
        `instance_map` is a dict of types to instances.
        """
        seen = {}
        def rec(inst):
            if inst in seen: return seen[inst]
            deps = filter(lambda d: d is not inst, map(instance_map.get,
                       callable(inst.deps) and inst.deps(self) or inst.deps))
            return seen.setdefault(inst, 1+max([0]+map(rec, deps)))
        map(rec, unsorted_instances)
        return sorted(unsorted_instances, key=seen.get)

    def _safeupdate(self, dst, src):
        """dst.update(src), but no duplicates allowed"""
        non_uniq = [k for k in src if k in dst]
        if non_uniq: raise KeyError("Duplicate keys: %s" % ','.join(key))
        dst.update(src)

    def assemble(self, ctx, entries, build_tests):
        """
        Build the PTX source for the given set of entries.
        """
        # Get a property, dealing with the callable-or-data thing. This is
        # cumbersome, but flexible; when finished, it may be simplified.
        def pget(prop):
            if callable(prop): return prop(ctx)
            return prop

        instances = {}
        unvisited_entries = list(entries)
        entry_names = {}
        tests = []
        parsed_entries = []
        while unvisited_entries:
            ent = unvisited_entries.pop(0)
            seen, unvisited = set(), [ent]
            while unvisited:
                frag = unvisited.pop(0)
                seen.add(frag)
                inst = instances.setdefault(frag, frag())
                for dep in pget(inst.deps):
                    if dep not in seen:
                        unvisited.append(dep)
                if build_tests:
                    for test in pget(inst.tests):
                        if test not in tests:
                            if test not in instances:
                                unvisited_entries.append(test)
                            tests.append(test)

            tmpl_namespace = {'ctx': ctx}
            entry_start, entry_end = [], []
            for inst in self.deporder(map(instances.get, seen), instances, ctx):
                self._safeupdate(tmpl_namespace, pget(inst.subs))
                entry_start.append(pget(inst.entry_start))
                entry_end.append(pget(inst.entry_end))
            entry_start_tmpl = '\n'.join(filter(None, entry_start))
            entry_end_tmpl = '\n'.join(filter(None, reversed(entry_end)))
            name, args, body = pget(instances[ent].entry)
            tmpl_namespace.update({'_entry_name_': name, '_entry_args_': args,
                '_entry_body_': body, '_entry_start_': entry_start_tmpl,
                '_entry_end_': entry_end_tmpl})

            entry_tmpl = (".entry {{ _entry_name_ }} ({{ _entry_args_ }})\n"
                "{\n{{_entry_start_}}\n{{_entry_body_}}\n{{_entry_end_}}\n}\n")
            parsed_entries.append(multisub(entry_tmpl, tmpl_namespace))
            entry_names[ent] = name

        prelude = []
        tmpl_namespace = {'ctx': ctx}
        for inst in self.deporder(instances.values(), instances, ctx):
            prelude.append(pget(inst.prelude))
            self._safeupdate(tmpl_namespace, pget(inst.subs))
        tmpl_namespace['_prelude_'] = '\n'.join(filter(None, prelude))
        tmpl_namespace['_entries_'] = '\n\n'.join(parsed_entries)
        tmpl = "{{ _prelude_ }}\n{{ _entries_ }}"

        self.entry_names = entry_names
        self.source = ppr_ptx(multisub(tmpl, tmpl_namespace))
        self.instances = instances
        self.tests = tests

class PTXFragment(object):
    """
    Wrapper for sections of template PTX.

    In order to provide the best optimization, and avoid a web of hard-coded
    parameters, the PTX module may be regenerated and recompiled several times
    with different or incomplete launch context parameters. To this end, avoid
    accessing the GPU in such functions, and do not depend on context values
    which are marked as "tuned" in the LaunchContext docstring being
    available.

    The final compilation pass is guaranteed to have all "tuned" values fixed
    in their final values for the stream.

    Template code will be processed recursively until all "{{" instances have
    been replaced, using the same namespace each time.

    Note that any method which does not depend on 'ctx' can be replaced with
    an instance of the appropriate return type. So, for example, the 'deps'
    property can be a flat list instead of a function.
    """

    def deps(self, ctx):
        """
        Returns a list of PTXFragment objects on which this object depends
        for successful compilation. Circular dependencies are forbidden,
        but multi-level dependencies should be fine.
        """
        return [DeviceHelpers]

    def subs(self, ctx):
        """
        Returns a dict of items to add to the template substitution namespace.
        The entire dict will be assembled, including all dependencies, before
        any templates are evaluated.
        """
        return {}

    def prelude(self, ctx):
        """
        Returns a template string containing any code (variable declarations,
        probably) that should be inserted at module scope. The prelude of
        all deps will be inserted above this prelude.
        """
        return ""

    def entry_start(self, ctx):
        """
        Returns a template string that should be inserted at the top of any
        entry point which depends on this method. The entry starts of all
        deps will be inserted above this entry prelude.
        """
        return ""

    def entry_end(self, ctx):
        """
        As above, but at the end of the calling function, and with the order
        reversed (all dependencies will be inserted after this).
        """
        return ""

    def tests(self, ctx):
        """
        Returns a list of PTXTest classes which will test this fragment.
        """
        return []

    def set_up(self, ctx):
        """
        Do start-of-stream initialization, such as copying data to the device.
        """
        pass

class PTXEntryPoint(PTXFragment):
    # Human-readable entry point name
    name = ""

    def entry(self, ctx):
        """
        Returns a 3-tuple of (name, args, body), which will be assembled into
        a function.
        """
        raise NotImplementedError

    def call(self, ctx):
        """
        Calls the entry point on the device. Haven't worked out the details
        of this one yet.
        """
        pass

class PTXTest(PTXEntryPoint):
    """PTXTests are semantically equivalent to PTXEntryPoints, but they
    differ slightly in use. In particular:

    * The "name" property should describe the test being performed,
    * ctx.stream will be synchronized before 'call' is run, and should be
      synchronized afterwards (i.e. sync it yourself or don't use it),
    * call() should return True to indicate that a test passed, or
      False (or raise an exception) if it failed.
    """
    pass

class DeviceHelpers(PTXFragment):
    def __init__(self):
        self._forstack = []

    prelude = ".version 2.1\n.target sm_20\n\n"

    def _get_gtid(self, dst):
        return "{\n// Load GTID into " + dst + """
        .reg .u16 tmp;
        .reg .u32 cta, ncta, tid, gtid;

        mov.u16         tmp,    %ctaid.x;
        cvt.u32.u16     cta,    tmp;
        mov.u16         tmp,    %ntid.x;
        cvt.u32.u16     ncta,   tmp;
        mul.lo.u32      gtid,   cta,    ncta;

        mov.u16         tmp,    %tid.x;
        cvt.u32.u16     tid,    tmp;
        add.u32         gtid,   gtid,   tid;
        mov.b32 """ + dst + ",  gtid;\n}"

    def subs(self, ctx):
        return {
            'PTRT': ctypes.sizeof(ctypes.c_void_p) == 8 and '.u64' or '.u32',
            'get_gtid': self._get_gtid
            }


