import sys, os, linecache

if getattr(sys, "frozen", False):
    # Taichi inspects the literal source text of @ti.kernel/@ti.func methods
    # at both decoration time and first-call time. That's not available from
    # a frozen PyInstaller build, so: (1) force Taichi's "is this a class
    # method" check to succeed (every kernel in this project is a method),
    # and (2) pre-register the real source of physics.py in linecache under
    # the exact filename Taichi will look it up by, using a copy of the file
    # that's bundled as a plain data file (see the .spec).
    import taichi.lang.kernel_impl as _tik
    _tik._inside_class = lambda level_of_class_stackframe: True

    for relpath in ("partevo/physics.py",):
        srcpath = os.path.join(sys._MEIPASS, relpath)
        if os.path.isfile(srcpath):
            with open(srcpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            linecache.cache[relpath] = (len(lines), None, lines, relpath)

from .physics import World
