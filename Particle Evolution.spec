# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# taichi ships compiled backend loaders + runtime bitcode that PyInstaller's
# static analysis can't find on its own; easygui pulls in tkinter's Tcl/Tk
# data files for its file dialogs. collect_all() finds both, but its result
# has to actually be fed into Analysis() below (previously it was collected
# and then overwritten with empty lists).
for pkg in ('taichi', 'easygui'):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# Non-code assets: bundle these so they land alongside the exe.
# SPECPATH is the folder containing this .spec file, set automatically by
# PyInstaller, so this works regardless of the current working directory.
datas += [
    (os.path.join(SPECPATH, 'examples'), 'examples'),
    (os.path.join(SPECPATH, 'img'), 'img'),
]

# Taichi has to re-read the literal source text of any @ti.kernel/@ti.func
# method to compile it, both at decoration time and first-call time. That
# text isn't available from compiled bytecode alone, so ship a plain-data
# copy of physics.py alongside the compiled one; partevo/__init__.py reads
# this copy back at startup and feeds it to linecache under the same
# filename Taichi looks it up by. Add more entries here if kernels/funcs
# ever move into another module.
datas += [
    (os.path.join(SPECPATH, 'partevo', 'physics.py'), 'partevo'),
]

a = Analysis(
    ['ui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# Linux build. PyInstaller isn't a cross-compiler — running this spec on
# Linux is what makes it a Linux binary; there's no separate "target OS"
# switch. The options below that only matter on macOS (target_arch,
# codesign_identity, entitlements_file, argv_emulation) are dropped since
# they're no-ops here.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Particle Evolution',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
)
