#!/usr/bin/env python3
"""Stable entrypoint for the namespaced Dark Factory runtime."""
from __future__ import annotations
import os
import runpy
import sys
from pathlib import Path

runtime = Path(__file__).resolve().parent / 'runtime'
os.environ.setdefault('DARKFAC_PROJECT_ROOT', str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(runtime))
if len(sys.argv) < 2:
    raise SystemExit('usage: darkfac.py <module|script|harness> ...')
mode = sys.argv.pop(1)
if mode == 'harness':
    sys.argv[0] = 'core.harness.runner'
    runpy.run_module('core.harness.runner', run_name='__main__', alter_sys=True)
elif mode == 'module' and len(sys.argv) >= 2:
    module = sys.argv.pop(1)
    sys.argv[0] = module
    runpy.run_module(module, run_name='__main__', alter_sys=True)
elif mode == 'script' and len(sys.argv) >= 2:
    relative = Path(sys.argv.pop(1))
    if relative.is_absolute() or '..' in relative.parts:
        raise SystemExit('runtime script path must be relative')
    script = (runtime / relative).resolve()
    if runtime.resolve() not in script.parents:
        raise SystemExit('runtime script escapes bundle')
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name='__main__')
else:
    raise SystemExit(f'unsupported Dark Factory command: {mode}')
