"""Shared environment and interruption handling for guarded process launchers."""
import os
from pathlib import Path
import signal
import subprocess
import sys


def launcher_environment():
    """Preserve imports and allocation while limiting each worker to one CPU thread."""
    environment = os.environ.copy()
    root = str(Path(__file__).resolve().parent.parent)
    environment['PYTHONPATH'] = os.pathsep.join([root, *filter(None, sys.path)])
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[key] = '1'
    return environment


def wait_for_process(process, interrupt_signal=signal.SIGTERM):
    """Wait for a launcher, stopping and reaping its workers if the caller exits."""
    try:
        return process.wait()
    except BaseException:
        try:
            if os.name == 'posix':
                os.killpg(process.pid, interrupt_signal)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()
        raise
