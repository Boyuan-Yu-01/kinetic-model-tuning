"""Keep Cantera's parsed-YAML cache bounded while evaluating changing mechanisms."""
import atexit
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp
from threading import Lock
import time
import yaml


_lock = Lock()
_directory = None
_owner_pid = None
_modified_second = 0


def _cleanup(directory, owner_pid):
    """Remove only this process's scratch directory on normal shutdown."""
    if os.getpid() == owner_pid:
        shutil.rmtree(directory, ignore_errors=True)


@contextmanager
def candidate_file(scheme):
    """Reuse one private filename per process, invalidating Cantera's cache on each write."""
    global _directory, _owner_pid, _modified_second
    # Hold the lock through simulation so another thread cannot replace the file.
    with _lock:
        if _owner_pid != os.getpid():
            _directory = Path(mkdtemp(prefix='mt_candidate_'))
            _owner_pid = os.getpid()
            _modified_second = 0
            atexit.register(_cleanup, _directory, _owner_pid)
        mechanism = _directory / 'model.yaml'
        mechanism.write_text(yaml.safe_dump(scheme, sort_keys=False))
        # Cantera keys cached YAML by pathname and whole-second modification time.
        # Ordinary rapid rewrites can otherwise silently reuse the previous rates.
        _modified_second = max(int(time.time()), _modified_second + 1)
        timestamp = _modified_second * 1_000_000_000
        os.utime(mechanism, ns=(timestamp, timestamp))
        if mechanism.stat().st_mtime_ns != timestamp:
            raise RuntimeError('Cannot set a distinct candidate-file modification time.')
        yield mechanism
