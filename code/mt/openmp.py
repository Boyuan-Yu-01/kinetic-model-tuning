"""OpenMP scheduling with isolated persistent Python worker processes.

Functions and arguments must be pickleable (the same restriction as a process
pool). Native OpenMP threads dynamically claim queued tasks. Each owns one
persistent Python process, so Cantera and LSODA never share an interpreter.
"""
from concurrent.futures import Executor, Future
from pathlib import Path
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import threading


class OpenMPTaskError(RuntimeError):
    """Worker exception, including its remote traceback."""


class OpenMPExecutor(Executor):
    def __init__(self, max_workers=None, *, executable=None, max_tasks_per_worker=None):
        self._count = (os.cpu_count() or 1) if max_workers is None else max_workers
        if self._count < 1:
            raise ValueError('max_workers must be positive')
        if max_tasks_per_worker is not None and (not isinstance(max_tasks_per_worker, int) or max_tasks_per_worker < 1):
            raise ValueError('max_tasks_per_worker must be a positive integer')
        root = Path(__file__).resolve().parent.parent
        binary = Path(executable or os.environ.get('MT_OPENMP_RUNNER', root / 'native/openmp_runner'))
        if not binary.is_file():
            raise RuntimeError('OpenMP runner missing: run python native/build_openmp.py first')
        env = os.environ.copy()
        env['MT_OPENMP_PARENT_PID'] = str(os.getpid())
        env['PYTHONPATH'] = os.pathsep.join([str(root), *filter(None, sys.path)])
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '1'
        self._directory = tempfile.TemporaryDirectory(prefix='mt-openmp-')
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending = {}
        self._serial = 0
        self._closed = False
        self._failure = None
        try:
            self._process = subprocess.Popen(
                [str(binary.resolve()), str(self._count), sys.executable,
                 str(root / 'mt/openmp_worker.py'), str(max_tasks_per_worker or 0)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, text=True, bufsize=1, env=env,
                start_new_session=True)
        except BaseException:
            self._directory.cleanup()
            raise
        self._reader = threading.Thread(target=self._read, name='openmp-results', daemon=True)
        self._reader.start()

    def submit(self, fn, /, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError('cannot submit after shutdown')
            if self._failure:
                raise RuntimeError(self._failure)
            self._serial += 1
            path = str(Path(self._directory.name) / str(self._serial))
            future = Future()
            try:
                with open(path, 'wb') as stream:
                    pickle.dump((fn, args, kwargs), stream, protocol=pickle.HIGHEST_PROTOCOL)
                future.set_running_or_notify_cancel()
                self._pending[path] = future
            except BaseException as error:
                self._pending.pop(path, None)
                Path(path).unlink(missing_ok=True)
                future.set_exception(error)
                return future
        try:
            with self._write_lock:
                self._process.stdin.write(path + '\n')
                self._process.stdin.flush()
        except BaseException as error:
            with self._lock:
                pending = self._pending.pop(path, None)
            if pending is not None:
                future.set_exception(error)
                Path(path).unlink(missing_ok=True)
        return future

    def _read(self):
        try:
            for line in self._process.stdout:
                status, path = line.rstrip('\n').split(' ', 1)
                with self._lock:
                    future = self._pending.pop(path, None)
                if future is None:
                    continue
                try:
                    if status != 'OK':
                        raise OpenMPTaskError('Python worker exited unexpectedly')
                    with open(path + '.result', 'rb') as stream:
                        success, value = pickle.load(stream)
                    if not success:
                        raise OpenMPTaskError(value)
                    future.set_result(value)
                except BaseException as error:
                    future.set_exception(error)
                finally:
                    Path(path).unlink(missing_ok=True)
                    Path(path + '.result').unlink(missing_ok=True)
        finally:
            with self._lock:
                self._failure = 'OpenMP coordinator exited'
                pending, self._pending = self._pending, {}
            for future in pending.values():
                future.set_exception(OpenMPTaskError(self._failure))

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.terminate()
        else:
            self.shutdown(wait=True)
        return False

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._write_lock:
            with self._lock:
                self._closed = True
            if not self._process.stdin.closed:
                self._process.stdin.close()
        if wait:
            try:
                self._reader.join()
                self._process.wait()
            except BaseException:
                self.terminate()
                raise
            self._process.stdout.close()
            self._directory.cleanup()
        else:
            threading.Thread(target=self.shutdown, daemon=True).start()

    def terminate(self):
        """Abort the whole process group, fail pending futures, and reap workers."""
        try:
            os.killpg(self._process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self._process.wait()
        self._reader.join()
        self._closed = True
        try:
            self._process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        self._process.stdout.close()
        self._directory.cleanup()
