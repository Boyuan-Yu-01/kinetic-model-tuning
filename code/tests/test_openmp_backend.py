"""Run with python -m unittest discover -s tests -p test_openmp_backend.py.

Requires native/build_openmp.py first; deliberately runs no combustion models.
"""
import importlib.util
import operator
import os
import select
import signal
import subprocess
import sys
from pathlib import Path
import time
import unittest
from concurrent.futures import as_completed

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mt_openmp_test_backend', ROOT / 'mt/openmp.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


@unittest.skipUnless((ROOT / 'native/openmp_runner').is_file(), 'build OpenMP runner first')
class OpenMPTests(unittest.TestCase):
    def test_map_and_exception_and_worker_reuse(self):
        with backend.OpenMPExecutor(2) as pool:
            self.assertEqual(list(pool.map(abs, [-2, -1, 0, 1])), [2, 1, 0, 1])
            pids = {pool.submit(os.getpid).result(timeout=10) for _ in range(8)}
            self.assertLessEqual(len(pids), 2)
            self.assertNotIn(os.getpid(), pids)
            with self.assertRaises(backend.OpenMPTaskError):
                pool.submit(operator.truediv, 1, 0).result(timeout=10)
            self.assertEqual(pool.submit(pow, 2, 5).result(timeout=10), 32)

    def test_dynamic_completion(self):
        with backend.OpenMPExecutor(2) as pool:
            slow = pool.submit(time.sleep, 0.4)
            fast = pool.submit(abs, -5)
            self.assertIs(next(as_completed([slow, fast], timeout=10)), fast)

    def test_many_submissions_do_not_deadlock(self):
        with backend.OpenMPExecutor(2) as pool:
            futures = [pool.submit(abs, -i) for i in range(2000)]
            self.assertEqual(sum(f.result(timeout=10) for f in futures), sum(range(2000)))

    def test_worker_crash_propagates(self):
        with backend.OpenMPExecutor(1) as pool:
            with self.assertRaises(backend.OpenMPTaskError):
                pool.submit(os._exit, 7).result(timeout=10)
            self.assertEqual(pool.submit(abs, -1).result(timeout=10), 1)

    def test_worker_recycling(self):
        with backend.OpenMPExecutor(1, max_tasks_per_worker=1) as pool:
            pids = [pool.submit(os.getpid).result(timeout=10) for _ in range(3)]
            self.assertEqual(len(set(pids)), 3)

    def test_exception_context_terminates(self):
        started = time.monotonic()
        with self.assertRaisesRegex(ValueError, 'interrupt test'):
            with backend.OpenMPExecutor(1) as pool:
                pool.submit(time.sleep, 30)
                raise ValueError('interrupt test')
        self.assertLess(time.monotonic() - started, 10)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux parent-death protection')
    def test_parent_death_kills_private_worker_group(self):
        source = f"""
import importlib.util, os, time
spec = importlib.util.spec_from_file_location('backend', {str(ROOT / 'mt/openmp.py')!r})
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
pool = backend.OpenMPExecutor(1)
worker = pool.submit(os.getpid).result(timeout=10)
print(pool._process.pid, worker, flush=True)
time.sleep(60)
"""
        helper = subprocess.Popen([sys.executable, '-c', source], stdout=subprocess.PIPE, text=True)
        coordinator = None
        try:
            self.assertTrue(select.select([helper.stdout], [], [], 15)[0], 'helper did not start')
            coordinator, worker = map(int, helper.stdout.readline().split())
            helper.kill()
            helper.wait(timeout=10)

            def running(pid):
                try:
                    # Orphan zombies awaiting init's reap are not live workers.
                    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
                except FileNotFoundError:
                    return False

            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and any(running(pid) for pid in (coordinator, worker)):
                time.sleep(0.05)
            self.assertFalse(running(coordinator), 'orphan native coordinator')
            self.assertFalse(running(worker), 'orphan Python worker')
        finally:
            if helper.poll() is None:
                helper.kill()
                helper.wait(timeout=10)
            helper.stdout.close()
            if coordinator is not None:
                try:
                    os.killpg(coordinator, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_terminate(self):
        pool = backend.OpenMPExecutor(1)
        future = pool.submit(time.sleep, 30)
        pool.terminate()
        with self.assertRaises(backend.OpenMPTaskError):
            future.result(timeout=10)


if __name__ == '__main__':
    unittest.main()
