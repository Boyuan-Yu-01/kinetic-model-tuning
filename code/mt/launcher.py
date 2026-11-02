"""Run process pools from a guarded library entry point, not the user's script."""
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback
from ._execution import launcher_environment, wait_for_process


def launch(problem, operation, settings):
    """Exchange trusted local objects through private temporary files."""
    with TemporaryDirectory(prefix='mt_launcher_') as directory:
        request = Path(directory) / 'request.pkl'
        response = Path(directory) / 'response.pkl'
        with request.open('wb') as stream:
            pickle.dump((problem, operation, settings), stream)
        # A separate entry point prevents workers from rerunning the user script.
        process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'mt.launcher', str(request), str(response)],
            env=launcher_environment(),
            start_new_session=(os.name == 'posix'))
        code = wait_for_process(process, interrupt_signal=signal.SIGINT)
        if not response.exists():
            raise RuntimeError(f'Optimization launcher exited with code {code} without a result.')
        with response.open('rb') as stream:
            ok, value, reference, signature = pickle.load(stream)
        if not ok:
            raise RuntimeError(f'Optimization launcher failed:\n{value}')
        if code:
            raise RuntimeError(f'Optimization launcher exited with code {code}.')
        # Return the reference cache to the original caller as well as the result.
        problem._reference_results = reference
        problem._reference_signature = signature
        if operation == '_optimize':
            value.problem = problem
        return value


def main():
    """Execute a private operation in the launcher and serialize its result or traceback."""
    request, response = map(Path, sys.argv[1:])
    with request.open('rb') as stream:
        problem, operation, settings = pickle.load(stream)
    try:
        if operation not in ('_optimize', '_sample_candidates', '_starting_points'):
            raise ValueError('Unknown launcher operation.')
        # Private methods run here directly, without launching another subprocess.
        value = getattr(problem, operation)(**settings)
        payload = (True, value, problem._reference_results, problem._reference_signature)
    except BaseException:
        payload = (False, traceback.format_exc(), None, None)
    with response.open('wb') as stream:
        pickle.dump(payload, stream)
    return 0 if payload[0] else 1


if __name__ == '__main__':
    sys.exit(main())
