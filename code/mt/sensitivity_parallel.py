"""Spawn-safe reaction workers and launcher for plain sensitivity run scripts."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback
from ._execution import launcher_environment, wait_for_process

_state = None


def _initialize(mechanism, upstream, frozen, speed, solver):
    """Load two private Cantera objects once per process; no objects are shared."""
    import cantera as ct
    global _state
    gas1 = ct.Solution(mechanism)
    gas1.TPX = upstream
    gas = ct.Solution(mechanism)
    _state = (gas, gas1, frozen, speed, solver)


def _evaluate(task):
    """Evaluate both perturbations of one reaction, restoring state before each."""
    from .znd_functions import _sensitivity_lengths, _sensitivity_pair, zndsolve
    gas, gas1, frozen, speed, solver = _state
    index, perturbation = task

    def solve(reaction, multiplier):
        """Run one independent perturbed integration."""
        gas.set_multiplier(1.0)
        gas.TPX = frozen
        gas.set_multiplier(multiplier, reaction)
        return _sensitivity_lengths(zndsolve(gas, gas1, speed, **solver), solver['t_end'])

    try:
        return _sensitivity_pair(index, perturbation, solve)
    finally:
        gas.set_multiplier(1.0)


def reaction_results(indices, perturbation, workers, mechanism, upstream, frozen, speed, solver):
    """Yield completed reaction pairs immediately, keeping scheduling dynamic."""
    with ProcessPoolExecutor(
        max_workers=min(workers, len(indices)), mp_context=mp.get_context('spawn'),
        initializer=_initialize, initargs=(mechanism, upstream, frozen, speed, solver),
    ) as pool:
        futures = [pool.submit(_evaluate, (i, perturbation)) for i in indices]
        for future in as_completed(futures):
            yield future.result()


def launch(settings):
    """Use a guarded library entry point so the user's script needs no main guard."""
    with TemporaryDirectory(prefix='mt_sensitivity_') as directory:
        request, response = Path(directory) / 'request.pkl', Path(directory) / 'response.pkl'
        with request.open('wb') as stream:
            pickle.dump(settings, stream)
        process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'mt.sensitivity_parallel', str(request), str(response)],
            env=launcher_environment(), start_new_session=(os.name == 'posix'),
        )
        code = wait_for_process(process)
        if not response.exists():
            raise RuntimeError(f'Sensitivity launcher exited with code {code} without a result.')
        with response.open('rb') as stream:
            ok, value = pickle.load(stream)
        if not ok:
            raise RuntimeError(f'Sensitivity launcher failed:\n{value}')
        if code:
            raise RuntimeError(f'Sensitivity launcher exited with code {code}.')
        return value


def main():
    """Run the coordinator and return a compact result or an explicit failure."""
    from .znd_functions import _znd_sensitivity
    request, response = map(Path, sys.argv[1:])
    with request.open('rb') as stream:
        settings = pickle.load(stream)
    try:
        payload = (True, _znd_sensitivity(**settings))
    except Exception:
        payload = (False, traceback.format_exc())
    with response.open('wb') as stream:
        pickle.dump(payload, stream)
    return 0 if payload[0] else 1


if __name__ == '__main__':
    sys.exit(main())
