"""Execution backends shared by kinetic simulations and optimization."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def create_executor(backend, workers, *, max_tasks_per_worker=None):
    """Use isolated solver processes, scheduled by Python or native OpenMP."""
    if backend == 'openmp':
        from .openmp import OpenMPExecutor
        return OpenMPExecutor(workers, max_tasks_per_worker=max_tasks_per_worker)
    if backend == 'process':
        options = dict(max_workers=workers, mp_context=mp.get_context('spawn'))
        if max_tasks_per_worker is not None:
            options['max_tasks_per_child'] = max_tasks_per_worker
        return ProcessPoolExecutor(**options)
    raise ValueError("backend must be 'process' or 'openmp'")
