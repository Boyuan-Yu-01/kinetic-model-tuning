"""MPI entry point: mpiexec -n N python -m mt.mpi optimizer.py [arguments]."""
import argparse
from pathlib import Path
import runpy
import sys
import traceback


def main():
    """Run the user script only on rank zero; other ranks serve evaluation tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script', help='Plain Python optimization script on shared storage')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor
    except ImportError as error:
        raise RuntimeError('MPI mode requires mpi4py and a compatible MPI runtime.') from error
    from .search import limit_threads, mpi_context
    comm = MPI.COMM_WORLD
    try:
        if comm.size < 2:
            raise ValueError('MPI mode needs at least two ranks: one coordinator and one worker.')
        if MPI.Query_thread() < MPI.THREAD_SERIALIZED:
            raise RuntimeError('MPI mode requires at least MPI.THREAD_SERIALIZED support.')
        script = Path(args.script).resolve()
        if not script.is_file():
            raise FileNotFoundError(script)
        limit_threads()
        sys.path.insert(0, str(script.parent))
        # Use already allocated ranks; never dynamically spawn MPI processes.
        with MPICommExecutor(comm, root=0, main=False, path=list(sys.path)) as pool:
            if pool is not None:
                with mpi_context(pool, comm.size - 1):
                    print(f'MPI: {comm.size} ranks; 1 coordinator and {comm.size - 1} workers.', flush=True)
                    sys.argv = [str(script), *args.arguments]
                    try:
                        runpy.run_path(str(script), run_name='__main__')
                    except SystemExit as error:
                        if error.code not in (None, 0):
                            traceback.print_exc()
                            comm.Abort(1)
                    except BaseException:
                        # Abort inside the executor context so workers cannot hang on shutdown.
                        traceback.print_exc()
                        comm.Abort(1)
    except BaseException:
        traceback.print_exc()
        comm.Abort(1)


if __name__ == '__main__':
    main()
