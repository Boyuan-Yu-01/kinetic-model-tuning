"""Build on the deployment host: CXX=g++ python native/build_openmp.py."""
from pathlib import Path
import os
import shlex
import subprocess

if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    command = [*shlex.split(os.environ.get('CXX', 'g++')), '-O2', '-std=c++17',
               '-fopenmp', str(here / 'openmp_runner.cpp'), '-o', str(here / 'openmp_runner')]
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
