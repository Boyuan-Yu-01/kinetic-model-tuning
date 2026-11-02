# Kinetic model tuning

Sensitivity-guided tuning of human-initialized reduced combustion mechanisms against FFCM2. Supports laminar flame speed, ignition delay, and ZND objectives, staged A / A–Ea / A–Ea–b updates, uncertainty constraints, and optional monotonic PLOG rates.

## Layout

- `code/mt/`: shared Python implementation.
- `code/native/`: optional cluster OpenMP scheduler source; binaries are built on the target machine.
- `code/tests/`: regression tests and fixtures.
- `ethylene_LFS/`, `ethylene_LFS_IDT/`, `ethylene_ZND/`, `methane_LFS_IDT/`, `methane_ZND/`: inputs and editable run scripts.
- `SKETCH/`: previous local work, excluded from Git.

## Local setup

Use Python 3.11 or newer. Install dependencies in your environment:

```bash
python -m pip install -r code/requirements.txt
```

SDToolbox (`sdtoolbox`) must also be installed in the same environment for ZND and modules that import the shared ZND utilities. It is an external dependency, not bundled here. The existing `ct-env` environment contains it.

```bash
conda activate ct-env
cd methane_LFS_IDT
python run.py
```

Run scripts resolve inputs relative to their own location. Local runs use Python multiprocessing with eight workers by default; edit `execution` to change this. The launcher supports plain run scripts without adding a multiprocessing guard. Local runs do not require an OpenMP compiler.

Each example writes `output/`. Use a new output path or remove an old output before a fresh run. Full optimization can be expensive; running a script starts it immediately.

## Cluster

Slurm scripts are local-only and excluded from this repository. If you have a local submission script, edit its account, partition, nodes, and environment modules for your cluster, then submit from the example directory:

```bash
cd methane_LFS_IDT
sbatch methane_LFS_IDT.slurm
```

Slurm runs use the allocated CPUs and the OpenMP process scheduler. Dependencies must exist on the cluster. Batch stdout/stderr are written in the example directory; detailed diagnostics are in generated output folders.

## Tests

```bash
cd code
python -m unittest discover -s tests -v
```

Some tests require a compiled OpenMP scheduler and skip when unavailable. A passing unit suite does not establish convergence for every mechanism or condition.

## Repository contents

Source, example inputs, tests, and documentation are included. Generated outputs, caches, native binaries, and `SKETCH/` are ignored. Slurm submission scripts are also excluded.

FFCM2 mechanism data and other external material retain their original notices. See `THIRD_PARTY.md`. No project-wide license has been assigned; choose a license and confirm redistribution terms before publishing.
