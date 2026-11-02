# Kinetic Model Tuning

Sensitivity-guided optimization of reduced chemical kinetic mechanisms against a reference mechanism using **laminar flame speed**, **ignition delay time**, and **ZND detonation** targets.

Start with a human-initialized Cantera YAML mechanism, define the operating conditions and objectives in Python, and tune its rate parameters. The included methane–air and ethylene–air examples use FFCM2 as the reference.

## Capabilities

- Staged optimization of Arrhenius parameters: **A → A/Ea → A/Ea/b**.
- Sensitivity-based reaction selection, repeated as the model changes.
- Separate flame, ignition, and ZND condition sets and objective settings.
- Rate uncertainty constraints for reference-derived reactions and configurable bounds for artificial reactions.
- Optional sensitivity-guided PLOG insertion, independently adjustable pressure-entry magnitudes, and increasing or decreasing rate monotonicity constraints.
- Local multiprocessing and an optional native OpenMP scheduler for isolated solver processes.
- Saved mechanisms, optimization summaries, solver diagnostics, and comparison plots.

The workflow tunes an existing reaction scheme; it does not automatically discover a reaction mechanism. Available settings and validation rules vary by workflow.

## Examples

| Directory | Fuel | Targets |
|---|---|---|
| [`ethylene_LFS`](ethylene_LFS/run.py) | Ethylene–air | Flame speed and burned temperature |
| [`ethylene_LFS_IDT`](ethylene_LFS_IDT/run.py) | Ethylene–air | Flame speed, ignition delay, and both final temperatures |
| [`ethylene_ZND`](ethylene_ZND/run.py) | Ethylene–air | Final ZND temperature, induction length, and exothermic length |
| [`methane_LFS_IDT`](methane_LFS_IDT/run.py) | Methane–air | Flame speed, ignition delay, and both final temperatures |
| [`methane_ZND`](methane_ZND/run.py) | Methane–air | Final ZND temperature, induction length, and exothermic length |

Each directory contains a `run.py` and an `inputs/` folder. Conditions, uncertainty factors, parameter bounds, objectives, and execution settings are editable in `run.py`. Methane examples disable PLOG adaptation.

## Installation

Use Python **3.11 or newer** in a dedicated environment:

```bash
git clone https://github.com/Boyuan-Yu-01/kinetic-model-tuning.git
cd kinetic-model-tuning
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Windows execution has not been validated.

**SDToolbox must be installed separately** in the same Python environment. It is not included in `requirements.txt` or vendored in this repository. The current shared imports can require it even for flame/ignition workflows. Verify the dependencies before running:

```bash
python -c "import cantera, numpy, scipy, matplotlib, yaml, sdtoolbox"
```

The repository is run directly from its source layout; `pip install .` is not currently supported.

## Run an example

```bash
cd methane_LFS_IDT
python run.py
```

Running the script starts the full optimization. Review the case count, exploration samples, iteration limits, and workers first; these examples are research workloads rather than short demonstrations.

The scripts find the shared code and inputs relative to their own locations. Outside Slurm, they default to Python multiprocessing with eight workers. For a smaller local run, edit the `execution` dictionary:

```python
execution = {
    'backend': 'process',
    'workers': 4,
    'comparison_plots': True,
}
```

Retain any other workflow-specific execution settings already present in the example. The supplied launchers support plain run scripts without requiring a user-written multiprocessing guard.

Use a fresh output directory for a new run. Keep an existing result by changing `problem['output']` rather than overwriting it.

## Optimization workflow

1. Calculate reference results at the requested conditions.
2. Check the initial reduced model and perform the configured exploration when required by the workflow.
3. Rank reactions using objective-residual sensitivities and update selected parameters within their constraints.
4. Repeat sensitivity selection and parameter updates through the configured stages.
5. Where enabled, trial PLOG adaptation and retain it only if its acceptance checks pass.
6. Validate the resulting model and save numerical comparisons and plots.

Flame/ignition examples use log-squared speed and delay errors and scaled squared temperature errors. Combined workflows average within each branch before combining branches, so a branch does not receive more weight simply because it has more cases. See each `run.py` for the actual objectives and scales.

**Convergence, a loss plateau, a stalled update, and a round limit are distinct outcomes.** Solver validation confirms the implemented numerical checks; it does not establish that a reduced mechanism reproduces every target accurately. Inspect the reported fit errors and comparison plots.

## Outputs

Results are written beneath the example's configured output directory. Exact filenames depend on the workflow; common artifacts include:

| Artifact | Contents |
|---|---|
| `best_model.yaml` | Tuned Cantera mechanism |
| `results.json` or `results.yaml` | Stage outcomes and objective comparisons |
| Stage/round records | Parameter changes, sensitivities, and stop reasons |
| Validation records | Solver refinement and constraint checks |
| Comparison PNGs | Reference and tuned-model predictions |
| `optimizer.log` and solver logs | Detailed numerical diagnostics |

Generated results, logs, caches, and compiled binaries are excluded from Git.

## Parallel and cluster execution

Local multiprocessing requires no OpenMP compiler. For the optional native scheduler, build it on the target machine using a C++17 compiler with OpenMP support:

```bash
cd code
CXX=g++ python native/build_openmp.py
```

Then configure `backend='openmp'`. The scheduler runs isolated Python solver processes; it does not make an individual Cantera calculation use every allocated CPU.

The supplied examples select OpenMP when `SLURM_JOB_ID` is present and read their worker count from `SLURM_CPUS_PER_TASK`. Install the dependencies and build the scheduler before starting a Slurm run. Site-specific submission scripts are intentionally excluded from GitHub; provide your own allocation, environment setup, and `python run.py` launch command.

## Tests

From the repository root:

```bash
cd code
python -m unittest discover -s tests -v
```

Tests cover settings, constraints, sensitivity handling, PLOG adaptation, reporting, and execution helpers. OpenMP-dependent tests may skip when the native scheduler is unavailable. Passing the tests does not guarantee optimization convergence for a new mechanism.

## Source layout

```text
code/
  mt/                 Shared tuning and combustion calculation code
  native/             Optional OpenMP scheduler source and build helper
  tests/              Regression tests and fixtures
  requirements.txt    Python dependencies, excluding SDToolbox
ethylene_LFS/
ethylene_LFS_IDT/
ethylene_ZND/
methane_LFS_IDT/
methane_ZND/
```

See [flame temperature-profile documentation](code/FLAME_PROFILE.md) and the [combined ethylene example notes](ethylene_LFS_IDT/README.md) for further details.

## Attribution and licensing

FFCM2 mechanism inputs retain their original notices. External dependencies and reference data have their own licensing terms; see [THIRD_PARTY.md](THIRD_PARTY.md).

No project-wide license has been assigned yet. Public availability does not itself grant an open-source license.
