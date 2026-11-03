# Kinetic Model Tuning

Sensitivity-guided optimization of reduced chemical kinetic mechanisms against a reference mechanism using **laminar flame speed**, **ignition delay time**, and **ZND detonation** targets.

Start with a human-initialized Cantera YAML mechanism, define the operating conditions and objectives in Python, and tune its rate parameters. The included methane–air and ethylene–air examples use FFCM2 as the reference.

## Capabilities

- Staged optimization of Arrhenius parameters: **A → A/Ea → A/Ea/b**.
- Optional joint tuning of explicit reaction orders, with fixed reference-state anchors and optional order-sum bounds.
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
| [`ethylene_LFS_IDT`](example/ethylene_LFS_IDT/run.py) | Ethylene–air | Flame speed, ignition delay, and both final temperatures |
| [`ethylene_ZND`](example/ethylene_ZND/run.py) | Ethylene–air | Final ZND temperature, induction length, and exothermic length |
| [`methane_LFS_IDT`](example/methane_LFS_IDT/run.py) | Methane–air | Flame speed, ignition delay, and both final temperatures |
| [`hydrogen_LFS`](example/hydrogen_LFS/run.py) | Hydrogen–oxygen–argon | Flame speed, flame thickness, burned temperature; tunable reaction orders |

For individual calculations without optimization, see [`example/calculations`](example/calculations/README.md): commented `run_ZND.py`, `run_LFS.py`, and `run_IDT.py` scripts using FFCM2.

Each directory contains a `run.py` and an `inputs/` folder. Conditions, uncertainty factors, parameter bounds, objectives, and execution settings are editable in `run.py`. Methane examples disable PLOG adaptation.

## Installation

Use Python **3.11 or newer** in a dedicated environment:

```bash
git clone https://github.com/Boyuan-Yu-01/kinetic-model-tuning.git
cd kinetic-model-tuning
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Windows execution has not been validated.

**SDToolbox must be installed separately** in the same Python environment. It is not included in `requirements.txt` or vendored in this repository. The current shared imports can require it even for flame/ignition workflows. Verify the dependencies before running:

```bash
python -c "import cantera, numpy, scipy, matplotlib, yaml, sdtoolbox"
```

The distribution is named `kinetic-model-tuning`; its Python import name is `mt`. An editable installation makes source edits available immediately and allows imports from any directory. Keep this checkout in place, or reinstall after moving it.

For an existing Conda environment, activate it and install the checkout:

```bash
conda activate ct-env
python -m pip install -e /path/to/kinetic-model-tuning
python -c "import mt; print(mt.__file__)"
```

SDToolbox remains a separately installed dependency. The optional OpenMP build instructions below assume an editable source checkout.

### Using the installed package

After installation, activate the same environment whenever you use the package:

```bash
conda activate ct-env
```

Your scripts can then import `mt` from any working directory, without adding the repository to `sys.path`:

```python
from mt import CombustionCalculator
from mt.flame_ignition import run_flame_ignition_case
```

An editable install links the environment to your source checkout rather than copying it. Source changes are available to newly started Python processes without reinstalling. Restart an existing Python session to load updated code. Keep the checkout in place; if you move it, rerun `python -m pip install -e /new/path/to/kinetic-model-tuning` in the same environment.

Installing the package does not start an optimization or automatically download future GitHub updates. To update a cloned checkout, use `git pull` after handling any local changes. Rerun the installation command if package dependencies change.


## Run an example

```bash
cd example/methane_LFS_IDT
python run.py
```

Running the script starts the full optimization. Review the case count, exploration samples, iteration limits, and workers first; these examples are research workloads rather than short demonstrations.

The scripts import the installed `mt` package and find inputs relative to their own locations. Outside Slurm, they default to Python multiprocessing with eight workers. For a smaller local run, edit the `execution` dictionary:

```python
execution = {
    'backend': 'process',
    'workers': 4,
    'comparison_plots': True,
}
```

Retain any other workflow-specific execution settings already present in the example. The supplied launchers support plain run scripts without requiring a user-written multiprocessing guard.

Use a fresh output directory for a new run. Keep an existing result by changing `problem['output']` rather than overwriting it.

## Validation without optimization

The shared `mt.validation.Validation` API calculates reference and candidate predictions,
then returns cached results for summaries and plots. See the short run scripts in
[`ethylene_ZND/validate`](example/ethylene_ZND/validate/plot_comparison.py),
[`ethylene_LFS_IDT/validate`](example/ethylene_LFS_IDT/validate/flame_speed.py), and
[`methane_LFS_IDT/validate`](example/methane_LFS_IDT/validate/ignition_delay.py).

```python
from mt.validation import Validation

validation = Validation(
    reference='FFCM2.yaml',
    models={'Tuned': 'best_model.yaml', 'Untuned': 'mechanism.yaml'},
    fuel='CH4', workers=8, output_dir='results',
    save_results=False, save_logs=False,
)
result = validation.flame(
    conditions={'pressure_atm': [1, 10, 50], 'temperature_K': [298.],
                'phi': [.7, 1., 1.4]},
    solver={'transport_model': 'multicomponent', 'soret': False},
)
result.print_summary()
result.plot(quantities=['laminar_flame_speed', 'T_flame'],
            x='phi', separate_by='pressure')
```

- `.flame()` and `.ignition()` expand Cartesian condition grids. `.znd()` pairs
  pressure, temperature, phi and `end_time_s` arrays; scalar values broadcast.
- Use either `pressure_atm` or `pressure_bar`, never both. Temperatures are in K.
- `workers=1` runs sequentially; larger values use isolated worker processes.
  Plain calling scripts do not need a multiprocessing main guard.
- Summaries include signed percentage errors, absolute natural-log ratios for
  positive speeds/delays/lengths, and temperature differences in K.
- Flame and ZND plots default to linear axes; ignition delay defaults to logarithmic.
  Use `yscales`, `styles`, `quantities`, and `x` to change plots without recalculating.
- Calculations return SI values. Plots display flame speed in cm/s, delay in µs,
  and ZND distance in mm. IDT plots support `x='1000/T'`.
- Failed solves are retained in `result.failures`; other predictions remain usable.
  Plots leave gaps at failures. Warnings are retained in individual result rows.
- Only calling `.plot()` saves PNGs by default. Enable `save_results` for JSON or
  `save_logs` for solver logs. These prediction comparisons are not a substitute
  for a grid/tolerance convergence study.

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

Existing cluster results and logs are included as snapshots under each example’s `output/` and `logs/`. These are records of completed or interrupted runs, not a guarantee of convergence. Caches and compiled binaries remain excluded from Git.

To rerun an example while preserving the published snapshot, change `problem['output']` to a fresh directory such as `ROOT / 'output_new'`.

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
example/
  calculations/       Standalone ZND, flame, and ignition examples
  ethylene_LFS_IDT/
  ethylene_ZND/
  methane_LFS_IDT/
```

See [flame temperature-profile documentation](code/FLAME_PROFILE.md) and the [combined ethylene example notes](example/ethylene_LFS_IDT/README.md) for further details.

## Optional reaction-order tuning

In the staged workflows, add actual order bounds under `problem['reactions']`:

```python
'orders': {
    1: {
        'bounds': {'H2': (-2., 2.), 'O2': (-2., 2.)},
        # Optional sum of these selected orders:
        'sum_bounds': (1., 2.),
    },
},
```

Omit `orders` to retain the existing fixed-order behavior. These bounds are
additive actual orders, unlike the legacy `OrderParameters.ratios` API.
The selected reactant orders must already appear in the input YAML. Order
tuning supports irreversible Arrhenius and PLOG reactions declared artificial;
published rate uncertainty factors are not applied to changing-order reactions.
Borrowed reactions retain their original rate uncertainty constraints.

Orders remain active during all A / A–Ea / A–Ea–b stages. Each reaction's rate
and order coordinates are updated jointly; unselected reactions keep their
current values. Numerical derivative probes use bounded steps and are rechecked
each round. Unavailable derivatives are not treated as zero sensitivities.

Reference anchors are sampled at maximum **dT/dx** for flames, maximum **dT/dt**
for ignition, and the **induction-length** location for ZND. One fixed anchor is
formed with a harmonic mean temperature and geometric mean positive reactant
concentrations, weighting cases equally within equally weighted families.
Anchors are independent of the chosen ignition-delay marker. They are computed
from the reference mechanism and retained throughout the run. YAML energy and
concentration units govern export; there is no separate energy-unit setting.

For n reactions with one Arrhenius expression each and m reactions with two
adjustable orders, the final stage has 3n + 2m coordinates. PLOG nodes can add
rate coordinates while sharing one set of orders per reaction. The complete
example is documented in [hydrogen_LFS](example/hydrogen_LFS/README.md).
