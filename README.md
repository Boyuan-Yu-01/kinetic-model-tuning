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

## Tuning mathematics

A single underline denotes a vector. A double underline denotes a matrix.

For an Arrhenius reaction with concentration-dependent forward kinetics, let $\mathcal S$ be the species in its forward rate law and $\mathcal U\subseteq\mathcal S$ the species whose orders are adjustable. Suppressing the reaction index, the forward rate is

```math
r_f=A T^b e^{-E_a/(RT)} \prod_{i\in\mathcal S}C_i^{n_i}.
```

The subscript $\ast$ means evaluated at the fixed anchor state. The comma separates labels. For example, $r_{f,0,\ast}$ is the forward rate of the baseline mechanism evaluated at the anchor. Likewise, $T_{\ast}$ is the anchor temperature and $C_{i,\ast}$ is the anchor concentration of species $i$.

Define $a=\ln(r_{f,\ast}/r_{f,0,\ast})$ as the log rate multiplier at a fixed reference state. Relative to the baseline rate,

```math
\ln\frac{r_f}{r_{f,0}}=
a+\Delta b\ln\frac{T}{T_{\ast}}
-\frac{\Delta E_a}{R}\left(\frac{1}{T}-\frac{1}{T_{\ast}}\right)
+\sum_{i\in\mathcal U}\Delta n_i\ln\frac{C_i}{C_{i,\ast}}.
```

### The anchor state

The tuner samples one state from each reference case according to the simulation type.

- **Laminar flame speed.** Find the interval with the largest $\Delta T/\Delta x$ and sample its midpoint.
- **Ignition delay.** Use the reactor state with the largest $dT/dt$ among the sampled states. This anchors the rate at the strongest heating even when ignition delay itself is measured with a species marker such as OH.
- **ZND.** Use the state at the induction length behind the shock, the location of peak thermicity. Interpolate the reference profile to that distance to obtain temperature, pressure, and composition, then calculate the species concentrations.

For a single simulation family with $N$ cases, combine the sampled temperatures and positive concentrations as follows:

```math
T_{\ast}=\left(\frac{1}{N}\sum_{j=1}^{N}\frac{1}{T_j}\right)^{-1},
\qquad
C_{i,\ast}=\left(\prod_{j=1}^{N} C_{i,j}\right)^{1/N}.
```

The harmonic mean averages inverse temperature, which appears in the activation-energy term. The geometric mean averages log concentration, which appears in the order term. For a fit that combines simulation families, the tuner averages inverse temperatures and log concentrations within each family first, then gives each family equal weight. The resulting anchor stays fixed across stages.

At the anchor, the temperature and concentration difference terms vanish, leaving $r_{f,\ast}/r_{f,0,\ast}=e^a$. If the tuner raises an order by $0.2$ at fixed $a$, it adjusts the YAML coefficient $A$ to preserve the rate at the anchor. At twice that species' anchor concentration, the order change multiplies the rate by $2^{0.2}\approx1.15$ with the other coordinates held fixed.

### Objective function

For example, a flame fit can use the residual vector

```math
\underline{r}_i=\begin{bmatrix}
\ln(S_{u,i}/S_{u,i}^{\mathrm{ref}})\\
\ln(\delta_i/\delta_i^{\mathrm{ref}})\\
(T_{b,i}-T_{b,i}^{\mathrm{ref}})/(25\,\mathrm K)
\end{bmatrix},
\qquad L=\frac{1}{3N}\sum_i\|\underline{r}_i\|_2^2.
```

### Sensitivity analysis

Stack the scaled residuals from all cases into a vector $\underline{r}$. For the objective above, each case contributes three rows. The Jacobian has one column per adjustable coordinate $x_j$ and records the effect of that coordinate on every residual.

```math
(\underline{\underline{J}})_{ij}=\frac{\partial r_i}{\partial x_j}
\approx \frac{r_i(\underline{x}+h_j\underline{e}_j)-r_i(\underline{x}-h_j\underline{e}_j)}{2h_j}.
```

Here $h_j$ is the perturbation in optimizer coordinates and $\underline{e}_j$ selects one coordinate. A positive entry means an increase in that coordinate raises the corresponding residual. For a log-ratio objective, the derivative is the fractional response of the predicted observable to that coordinate. For a temperature objective, it is the temperature response divided by the chosen scale. Consequently, objective weights and scales affect which changes the tuner prioritizes.

The centered formula illustrates the estimate when both perturbations are feasible and resolve successfully. Near bounds, the tuner can use feasible one-sided estimates. It checks estimate quality as the perturbation changes. Flame calculations can use adjoints for supported Arrhenius derivatives and uses finite differences for reaction orders and flame thickness. An unavailable derivative is skipped and rechecked in later rounds.

### From sensitivity to a proposed update

A Jacobian entry measures the signed response of one residual to one coordinate. To rank a reaction group $G$, the tuner scales each reliable column by its available movement $d_j$ and combines the column norms:

```math
\mathrm{score}(G)=
\frac{\sqrt{\sum_{j\in G_{\mathrm{reliable}}} d_j^2
\left\|(\underline{\underline{J}})_{:,j}\right\|_2^2}}
{\max(\|\underline{r}\|_2,1)}.
```

A large score means the reaction can move the residuals substantially. Whether those changes reduce the current loss depends on their signs and the current errors. For $L=\|\underline{r}\|_2^2$, the loss gradient is $2\underline{\underline{J}}^{\mathsf T}\underline{r}$.

### One optimization round

1. **Evaluate the current model.** Compute the observables and assemble the weighted, signed residuals. A profile contributes one row per comparison point.
2. **Estimate sensitivities and screen reactions.** Rank eligible reaction groups using their reliable derivatives and available movement. Recheck unavailable derivatives in later rounds.
3. **Propose a joint step.** For the selected columns $S$, solve the damped least-squares problem below. The feasible region combines parameter bounds, applicable rate constraints, and a trust region that limits the step size.
4. **Evaluate the trial mechanism.** Run the full simulations and check feasibility. Accept the step if it reduces the loss and the actual reduction agrees sufficiently with the predicted reduction. Reduce the trust radius and retry when a proposal performs poorly.
5. **Repeat and validate.** Refresh the sensitivities at the updated model. Record convergence, loss plateau, trust-region stall, or round limit separately. Check the saved model with the configured solver refinement.

```math
\min_{\Delta\underline{x}}
\left\|\underline{r}+\underline{\underline{J}}_S\Delta\underline{x}\right\|_2^2
+\lambda\left\|\Delta\underline{x}\right\|_2^2.
```

Damping $\lambda$ stabilizes the local step when parameters have similar effects. Optional baseline regularization instead penalizes departures of the mechanism from its starting parameters.

### Worked example

Suppose two cases both underpredict an observable. Two adjustable coordinates give the following scaled residual vector and Jacobian:

```math
\underline{r}=\begin{bmatrix}-1\\-0.6\end{bmatrix},
\qquad
\underline{\underline{J}}=\begin{bmatrix}2&1\\-2&1\end{bmatrix}.
```

The first column has the larger norm, but increasing its coordinate helps the first case and hurts the second. The second coordinate improves both. For illustration, set damping to zero and assume both coordinates are selected. The unconstrained joint step is $(0.1,0.8)^{\mathsf T}$, which zeros both linearized residuals. If the second coordinate can increase by at most $0.5$, the constrained solution is

```math
\Delta\underline{x}=\begin{bmatrix}0.1\\0.5\end{bmatrix},
\qquad
\underline{r}+\underline{\underline{J}}\Delta\underline{x}
=\begin{bmatrix}-0.3\\-0.3\end{bmatrix}.
```

The predicted squared loss falls from $1.36$ to $0.18$. Full simulations determine the actual reduction and whether the tuner accepts the step.

### What changes between stages

| Stage | Coordinates released | Coordinates held fixed | Purpose |
| --- | --- | --- | --- |
| **A** | Anchored rate strength and any configured reaction orders | Activation energy and temperature exponent | Adjust overall strength and concentration dependence. |
| **A/Ea** | Rate strength, activation energy, and any configured orders | Temperature exponent | Adjust inverse-temperature dependence while allowing the rate strength to readjust. |
| **A/Ea/b** | Rate strength, activation energy, temperature exponent, and any configured orders | Coordinates excluded by configuration | Add power-law temperature dependence for a final joint adjustment. |

Each stage uses the same sensitivity and update procedure. It starts from the preceding stage's best feasible model. Configured stage-relative limits are centered on that stage's entry model and intersected with the global bounds. Releasing $b$ last helps because the $b$ and $E_a$ terms can have similar effects over a narrow temperature range.

### Constraints and validation

- **Physical parameter bounds.** Limits apply to the reconstructed YAML values of $A$, $E_a$, and $b$, as configured. Because the anchored rate coordinate couples $A$ to the shape and order coordinates, the feasible update must satisfy the bounds jointly. Adjustable orders have individual bounds and may also have a bound on their sum.
- **Rate uncertainty.** For a borrowed reaction with a configured uncertainty factor $F$, the rate envelope constrains $k/k_0$ to the interval $[1/F,F]$ over the configured domain, relative to the designated baseline. A shared $A$-only multiplier scales the full rate directly, so its uncertainty bound reduces to a multiplier bound.
- **PLOG continuity and monotonicity.** Logarithmic interpolation between positive summed rates at pressure nodes makes the rate continuous in pressure. Monotonicity is an additional constraint. For an increasing rate, require $k(T,P_{j+1})\ge k(T,P_j)$ for every adjacent pair and every temperature in the configured interval. Reverse the inequality for decreasing rates. With either direction enabled, one direction must hold throughout the interval. Equal node rates permit flat segments. The tuner checks the resulting rate shape before accepting a candidate.
- **Simulation validity.** The mechanism must produce valid observables for the required cases. Final solver validation checks the saved mechanism separately from optimizer convergence.
