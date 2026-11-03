# Hydrogen flame tuning with reaction orders

This example reproduces the **15 paired Kyuho hydrogen–oxygen–argon conditions**
at equivalence ratio 0.25. It fits three equally weighted targets against
`FFCM2H2.yaml`: log flame-speed error, log thermal-flame-thickness error, and
burned-temperature error scaled by 25 K. Thermal thickness is
`(T_burned - T_unburned) / max(dT/dx)`.

Run in an environment with this package, Cantera and SDToolbox installed:

```bash
cd example/hydrogen_LFS
python run.py
```

`run.py` contains the same `problem`, `tuning`, and `execution` groups as the
other examples. Local runs use eight processes; Slurm runs use the allocated
CPUs through the OpenMP scheduler. The launcher handles multiprocessing safely.

## Inputs and initial candidate

- `inputs/H2_1_step_stoi.yaml`: immutable original Kyuho template. Its units,
  parameter bounds and rate origins remain unchanged.
- `inputs/FFCM2H2.yaml`: reference mechanism.
- `inputs/start_model.yaml`: saved Kyuho candidate from run
  `h2_1_step_stoi_6491786`, used as the initial rate/order values. The raw
  template produced an unphysical supersonic FreeFlame solution in the first
  condition; the saved candidate has a resolved subsonic flame.

The old rate regularization is **commented out**, so the loss contains only the
three fitting objectives. This does not disable bounds or step limits.

## Tuning

Reaction 1 has explicit H2 and O2 orders, each bounded between -2 and 2.
These are **actual order bounds**, not multipliers. Optional `sum_bounds`
constrains the sum of the selected orders. All other reactions retain their
orders unless explicitly listed.

Stages release **A + orders**, then **A/Ea + orders**, then **A/Ea/b + orders**.
Sensitivity estimates are recomputed each round. Selected coordinates are
updated together using the existing constrained trust-region optimizer.
Order derivatives always use full-simulation finite differences. Available
Cantera adjoints are retained for flame-speed and burned-temperature rate
parameter derivatives; flame thickness is differentiated numerically.

The numerical anchor is obtained once from the reference flames' maximum
spatial temperature-gradient locations, then held fixed across stages.
Temperature is averaged harmonically and positive reactant concentrations
geometrically. The internal magnitude coordinate is the rate at this anchor;
exports retain ordinary YAML `A`, `b`, `Ea`, and `orders`. A is converted using
the YAML concentration unit system when orders change.

The original A bounds are 1e-3–1e8 times the template A. Absolute Ea limits are
4500–75000 cal/mol and b limits are -5–10. Later-stage A changes additionally
stay within 0.5–2 times the stage-entry A. Trust-region steps limit simultaneous
Ea, b and order changes. PLOG insertion is disabled for this example.

## Outputs

- `output/best_model.yaml`: latest model that passed stage validation.
- `output/results.json`: stage losses, case comparisons and fitted orders.
- `output/comparison.png`: speed, burned temperature and flame thickness.
- `output/order_anchor.json`: fixed reference anchor.
- `output/A`, `output/AE`, `output/AEB`: stage checkpoints, sensitivities,
  validation records and optimizer diagnostics.
- `output/solver_logs`: solver details kept out of the concise terminal report.
- `logs`: Slurm stdout/stderr when run on the cluster.

Final validation disables sensitivity calculations, repeats predictions on a
finer grid, and checks speed and thickness changes against 2%. A plateau or
iteration limit is reported separately from convergence.

## Tuning mathematics

For the irreversible hydrogen reaction, the forward rate is

$$
r_f=A T^b e^{-E_a/(RT)} C_{\mathrm{H_2}}^{n_{\mathrm{H_2}}} C_{\mathrm{O_2}}^{n_{\mathrm{O_2}}}.
$$

Define $a=\ln(r_{f,*}/r_{f,0,*})$ as the log rate multiplier at a fixed reference state. Relative to the baseline rate,

$$
\ln\frac{r_f}{r_{f,0}}=
a+\Delta b\ln\frac{T}{T_*}
-\frac{\Delta E_a}{R}\left(\frac1T-\frac1{T_*}\right)
+\sum_i\Delta n_i\ln\frac{C_i}{C_{i,*}}.
$$

At the reference state, only $a$ contributes. The tuner adjusts the YAML value of $A$ to enforce this relation. The anchor uses the maximum temperature gradient region of each reference flame, with a harmonic mean for temperature and geometric means for concentrations.

| Stage | Adjustable coordinates |
| --- | --- |
| A | $a$, $n_{\mathrm{H_2}}$, $n_{\mathrm{O_2}}$ |
| A/Ea | $a$, $E_a$, $n_{\mathrm{H_2}}$, $n_{\mathrm{O_2}}$ |
| A/Ea/b | $a$, $E_a$, $b$, $n_{\mathrm{H_2}}$, $n_{\mathrm{O_2}}$ |

Each stage starts from the preceding stage's best model. Order changes are additive, $n_i=n_{i,0}+\Delta n_i$, and each actual order is bounded between $-2$ and $2$ in this example. At fixed $a$, increasing the hydrogen order by $0.2$ multiplies the rate by $(C_{\mathrm{H_2}}/C_{\mathrm{H_2},*})^{0.2}$.

For each flame case, the fit uses the residual vector

$$
\mathbf r_i=\begin{bmatrix}
\ln(S_{u,i}/S_{u,i}^{\mathrm{ref}})\\
\ln(\delta_i/\delta_i^{\mathrm{ref}})\\
(T_{b,i}-T_{b,i}^{\mathrm{ref}})/(25\,\mathrm K)
\end{bmatrix},
\qquad L=\frac{1}{3N}\sum_i\|\mathbf r_i\|_2^2.
$$

The tuner estimates the order columns of $J=\partial\mathbf r/\partial\mathbf x$ by finite differences. It uses $\mathbf r+J\Delta\mathbf x$ to propose a bounded joint step and checks its improvement with full flame solves.
