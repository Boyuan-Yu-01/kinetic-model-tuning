# Combined ethylene LFS and IDT tuning

`run.py` uses 15 flame cases and 27 ignition cases.
No supplied reference delays are needed. All reference and candidate ignition cases
use the same physical integration end time of 0.003 s. There is no ignition
wall-clock timeout. FFCM2 is recalculated for reference delay and final temperature.
IDT uses maximum OH concentration gradient, constant-volume adiabatic reactors.

The combined loss is (L_flame + L_ignition)/2 by default. Each branch averages
its own cases; each averages its two equally weighted objectives:
log-squared speed/delay and squared final-temperature error scaled by 25 K.
`branch_weights` and objective weights are editable. Invalid cases reject the
candidate. Temperature-profile fitting is disabled.

The A → AE → AEB stages and optional sensitivity-selected R2 PLOG trial at
1, 50, 100 bar reuse the flame workflow. Original uncertainty anchors are retained.
Validation refines flame grids and tightens IDT tolerances tenfold; accepted
models require flame-speed and IDT drift <=2%, IDT final-T drift <=1 K, and valid
rate-envelope and PLOG monotonicity checks.

Outputs are isolated in output/. The overall results.json lists flame and
ignition comparisons separately. comparison.png shows flame comparisons;
ignition_comparison.png shows reference/model delay and final temperatures.
Plots default to linear flame speed, logarithmic IDT, and linear temperature.

No optimization has been submitted. Run with the project's configured Python
environment; backend and workers remain editable in execution.

## Independent validation

`validate/` contains FFCM2 and a snapshot of the tuned `best_model.yaml`. Replace
that model file when validating a later optimization result.

With `ct-env` active, run either script independently from `validate/`:

```bash
python flame_speed.py
python ignition_delay.py
```

Both use eight worker processes (`WORKERS = 8`) and print errors to the terminal.
They save comparison PNGs only, with no persistent solver logs or JSON files.

- **Flame:** 27 cases; P = 1, 10, 50 atm, T = 298 K, phi = 0.6–1.4 in steps of 0.1.
  Saves `results/flame_<pressure>_atm.png`, with flame speed and burned temperature in each pressure figure.
- **Ignition:** 90 cases; P = 1, 10, 50 atm, phi = 0.6, 1.0, 1.3, and ten temperatures
  evenly spaced in 1000/T from 1000/1600 to 1000/1400. The temperature spacing in K
  is therefore nonuniform. Uses maximum d[OH]/dt in an adiabatic constant-volume
  reactor and a shared 0.003 s endpoint. Saves one comparison figure per pressure,
  with logarithmic delay and linear final temperature versus 1000/T.

Each case is evaluated for both mechanisms. These are prediction comparisons at
the original solver settings, not independent grid/tolerance convergence studies.

Both scripts use `mt.validation.Validation`; calculations, summaries, and plotting
are implemented in the package. Set `workers=1` for sequential execution. Cached
results can be replotted without repeating the calculations.
