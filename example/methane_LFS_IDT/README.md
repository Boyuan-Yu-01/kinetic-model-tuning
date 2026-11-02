# Methane–air LFS/IDT tuning

`run.py` fits a reduced methane mechanism against FFCM2. PLOG adaptation is disabled.

## Independent validation

The `validate/` folder contains reference and tuned mechanism snapshots and two
independent scripts. With the editable package installed in `ct-env`:

```bash
cd validate
python flame_speed.py
python ignition_delay.py
```

Both scripts use eight worker processes, print numerical comparisons, and save
plots only under `results/`. They do not retain solver logs or JSON results.

- **Flame:** 18 cases; pressure = 1, 10, 50 atm, initial temperature = 298 K,
  phi = 0.70, 0.84, 0.98, 1.12, 1.26, 1.40 (six evenly spaced validation points). Produces `results/flame_<pressure>_atm.png`
  for flame speed and burned temperature, separately at each pressure.
- **Ignition:** 90 cases; pressure = 1, 10, 50 atm, phi = 0.6, 1.0, 1.3, and ten
  temperatures uniformly spaced in 1000/T over 1600–1700 K. These phi values differ
  from the optimization grid. Delay is defined by maximum dT/dt for both mechanisms,
  since the reduced methane mechanism has no OH. Uses a constant-volume adiabatic
  reactor, end time 0.003 s, rtol 1e-11 and atol 1e-17. Produces a separate figure
  per pressure, with logarithmic ignition delay and linear final temperature
  versus 1000/T.

Each case is evaluated for FFCM2, the tuned `best_model.yaml`, and the untuned
`mechanism.yaml` copied from `inputs/`. Ignition plots use a solid reference line,
a dashed tuned line with open circles, and a dotted untuned line. Sensitivities are disabled during
validation. Replace `validate/best_model.yaml` to check a different tuned model.
These prediction comparisons do not replace a solver-convergence study.

Both scripts use `mt.validation.Validation`; calculations, summaries, and plotting
are implemented in the package. Set `workers=1` for sequential execution. Cached
results can be replotted without repeating the calculations.
