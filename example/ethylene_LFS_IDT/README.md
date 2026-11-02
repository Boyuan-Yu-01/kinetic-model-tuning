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
