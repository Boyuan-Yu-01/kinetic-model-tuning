# Flame temperature profile objective

The flame example accepts this optional entry in `problem['objectives']`:

```python
'T_profile': {
    'loss': 'squared', 'weight': 1.,
    'scale': 25.,  # K; alternatively {'reference_temperature_rise_fraction': .01}
    'points': 100,
    'lower_fraction': .01,
    'upper_fraction': .99,
    'margin_fraction': .1,
},
```

Each flame is shifted to its own first upward 50% temperature-rise crossing.
Reference crossings at lower_fraction and upper_fraction define a width W.
The interval extends margin_fraction * W beyond each crossing. Uniform points
include both endpoints. The grid stays fixed across every candidate and stage.
PCHIP interpolation preserves local shape; no distance stretching or extrapolation.
A reference lacking coverage raises an error; an uncovered candidate is invalid.
Increase solver width or reduce the interval margins if coverage is insufficient.

The profile objective is the mean squared temperature error divided by scale².
It is averaged with speed and burned-temperature objectives by their weights,
then averaged across cases. Increasing point count does not multiply its weight.
Existing scalar-only callers remain unchanged when profile is omitted.

Raw reference profiles and comparison grids are saved in FFCM2_results.json;
accepted and refined stage profiles are saved in best_profiles.json and
validation_profiles.json. Existing grid-refinement acceptance is based on speed
(2%); profile coverage and finite residuals are checked, but a separate quantitative
profile grid-convergence tolerance is not yet implemented.

## Optional PLOG adaptation for flame tuning

the flame workflow now includes the same `tuning['plog']` settings as ZND:
R2 is eligible, with fixed nodes at 1, 50, 100 **bar** (inlet cases remain in atm).
After the A stage, resolved reaction sensitivities determine trial eligibility.
The trial initially preserves the accepted rate, then releases the PLOG node
magnitudes with either monotonic direction allowed. A trial is retained only
when its loss improvement exceeds 0.1% and flame grid/rate/monotonicity checks
pass. Rejected trials restore the previous model and coordinates; AE/AEB
continue from the accepted model. Original model files and reference anchors
are not overwritten. `plog_decision.json` records the decision.
No trial is forced when the eligible reaction has unresolved sensitivity.
Each stage currently recomputes its reference flames; this is correct but costs
additional time. Jobs are not submitted by editing these settings.
