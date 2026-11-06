"""Reaction-rate sensitivities of ZND induction and exothermic lengths."""
from pathlib import Path

from mt.znd_functions import znd_sensitivity

ROOT = Path(__file__).resolve().parent
MECHANISM = ROOT.parent / 'ethylene_ZND' / 'inputs' / 'FFCM2.yaml'
END_TIME = 1.5e-5  # Seconds.
WORKERS = 8  # Set to 1 for sequential calculations.

# Pressure is in Pa. All reactions are perturbed individually by +/-1%.
# Each multiplier scales the complete reaction rate, including every PLOG entry.
result = znd_sensitivity(
    P0=1e5, T0=298., fuel='C2H4', oxidizer='O2:1, N2:3.76',
    phi=1., mechanism=str(MECHANISM),
    t_end=END_TIME, max_step=END_TIME / 4000,
    method='LSODA', rel_tol=1e-10, abs_tol=1e-14,
    perturbation=.01, workers=WORKERS, progress=True,
    plot=True, top_n=15,
    plot_file=str(ROOT / 'znd_sensitivity.pdf'),
)
# Two plots show the 15 largest-magnitude sensitivities for each length.
for path in result['plot_files'].values():
    print(f'Saved {path}')
