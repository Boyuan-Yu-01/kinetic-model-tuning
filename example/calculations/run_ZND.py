"""ZND detonation and temperature profile using FFCM2."""
from mt.znd_functions import calculate_znd, plot_temperature
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Reuse the FFCM2 reference already included in the repository.
MECHANISM = ROOT.parent / 'ethylene_ZND' / 'inputs' / 'FFCM2.yaml'
FUEL = 'C2H4'
OXIDIZER = 'O2:1, N2:3.76'  # Molar air composition.

# 1. ZND: an existing ethylene_ZND condition (1 bar, 298 K, phi = 1).
# Unlike CombustionCalculator, calculate_znd takes pressure in PASCALS.
# Use LSODA and the tuning tolerances; file_name=None avoids a large text export.
end_time = 8e-6  # Seconds; the integration horizon for this reference case.
znd = calculate_znd(
    P0=1e5, T0=298., fuel=FUEL, oxidizer=OXIDIZER, phi=1.,
    mechanism=str(MECHANISM), file_name=None,
    t_end=end_time, max_step=end_time / 4000,
    rel_tol=1e-10, abs_tol=1e-14, method='LSODA',
)
print(f'ZND final temperature: {znd["T"][-1]:.3f} K')
print(f'ZND induction length: {znd["ind_len_ZND"]:.6e} m')
print(f'ZND exothermic length: {znd["exo_len_ZND"]:.6e} m')
# Distance is converted to mm by this plotting function.
plot_temperature(znd, labels=['FFCM2'], file_name=str(ROOT / 'znd_temperature.png'),
                 title='Ethylene–air: 1 bar, 298 K, phi=1')

