"""Ignition delay and endpoint temperature using FFCM2."""
from mt import CombustionCalculator
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Reuse the FFCM2 reference already included in the repository.
MECHANISM = ROOT.parent / 'ethylene_ZND' / 'inputs' / 'FFCM2.yaml'
FUEL = 'C2H4'
OXIDIZER = 'O2:1, N2:3.76'  # Molar air composition.

# 3. Ignition: an existing ethylene_LFS_IDT condition (1 atm, 1500 K, phi = 1).
ignition = CombustionCalculator(
    mechanism=str(MECHANISM), fuel=FUEL, oxidizer=OXIDIZER,
    P=1., T=1500., phi=1.,
)
# Adiabatic constant-volume reactor. Delay is the time of maximum d[OH]/dt.
# Set marker_species='temperature' instead to measure maximum dT/dt.
delay = ignition.ignition_delay(
    end_time=.003, marker_species='OH', reactor_type='constant-volume',
    rtol=1e-9, atol=1e-15,
)
print(f'Ignition delay: {delay:.6e} s ({delay * 1e6:.3f} microseconds)')
# This is the temperature at 0.003 s, not a separately computed equilibrium value.
print(f'Final ignition temperature: {ignition.final_temperature:.3f} K')
