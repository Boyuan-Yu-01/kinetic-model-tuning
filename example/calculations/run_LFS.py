"""Laminar flame speed, temperature, and thicknesses using FFCM2."""
from mt import CombustionCalculator
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Reuse the FFCM2 reference already included in the repository.
MECHANISM = ROOT.parent / 'ethylene_ZND' / 'inputs' / 'FFCM2.yaml'
FUEL = 'C2H4'
OXIDIZER = 'O2:1, N2:3.76'  # Molar air composition.

# 2. Laminar flame: an existing ethylene_LFS_IDT condition.
# CombustionCalculator takes pressure in ATM, temperature in K.
flame = CombustionCalculator(
    mechanism=str(MECHANISM), fuel=FUEL, oxidizer=OXIDIZER,
    P=1., T=298., phi=1.,
)
# Adiabatic freely propagating flame; multicomponent transport, Soret off by default.
# Refinement settings match the existing flame example.
speed, burned_temperature = flame.solve_flame(
    width=.02, transport_model='multicomponent',
    refine_ratio=3., refine_slope=.015, refine_curve=.03, loglevel=0,
)
print(f'Laminar flame speed: {speed * 100:.4f} cm/s')  # Returned speed is m/s.
print(f'Burned flame temperature: {burned_temperature:.3f} K')
# These reuse the solved flame rather than solving it again.
print(f'Flame thickness: {flame.flame_thickness() * 1e6:.3f} micrometres')
print(f'Diffusion thickness: {flame.diffusion_thickness() * 1e6:.3f} micrometres')

