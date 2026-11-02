"""Ethylene-air IDT: time of maximum d[OH]/dt, with [OH] in kmol/m^3."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parents[2]
sys.path[:0] = [str(CODE / 'code'), str(CODE)]

from mt.ignition_cases import run_ignition_cases

MECHANISM = CODE / 'example/ethylene_ZND/inputs/FFCM2.yaml'
PRESSURES_ATM = [1, 10, 50]
TEMPERATURES_K = [1400, 1500, 1600]
PHIS = [0.6, 0.8, 1.0, 1.2, 1.4]
REACTOR_TYPE = 'constant-volume'  # Adiabatic; alternatively 'constant-pressure'.
END_TIME = 1e-2  # s; cases without resolved ignition are reported as failures.
OUTPUT = ROOT / 'ignition_delay_results.yaml'

problem = {
    'comparison_plots': True,
    'plot_yscales': {'ignition_delay_time': 'log'},
    'mechanism': MECHANISM,
    'output': OUTPUT,
    'mixture': {'fuel': 'C2H4', 'oxidizer': 'O2:1, N2:3.76'},
    'conditions': {'ignition': {
        'pressure_atm': PRESSURES_ATM,
        'temperature_K': TEMPERATURES_K,
        'phi': PHIS,
    }},
    'solver': {'end_time': END_TIME, 'marker_species': 'OH',
               'reactor_type': REACTOR_TYPE, 'rtol': 1e-9, 'atol': 1e-15},
}
result = run_ignition_cases(problem)
