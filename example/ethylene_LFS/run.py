"""Ethylene-air flame tuning. Reaction numbers are one-based; Ea uses YAML units."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parents[1]/'code'), str(ROOT.parents[1])]

problem = {
    'model': ROOT/'inputs/model_populated.yaml',
    'reference': ROOT/'inputs/FFCM2.yaml',
    'output': ROOT/'output',
    # All combinations: 3 pressures x 1 temperature x 5 equivalence ratios.
    'conditions': {'flame': {
        'pressure_atm': [1, 10, 50],
        'temperature_K': [298.],
        'phi': [.6, .8, 1., 1.2, 1.4],
    }},
    'objectives': {
        'laminar_flame_speed': {'loss': 'log_squared', 'weight': 1.},
        'T_flame': {'loss': 'squared', 'scale': 25., 'weight': 1.},
    },
    'solver': {
        'width': .02, 'transport_model': 'multicomponent', 'soret': False,
        'slope': .015, 'curve': .03, 'timeout_seconds': 900,
    },
    'reactions': {
        'artificial': [1, 2, 3],
        # Fixed original reference anchors, preserved through every stage.
        'uncertainty_factors': {4:1.15, 5:1.6, 6:1.6, 7:1.2, 8:1.4, 9:3.,
            10:3., 11:5., 12:3.2, 13:1.5, 14:3., 15:10., 16:1.2, 17:1.5},
    },
}

tuning = {
    # Used if the original model fails to produce valid flames in all cases.
    'exploration': {'samples': 128, 'batch_size': 8, 'seed': 42},
    'limits': {'a_ratio': (1e-6, 1e8), 'ea_delta': 60000., 'b_delta': .2},
    'stages': [
        {'parameters': 'A', 'a_ratio': (1e-6, 1e8)},
        {'parameters': 'AE', 'a_ratio': (.5, 2.), 'ea_delta': 60000.},
        {'parameters': 'AEB', 'a_ratio': (.5, 2.), 'ea_delta': 1195.03},
    ],
    'plog': {
        # New insertion needs independent pressure-entry A adjustments.
        'enabled': True,
        'reactions': [2],
        'pressures_bar': [1.0, 50.0, 100.0],
        'direction': 'either',
        'plog_free': True,
    },
    'stopping': {'max_rounds': 20, 'inner_iterations': 15},
}

execution = {
    'plot_yscales': {'laminar_flame_speed': 'linear', 'T_flame': 'linear'},
    "comparison_plots": True,  # False skips comparison.png; numerical results are still saved.
    'backend': 'openmp' if os.environ.get('SLURM_JOB_ID') else 'process',
    'workers': int(os.environ.get('SLURM_CPUS_PER_TASK', '8')),
}

from mt.flame_settings import run_flame_tuning_case
result = run_flame_tuning_case(problem, tuning, execution)
