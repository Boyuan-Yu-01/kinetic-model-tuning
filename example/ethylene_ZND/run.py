"""Editable ethylene experiment. Reaction numbers are one-based; Ea uses YAML units."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent

problem = {
    'model': ROOT/'inputs/model_populated.yaml',
    'reference': ROOT/'inputs/FFCM2.yaml',
    'output': ROOT/'output',
    'mixture': {'fuel': 'C2H4', 'oxidizer': 'O2:1, N2:3.76'},
    'conditions': {'znd': {
        'pressure_bar': [1,1,1,5,5,5,20,20,20],
        'temperature_K': 298.,
        'phi': [.5,1.,2.]*3,
        'end_time_s': [1e-4,8e-6,9e-6,5e-5,1e-6,2e-6,3e-5,1e-6,2e-6],
    }},
    'objectives': {
        'T_ZND': {'loss': 'squared', 'scale': 25.},
        'ind_len_ZND': {'loss': 'log_squared'},
        'exo_len_ZND': {'loss': 'log_squared'},
    },
    'reactions': {
        'artificial': [1,2,3],
        # Original reference anchors remain fixed across starts and stages.
        'uncertainty_factors': {4:1.15,5:1.6,6:1.6,7:1.2,8:1.4,9:3.,
            10:3.,11:5.,12:3.2,13:1.5,14:3.,15:10.,16:1.2,17:1.5},
    },
}

tuning = {
    'exploration': {
        'samples': 2000, 'keep_candidates': 20, 'batch_size': 32, 'seed': 42,
        'a_ratio': {1:(1e-6,1e5), 2:(1e-3,1e8), 3:(1e-5,1e5)},
    },
    'multistart': {'starts': 5, 'selection': 'best'},
    'selection': {'top_reactions': 5},
    'limits': {'a_ratio': (1e-6,1e8), 'ea_delta': 60000., 'b_delta': .2},
    'stages': [
        {'parameters':'A', 'a_ratio':(1e-6,1e8)},
        {'parameters':'AE', 'a_ratio':(.5,2.), 'ea_delta':60000.},
        {'parameters':'AEB', 'a_ratio':(.5,2.), 'ea_delta':1195.03},
    ],
    'plog': {
        # New insertion needs independent pressure-entry A adjustments.
        'enabled': True,
        'reactions': [2],
        'pressures_bar': [1.0, 50.0, 100.0],
        'direction': 'either',
        'plog_free': True,
    },
    'stopping': {
        'max_rounds': 20, 'threshold': .01,
        'stagnation_rounds': 3, 'stagnation_relative_tolerance': 1e-5,
    },
    'advanced': {},  # Optional overrides of mt.workflow_defaults.NUMERICS.
}

execution = {
    'plot_yscales': {'T_profile': 'linear'},
    'backend': 'openmp' if os.environ.get('SLURM_JOB_ID') else 'process',
    'workers': int(os.environ.get('SLURM_CPUS_PER_TASK', '8')),
    'resume': False,
    'save_sensitivity_plots': True,
    'reference_cache': ROOT/'diagnostics/reference_cache',
    'fit_targets': {'final_temperature_percent':5., 'induction_percent':10., 'exothermic_percent':10.},
}


from mt.multistart_znd import run_tuning
result = run_tuning(problem, tuning, execution)
