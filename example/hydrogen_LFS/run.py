"""Kyuho hydrogen/argon case: jointly tune rate parameters and explicit orders."""
from pathlib import Path
import os
from mt.flame_settings import run_flame_tuning_case

ROOT = Path(__file__).resolve().parent
problem = {
    'model': ROOT/'inputs/H2_1_step_stoi.yaml',
    # Continue the saved Kyuho fit; the original template still anchors bounds.
    'initial_model': ROOT/'inputs/start_model.yaml',
    'reference': ROOT/'inputs/FFCM2H2.yaml',
    'output': ROOT/'output',
    'mixture': {'fuel': 'H2:1', 'oxidizer': 'O2:1, AR:3.752'},
    'conditions': {
        # Paired arrays reproduce the 15 Kyuho conditions, not a Cartesian product.
        'flame': {
            'mode': 'paired',
            'pressure_atm': [10, 12.0350, 14.0358, 16.0102, 17.9630, 19.8976,
                             21.8165, 23.7215, 25.6140, 27.4952, 29.3660,
                             31.2271, 33.0793, 34.9231, 36.7590],
            'temperature_K': [473, 504.9436, 533.0350, 558.2610, 581.2541,
                              602.4472, 622.1522, 640.6019, 657.9753, 674.4139,
                              690.0312, 704.9201, 719.1579, 732.8092, 745.9292],
            'phi': [.25]*15,
        },
    },
    # Same three equally weighted fitting targets as optimize_one_step_with_stoi.py.
    'objectives': {
        'laminar_flame_speed': {'loss': 'log_squared', 'weight': 1.},
        'flame_thickness': {'loss': 'log_squared', 'weight': 1.},
        'T_flame': {'loss': 'squared', 'scale': 25., 'weight': 1.},
    },
    'solver': {
        'width': .05, 'transport_model': 'mixture-averaged', 'soret': False,
        'ratio': 2., 'slope': .01, 'curve': .01, 'timeout_seconds': 900,
        'flame_speed_sensitivities': True,
        'full_flame_adjoints': True,
    },
    'reactions': {
        'artificial': [1], 'uncertainty_factors': {},
        # Actual orders; additive changes support zero and negative values.
        # Omit this mapping to keep all reaction orders fixed.
        'orders': {
            1: {'bounds': {'H2': (-2., 2.), 'O2': (-2., 2.)}},
            # Optional: 'sum_bounds': (1., 2.) inside the reaction's settings.
        },
        # Optional historical penalty is disabled: fit the three combustion targets only.
        # 'rate_regularization': {'weight': .01, 'scale_A': 2.302585092994046,
        #                         'scale_b': .5, 'scale_Ea': 2000.},
    },
}
tuning = {
    # Used only if the human-initialized mechanism does not yield all required flames.
    'exploration': {'samples': 128, 'batch_size': 8, 'seed': 42},
    'limits': {
        'a_ratio': (1e-3, 1e8), 'ea_delta': 60000., 'b_delta': 9.,
        # These reproduce Ea/Ea0=(0.3,5) and b=(-5,10) from Kyuho.
        # Ea uses the input YAML's cal/mol unit; no independent unit setting.
        'rate_bounds': {1: {'Ea': (4500., 75000.), 'b': (-5., 10.)}},
    },
    # Orders remain adjustable in every stage. Later A ranges are stage-relative.
    'stages': [
        {'parameters': 'A', 'a_ratio': (1e-3, 1e8)},
        {'parameters': 'AE', 'a_ratio': (.5, 2.), 'ea_delta': 60000.},
        {'parameters': 'AEB', 'a_ratio': (.5, 2.), 'ea_delta': 1195.03},
    ],
    'selection': {'top_reactions': 1},
    'plog': {'enabled': False, 'plog_free': False},
    'stopping': {'max_rounds': 20, 'inner_iterations': 15},
}
execution = {
    'comparison_plots': True,
    'plot_yscales': {'laminar_flame_speed': 'linear', 'flame_thickness': 'linear', 'T_flame': 'linear'},
    'backend': 'openmp' if os.environ.get('SLURM_JOB_ID') else 'process',
    'workers': int(os.environ.get('SLURM_CPUS_PER_TASK', '8')),
}
result = run_flame_tuning_case(problem, tuning, execution)
