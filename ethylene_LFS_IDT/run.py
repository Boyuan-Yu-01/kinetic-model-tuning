"""Joint ethylene-air flame and ignition tuning; all editable settings are below."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
# Support cluster code/mt and a local installation with mt beside the examples.
sys.path[:0] = [str(ROOT.parent/'code'), str(ROOT.parent)]

problem = {
    'model': ROOT/'inputs/model_populated.yaml',  # Human-initialized reduced mechanism.
    'reference': ROOT/'inputs/FFCM2.yaml',        # Recomputed reference targets.
    'output': ROOT/'output',                    # Use a fresh directory for a new run.
    'conditions': {
        # Cartesian product: 3 pressures x 1 temperature x 5 phi values = 15 flames.
        'flame': {
            'pressure_atm': [1, 10, 50],
            'temperature_K': [298.],
            'phi': [.6, .8, 1., 1.2, 1.4],
        },
        # Separate Cartesian product: 3 x 3 x 3 = 27 ignition cases.
        'ignition': {
            'pressure_atm': [1, 10, 50],
            'temperature_K': [1400, 1500, 1600],
            'phi': [.6, 1., 1.4],
        },
    },
    'objectives': {
        # log_squared means ln(model/reference)^2 for positive speeds and delays.
        'laminar_flame_speed': {'loss': 'log_squared', 'weight': 1.},
        # squared temperature error is ((T_model - T_reference)/scale)^2; scale is K.
        'T_flame': {'loss': 'squared', 'scale': 25., 'weight': 1.},
        'ignition_delay_time': {'loss': 'log_squared', 'weight': 1.},
        'T_ignition': {'loss': 'squared', 'scale': 25., 'weight': 1.},
    },
    # Flame domain width is in metres; slope/curve control adaptive grid refinement.
    'solver': {
        'width': .02, 'transport_model': 'multicomponent', 'soret': False,
        'slope': .015, 'curve': .03, 'timeout_seconds': 900,  # Flame wall-clock limit only.
    },
    # Average each branch over its own cases, then combine with these weights.
    # Equal weights prevent 27 ignition cases from outweighing 15 flame cases by count.
    'branch_weights': {'flame': 1., 'ignition': 1.},
    'ignition_solver': {
        # Ignition time is the time of maximum OH concentration growth rate.
        'marker_species': 'OH', 'reactor_type': 'constant-volume',
        # Shared physical endpoint for reference and candidate; no ignition wall-clock timeout.
        # T_ignition is evaluated at this endpoint, not necessarily at equilibrium.
        'rtol': 1e-9, 'atol': 1e-15, 'end_time': 0.003,
    },
    'reactions': {
        'artificial': [1, 2, 3],  # One-based reaction numbers; no published uncertainty factors.
        # Borrowed-reaction uncertainty constraints retain their original reference anchors.
        'uncertainty_factors': {4:1.15, 5:1.6, 6:1.6, 7:1.2, 8:1.4, 9:3.,
            10:3., 11:5., 12:3.2, 13:1.5, 14:3., 15:10., 16:1.2, 17:1.5},
    },
}

tuning = {
    # Only used if the starting model fails any required flame/ignition evaluation.
    # Samples vary artificial-reaction A coordinates; batch_size counts parameter sets.
    # Each set requires 42 case solves. seed makes the random sequence reproducible.
    'exploration': {'samples': 128, 'batch_size': 8, 'seed': 42},
    # Global bounds: A multiplier, additive Ea allowance in YAML units, and additive b allowance.
    'limits': {'a_ratio': (1e-6, 1e8), 'ea_delta': 60000., 'b_delta': .2},
    # Release A, then A/Ea, then A/Ea/b. Stage bounds also respect global constraints.
    # Later-stage A ratios and Ea changes are measured relative to the stage-entry state.
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
    # Outer sensitivity-selection rounds and inner parameter-update iterations.
    # Reaching a cap or stalling is not reported as convergence.
    'stopping': {'max_rounds': 20, 'inner_iterations': 15},
}

execution = {
    # Display scales only: these do not change objective transformations or weights.
    'plot_yscales': {'laminar_flame_speed': 'linear', 'T_flame': 'linear', 'ignition_delay_time': 'log', 'T_ignition': 'linear'},
    'comparison_plots': True,  # False skips comparison figures; numerical results remain saved.
    'backend': 'openmp' if os.environ.get('SLURM_JOB_ID') else 'process',
    # Use the Slurm allocation when present; otherwise use 8 workers.
    'workers': int(os.environ.get('SLURM_CPUS_PER_TASK', '8')),
}

# Configuration conversion, parallel execution, tuning and reporting are implemented in mt/.
from mt.flame_ignition import run_flame_ignition_case
result = run_flame_ignition_case(problem, tuning, execution)
