"""Compare nine ZND cases with FFCM2; blue solid reference and dashed-circle model."""
from pathlib import Path
from mt.validation import Validation

ROOT = Path(__file__).resolve().parent
validation = Validation(
    reference=ROOT / 'FFCM2.yaml', models={'Tuned': ROOT / 'best_model.yaml'},
    fuel='C2H4', oxidizer='O2:1, N2:3.76', workers=8,
    output_dir=ROOT / 'temperature_profiles', save_results=False, save_logs=False,
)
# ZND arrays are paired, not a Cartesian grid. A scalar temperature applies to all cases.
result = validation.znd(
    conditions={
        'pressure_bar': [1, 1, 1, 5, 5, 5, 20, 20, 20],
        'temperature_K': 298., 'phi': [.5, 1., 2.] * 3,
        'end_time_s': [1e-4, 8e-6, 9e-6, 5e-5, 1e-6, 2e-6, 3e-5, 1e-6, 2e-6],
    },
    solver={'method': 'LSODA', 'rel_tol': 1e-10, 'abs_tol': 1e-14, 'steps': 4000},
)
# Includes induction/exothermic lengths, signed percentage errors and |ln(model/reference)|.
result.print_summary()
for path in result.plot(
    quantities=['T_profile'], x='distance_mm', separate_by='case',
    styles={'Reference': {'color': 'blue', 'linestyle': '-'},
            'Tuned': {'color': 'blue', 'linestyle': '--', 'marker': 'o',
                      'markerfacecolor': 'none'}},
):
    print(f'Saved {path}')
