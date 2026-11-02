"""Compare flame speeds and burned temperatures, separately for each pressure."""
from pathlib import Path
from mt.validation import Validation

ROOT = Path(__file__).resolve().parent
validation = Validation(
    reference=ROOT / 'FFCM2.yaml',
    models={'Tuned': ROOT / 'best_model.yaml'},
    fuel='C2H4', oxidizer='O2:1, N2:3.76',
    workers=8, output_dir=ROOT / 'results',
    save_results=False, save_logs=False,  # Only PNG plots are saved.
)

# Cartesian grid; pressure is in atm and temperature is in K.
result = validation.flame(
    conditions={'pressure_atm': [1, 10, 50], 'temperature_K': [298.], 'phi': [.6, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4]},
    solver={'width': .02, 'transport_model': 'multicomponent', 'soret': False,
            'slope': .015, 'curve': .03, 'timeout_seconds': 900},
)
result.print_summary()
for path in result.plot(quantities=['laminar_flame_speed', 'T_flame'],
                        x='phi', separate_by='pressure'):
    print(f'Saved {path}')
