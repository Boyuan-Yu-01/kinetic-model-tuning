"""Compare ignition delay and final temperature at ten inverse-temperature points."""
import numpy as np
from pathlib import Path
from mt.validation import Validation

ROOT = Path(__file__).resolve().parent
validation = Validation(
    reference=ROOT / 'FFCM2.yaml',
    models={'Tuned': ROOT / 'best_model.yaml', 'Untuned': ROOT / 'mechanism.yaml'},
    fuel='CH4', oxidizer='O2:1, N2:3.76',
    workers=8, output_dir=ROOT / 'results',
    save_results=False, save_logs=False,  # Only PNG plots are saved.
)

# Ten evenly spaced 1000/T points; each pressure/temperature/phi combination is solved.
temperatures = (1000. / np.linspace(1000. / 1700., 1000. / 1600., 10)).tolist()
result = validation.ignition(
    conditions={'pressure_atm': [1, 10, 50], 'temperature_K': temperatures,
                'phi': [.6, 1., 1.3]},
    # Maximum dT/dt, with one fixed integration end time in seconds.
    solver={'marker_species': 'temperature', 'reactor_type': 'constant-volume',
            'end_time': .003, 'rtol': 1e-11, 'atol': 1e-17},
)
result.print_summary()
for path in result.plot(quantities=['ignition_delay_time', 'T_ignition'],
                        x='1000/T', separate_by='pressure',
                        yscales={'ignition_delay_time': 'log', 'T_ignition': 'linear'}):
    print(f'Saved {path}')
