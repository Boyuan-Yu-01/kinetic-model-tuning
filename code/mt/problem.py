"""Python interaction layer for reduced-mechanism tuning."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import hashlib
import warnings
import numpy as np
import yaml
from .config import RateParameters, OrderParameters, TrendPenalty, SolverSettings, SquaredError, RelativeSquaredError, LogSquaredError, FlameSettings, IgnitionSettings
from .mechanism import rates, parameter_dict, order_dict, modify_orders, Balance
from .search import search, worker_count, mpi_active, pool_context, execution_backend
from .order_space import OrderSpace
from .candidate_file import candidate_file
from .history import HistoryWriter

# Each solver family has independent cases; outputs use SI units.
ZND_KEYS = ('T_ZND', 'ind_len_ZND', 'exo_len_ZND')
FLAME_KEYS = ('laminar_flame_speed', 'T_flame', 'flame_thickness', 'diffusion_thickness')
IGNITION_KEYS = ('ignition_delay_time', 'T_ignition')
KEYS = ZND_KEYS + FLAME_KEYS + IGNITION_KEYS


def validate_outputs(rows, count, keys=ZND_KEYS):
    """Return a case-by-objective array, requiring positive finite selected outputs."""
    try:
        data = np.asarray([[row[k] for k in keys] for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Missing or invalid outputs for {keys}.') from error
    if data.shape != (count, len(keys)) or not np.all(np.isfinite(data)) or np.any(data <= 0):
        raise ValueError(f'All cases require positive finite outputs for {keys}.')
    return data


class TuningProblem:
    """Configure, evaluate, and optimize a reduced mechanism against a reference mechanism."""
    def __init__(self, model, reference, fuel, oxidizer, conditions=None, solver=None,
                 *, znd_conditions=None, znd_solver=None, flame_conditions=None, flame_solver=None,
                 ignition_conditions=None, ignition_solver=None):
        """Load template/reference YAML snapshots and initialize conditions and default settings."""
        self.model_path = Path(model).resolve()
        self.reference_path = Path(reference).resolve()
        self.scheme = yaml.safe_load(self.model_path.read_text())
        self.reference_scheme = yaml.safe_load(self.reference_path.read_text())
        if conditions is not None and znd_conditions is not None:
            raise ValueError('Use znd_conditions or its legacy alias conditions, not both.')
        self.fuel, self.oxidizer = fuel, oxidizer
        self.conditions = znd_conditions if znd_conditions is not None else conditions
        if solver is not None and znd_solver is not None:
            raise ValueError('Use znd_solver or its legacy alias solver, not both.')
        self.znd_solver = znd_solver if znd_solver is not None else (solver or SolverSettings())
        self.flame_conditions = flame_conditions
        self.flame_solver = flame_solver or FlameSettings()
        self.ignition_conditions = ignition_conditions
        self.ignition_solver = ignition_solver or IgnitionSettings()
        self.original_rates = parameter_dict(self.scheme)
        self.original_orders = order_dict(self.scheme)
        self._rates = RateParameters()
        self._balances = []
        self._orders = []
        self._order_spaces = []
        self._objectives = {}
        self._trend = None
        if self.conditions is not None:
            self._objectives.update(T_ZND=SquaredError(100), ind_len_ZND=LogSquaredError(), exo_len_ZND=LogSquaredError())
        if flame_conditions is not None:
            self._objectives.update(laminar_flame_speed=LogSquaredError(), T_flame=SquaredError(100),
                                    flame_thickness=LogSquaredError(), diffusion_thickness=LogSquaredError())
        if ignition_conditions is not None:
            self._objectives.update(ignition_delay_time=LogSquaredError(), T_ignition=SquaredError(100))
        if not self._objectives:
            raise ValueError('Provide at least one set of ZND, flame, or ignition conditions.')
        self._reference_results = None
        self._reference_signature = None

    @property
    def solver(self):
        """Compatibility alias for the ZND solver settings."""
        return self.znd_solver

    @solver.setter
    def solver(self, value):
        """Update ZND settings through the legacy attribute name."""
        self.znd_solver = value

    @property
    def znd_conditions(self):
        """Expose ZND conditions under the explicit solver-family name."""
        return self.conditions

    @znd_conditions.setter
    def znd_conditions(self, value):
        """Replace ZND cases; reference caching checks the updated values."""
        self.conditions = value

    def parameters(self, rates=None, stoichiometry=None, orders=None):
        """Set rate, stoichiometric, and explicit-order tuning definitions; return this problem."""
        if rates is not None:
            self._rates = deepcopy(rates)
        if stoichiometry is not None:
            specs = stoichiometry if isinstance(stoichiometry, list) else [stoichiometry]
            if len({s.reaction for s in specs}) != len(specs):
                raise ValueError('Each tunable reaction needs one stoichiometric definition.')
            self._balances = [Balance(self.scheme, deepcopy(s)) for s in specs]
        if orders is not None:
            specs = orders if isinstance(orders, list) else [orders]
            if any(not isinstance(s, OrderParameters) for s in specs):
                raise TypeError('Use OrderParameters for reaction-order tuning.')
            if len({s.reaction for s in specs}) != len(specs):
                raise ValueError('Each tunable reaction needs one order definition.')
            for spec in specs:
                original = self.original_orders.get(spec.reaction, {})
                if not set(spec.ratios) <= set(original):
                    raise ValueError(f'Reaction {spec.reaction}: order species must already appear in its orders mapping.')
                for species, bounds in spec.ratios.items():
                    base = original[species]
                    if not np.isfinite(base) or base == 0:
                        raise ValueError(f'Reaction {spec.reaction}, {species}: multiplier tuning requires a finite nonzero order.')
                    with np.errstate(over='ignore', under='ignore'):
                        endpoints = base * np.asarray(bounds, dtype=float)
                    if not np.all(np.isfinite(endpoints)):
                        raise ValueError('Order multiplier bounds must produce finite orders.')
            copied = deepcopy(specs)
            spaces = [OrderSpace(s, self.original_orders[s.reaction]) for s in copied]
            self._orders, self._order_spaces = copied, spaces
        return self

    def objectives(self, **objectives):
        """Set weighted ZND, flame, and ignition objectives; return this problem for chaining."""
        if 'T_final' in objectives:
            raise ValueError('T_final has been renamed to T_ZND.')
        if not objectives or not set(objectives) <= set(KEYS):
            raise ValueError(f'Choose objectives from {KEYS}.')
        if any(not isinstance(o, (SquaredError, RelativeSquaredError, LogSquaredError)) for o in objectives.values()):
            raise TypeError('Use SquaredError, RelativeSquaredError or LogSquaredError.')
        if sum(o.weight for o in objectives.values()) <= 0:
            raise ValueError('At least one objective weight must be positive.')
        self._objectives = deepcopy(objectives)
        return self

    def trend(self, penalty=None):
        """Configure a residual-slope penalty; pass None to disable it."""
        if penalty is not None and not isinstance(penalty, TrendPenalty):
            raise TypeError('Use TrendPenalty or None.')
        self._trend_coordinates(penalty)
        self._trend = deepcopy(penalty)
        return self

    def _trend_coordinates(self, penalty):
        """Validate selected objectives and build coordinates from their own case sets."""
        if penalty is None:
            return {}
        if not set(penalty.objectives) <= set(KEYS):
            raise ValueError('Unknown trend objective.')
        if penalty.weight == 0:
            return {}
        active = {name for name, objective in self._objectives.items() if objective.weight > 0}
        if not set(penalty.objectives) <= active:
            raise ValueError('Each trend objective must also have a positive fit-objective weight.')
        coordinates = {}
        for _, cases, keys in self._groups():
            selected = set(keys) & set(penalty.objectives)
            if selected:
                values = penalty.coordinates(cases.P, cases.weights)
                coordinates.update({key: values for key in selected})
        return coordinates

    @property
    def bounds(self):
        """Return bounds for rates, stoichiometry, then configured order coordinates."""
        n = len(self.original_rates['A'])
        # Order multipliers use log coordinates for positive ranges, linear otherwise.
        result = [tuple(np.log10(self._rates.A_ratio))] * n + [self._rates.b] * n
        result += [tuple(np.log10(self._rates.Ea_ratio))] * n
        result += [c.effective_bounds() for b in self._balances for c in b.spec.independent.values()]
        result += [bound for space in self._order_spaces for bound in space.bounds]
        return np.asarray(result, dtype=float)

    def decode(self, vector):
        """Decode bounded coordinates into rates, stoichiometry, and optional explicit orders."""
        vector = np.asarray(vector, dtype=float)
        bounds = self.bounds
        if vector.shape != (len(bounds),) or not np.all(np.isfinite(vector)):
            raise ValueError('Parameter vector has invalid shape or values.')
        if np.any(vector < bounds[:, 0]) or np.any(vector > bounds[:, 1]):
            raise ValueError('Parameter vector is outside bounds.')
        n = len(self.original_rates['A'])
        params = {'rates': {
            'A': (np.array(self.original_rates['A']) * 10. ** vector[:n]).tolist(),
            'b': vector[n:2*n].tolist(),
            'Ea': (np.array(self.original_rates['Ea']) * 10. ** vector[2*n:3*n]).tolist(),
        }, 'stoichiometry': {}}
        offset = 3*n
        for balance in self._balances:
            names = list(balance.spec.independent)
            params['stoichiometry'][balance.spec.reaction] = dict(zip(names, map(float, vector[offset:offset+len(names)])))
            offset += len(names)
        if self._orders:
            params['orders'] = {}
            for space in self._order_spaces:
                count = len(space.bounds)
                params['orders'][space.spec.reaction] = space.decode(vector[offset:offset + count])
                offset += count
        return params

    def mechanism(self, parameters):
        """Return a modified mechanism dictionary while preserving the original template."""
        scheme = deepcopy(self.scheme)
        expected = {'rates', 'stoichiometry'} | ({'orders'} if self._orders else set())
        if set(parameters) != expected or set(parameters['rates']) != {'A', 'b', 'Ea'}:
            raise ValueError(f'Use separate parameter dictionaries: {sorted(expected)}.')
        entries = rates(scheme)
        for key in ('A', 'b', 'Ea'):
            values = np.asarray(parameters['rates'][key], dtype=float)
            if values.shape != (len(entries),) or not np.all(np.isfinite(values)):
                raise ValueError('Rate arrays must be finite and aligned.')
            for rate, value in zip(entries, values):
                rate[key] = float(value)
        if set(parameters['stoichiometry']) != {b.spec.reaction for b in self._balances}:
            raise ValueError('Stoichiometric reaction keys do not match configuration.')
        # Derive balanced reaction equations after updating the rate entries.
        for balance in self._balances:
            balance.apply(scheme, parameters['stoichiometry'][balance.spec.reaction])
        if self._orders:
            orders = parameters['orders']
            if set(orders) != {s.reaction for s in self._orders}:
                raise ValueError('Order reaction keys do not match configuration.')
            for space in self._order_spaces:
                space.validate(orders[space.spec.reaction])
            modify_orders(scheme, orders)
        return scheme

    def _groups(self, conditions=None, flame_conditions=None, ignition_conditions=None):
        """Yield each enabled solver family, its cases, and positive-weight objective keys."""
        cases = self.conditions if conditions is None else conditions
        flames = self.flame_conditions if flame_conditions is None else flame_conditions
        ignition = self.ignition_conditions if ignition_conditions is None else ignition_conditions
        for family, configuration, allowed in (('znd', cases, ZND_KEYS), ('flame', flames, FLAME_KEYS),
                                                ('ignition', ignition, IGNITION_KEYS)):
            keys = tuple(k for k, objective in self._objectives.items() if k in allowed and objective.weight > 0)
            if keys:
                if configuration is None:
                    raise ValueError(f'Provide {family} conditions for the selected objectives.')
                yield family, configuration, keys

    def _simulate(self, scheme, conditions=None, flame_conditions=None, ignition_conditions=None):
        """Evaluate enabled families, returning {family: [case outputs]} in SI units."""
        groups = list(self._groups(conditions, flame_conditions, ignition_conditions))
        results = {}
        # Reuse a private process-local file without accumulating Cantera YAML cache entries.
        with candidate_file(scheme) as mechanism:
            for family, cases, keys in groups:
                rows = []
                if family == 'znd':
                    from .znd_functions import calculate_znd
                    for P, T, phi, end in zip(cases.P, cases.T, cases.phi, cases.t_end):
                        result = calculate_znd(float(P), float(T), self.fuel, self.oxidizer,
                                               float(phi), str(mechanism), None, t_end=float(end), **asdict(self.znd_solver))
                        row = {k: float(result['T'][-1] if k == 'T_ZND' else result[k]) for k in keys}
                        validate_outputs([row], 1, keys)
                        rows.append(row)
                elif family == 'flame':
                    import cantera as ct
                    from .combustion_calculator import CombustionCalculator
                    for P, T, phi in zip(cases.P, cases.T, cases.phi):
                        # The standalone calculator takes atm; the tuning API takes Pa.
                        calculator = CombustionCalculator(str(mechanism), self.fuel, self.oxidizer,
                                                         float(phi), float(P) / ct.one_atm, T=float(T))
                        speed, temperature = calculator.solve_flame(**asdict(self.flame_solver))
                        values = dict(laminar_flame_speed=speed, T_flame=temperature)
                        # Thickness methods reuse the already solved flame.
                        if 'flame_thickness' in keys:
                            values['flame_thickness'] = calculator.flame_thickness()
                        if 'diffusion_thickness' in keys:
                            values['diffusion_thickness'] = calculator.diffusion_thickness()
                        row = {k: float(values[k]) for k in keys}
                        validate_outputs([row], 1, keys)
                        rows.append(row)
                else:
                    import cantera as ct
                    from .combustion_calculator import CombustionCalculator
                    for P, T, phi, end in zip(cases.P, cases.T, cases.phi, cases.t_end):
                        calculator = CombustionCalculator(str(mechanism), self.fuel, self.oxidizer,
                                                         float(phi), float(P) / ct.one_atm, T=float(T))
                        delay = calculator.ignition_delay(end_time=float(end), **asdict(self.ignition_solver))
                        values = dict(ignition_delay_time=delay, T_ignition=calculator.final_temperature)
                        row = {k: float(values[k]) for k in keys}
                        validate_outputs([row], 1, keys)
                        rows.append(row)
                results[family] = rows
        return results

    def evaluate_reference(self):
        """Cache reference results for matching mechanisms, cases, solvers, and active outputs."""
        active = {family: keys for family, _, keys in self._groups()}
        signature = json.dumps({'scheme': self.reference_scheme, 'conditions': self._condition_dict(),
                                'flame_conditions': self._condition_dict('flame'),
                                'ignition_conditions': self._condition_dict('ignition'), 'active': active,
                                'fuel': self.fuel, 'oxidizer': self.oxidizer, 'znd_solver': asdict(self.znd_solver),
                                'flame_solver': asdict(self.flame_solver),
                                'ignition_solver': asdict(self.ignition_solver)}, sort_keys=True)
        if self._reference_results is None or signature != self._reference_signature:
            self._reference_results = self._simulate(self.reference_scheme)
            self._reference_signature = signature
        return deepcopy(self._reference_results)

    def evaluate_baseline(self):
        """Evaluate the original template; no tuning bounds or edits are applied."""
        return self._simulate(self.scheme)

    def evaluate(self, parameters, conditions=None, *, znd_conditions=None, flame_conditions=None, ignition_conditions=None):
        """Evaluate a structured parameter dictionary, optionally on validation cases."""
        if conditions is not None and znd_conditions is not None:
            raise ValueError('Use znd_conditions or conditions, not both.')
        cases = znd_conditions if znd_conditions is not None else conditions
        return self._simulate(self.mechanism(parameters), cases, flame_conditions, ignition_conditions)

    def score(self, parameters, results, reference_results):
        """Return fit, rate/order penalties, optional residual trends, and total loss."""
        coordinates = self._trend_coordinates(self._trend)
        slopes = {}
        # Average each objective over its own family, then normalize objective weights.
        weights = np.array([o.weight for o in self._objectives.values()], dtype=float)
        weights /= weights.max()
        weights /= weights.sum()
        contributions = {key: 0. for key in self._objectives}
        for family, cases, keys in self._groups():
            if family not in results or family not in reference_results:
                raise ValueError(f'Missing {family} results.')
            actual = validate_outputs(results[family], len(cases.P), keys)
            reference = validate_outputs(reference_results[family], len(cases.P), keys)
            for index, key in enumerate(keys):
                residual = self._objectives[key].residual(actual[:, index], reference[:, index])
                contributions[key] = float(np.sum(cases.weights * residual**2))
                if key in coordinates:
                    x, case_weights, variance = coordinates[key]
                    centered = residual - np.sum(case_weights * residual)
                    slopes[key] = float(np.sum(case_weights * x * centered) / variance)
        fit = float(sum(w * contributions[k] for w, k in zip(weights, self._objectives)))
        p, p0, reg = parameters['rates'], self.original_rates, self._rates.regularization
        # Rate penalties and optional order-sum penalties remain separate from the fit.
        penalty = 0.
        if reg.weight:
            penalty = float(np.mean(
                ((np.log(p['A']) - np.log(p0['A'])) / reg.scale_A)**2
                + ((np.asarray(p['b']) - p0['b']) / reg.scale_b)**2
                + ((np.asarray(p['Ea']) - p0['Ea']) / reg.scale_Ea)**2))
        order_sums, order_penalties = {}, {}
        for space in self._order_spaces:
            number = space.spec.reaction
            order_sums[number], order_penalties[number] = space.penalty(parameters['orders'][number])
        order_penalty = float(sum(order_penalties.values()))
        trend_loss = float(np.mean(np.square(list(slopes.values())))) if slopes else 0.
        weighted_trend = self._trend.weight * trend_loss if slopes else 0.
        total = fit + reg.weight * penalty + order_penalty + weighted_trend
        if not np.isfinite(total):
            raise ValueError('Nonfinite loss.')
        return {'loss': total, 'fit_loss': fit, 'regularization': penalty,
                'weighted_regularization': reg.weight * penalty, 'objectives': contributions,
                'order_sums': order_sums, 'order_sum_penalties': order_penalties,
                'order_sum_penalty': order_penalty, 'trend_slopes': slopes,
                'trend_loss': trend_loss, 'weighted_trend_loss': weighted_trend}

    def sample_candidates(self, count=100, seed=42, workers='auto', backend='process'):
        """Evaluate random candidates without local refinement or output files.

        Set backend="openmp" for the native scheduler with isolated solver workers.
        Returned records are in sample order. Failed candidates retain their
        error and infinite loss. Reference results are cached in this object.
        """
        if not mpi_active() and worker_count(workers) > 1:
            from .launcher import launch
            return launch(self, '_sample_candidates', dict(count=count, seed=seed, workers=workers, backend=backend))
        return self._sample_candidates(count, seed, workers, backend)

    def _sample_candidates(self, count=100, seed=42, workers='auto', backend='process'):
        """Evaluate random bounded points and return records in sampling order."""
        from .search import limit_threads, completed, sample_task
        execution_backend(backend)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('count must be a positive integer.')
        n_workers = worker_count(workers)
        self._trend_coordinates(self._trend)
        reference = self.evaluate_reference()
        objective = CandidateObjective(self, reference)
        bounds = self.bounds
        unit = np.random.default_rng(seed).random((count, len(bounds)))
        points = np.clip(bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0]), bounds[:, 0], bounds[:, 1])
        limit_threads()
        records = [None] * count
        with pool_context(n_workers, backend) as pool:
            jobs = ((objective, i, point) for i, point in enumerate(points))
            for i, record in completed(pool, sample_task, jobs, n_workers):
                records[i] = record
        return records

    def _condition_dict(self, family='znd'):
        """Serialize the selected family's conditions, or None when unconfigured."""
        conditions = {'znd': self.conditions, 'flame': self.flame_conditions,
                      'ignition': self.ignition_conditions}[family]
        if conditions is None:
            return None
        keys = ('P', 'T', 'phi', 't_end', 'weights') if family != 'flame' else ('P', 'T', 'phi', 'weights')
        return {key: getattr(conditions, key).tolist() for key in keys}

    def optimize(self, *, samples=100, starts=5, neighbors=5, min_distance=.2,
                 max_evaluations=1000, seed=42, workers='auto', output_dir='runs/experiment', progress_every=100,
                 sample_batch_size=1024, max_in_memory_valid=100000,
                 worker_batch_size=16, history_mode='compact', resume=False, backend='process'):
        """Run screening/refinement with backend="process" or "openmp" for all families."""
        settings = dict(samples=samples, starts=starts, neighbors=neighbors,
                        min_distance=min_distance, max_evaluations=max_evaluations,
                        seed=seed, workers=workers, output_dir=output_dir,
                        progress_every=progress_every, sample_batch_size=sample_batch_size,
                        max_in_memory_valid=max_in_memory_valid, worker_batch_size=worker_batch_size,
                        history_mode=history_mode, resume=resume, backend=backend)
        if not mpi_active() and worker_count(workers) > 1:
            from .launcher import launch
            return launch(self, '_optimize', settings)
        return self._optimize(**settings)

    def _optimize(self, *, samples=100, starts=5, neighbors=5, min_distance=.2,
                  max_evaluations=1000, seed=42, workers='auto', output_dir='runs/experiment', progress_every=100,
                 sample_batch_size=1024, max_in_memory_valid=100000,
                 worker_batch_size=16, history_mode='compact', resume=False, backend='process'):
        """Manage reference evaluation, search, progress, and files; return a TuningResult."""
        if not isinstance(progress_every, int) or progress_every < 1:
            raise ValueError('progress_every must be a positive integer.')
        for name, value in [('sample_batch_size', sample_batch_size), ('max_in_memory_valid', max_in_memory_valid),
                            ('worker_batch_size', worker_batch_size)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        execution_backend(backend)
        count = worker_count(workers)
        list(self._groups())  # Reject missing case sets before creating output files.
        self._trend_coordinates(self._trend)
        output = Path(output_dir).resolve()
        if not isinstance(resume, bool):
            raise ValueError('resume must be True or False.')
        from .restart import run_lock
        if resume and not output.is_dir():
            raise FileNotFoundError(f'Restart directory does not exist: {output}')
        output.mkdir(parents=True, exist_ok=resume)
        search_settings = dict(samples=samples, starts=starts, neighbors=neighbors, min_distance=min_distance,
                               max_evaluations=max_evaluations, seed=seed, workers=count,
                               sample_batch_size=sample_batch_size, max_in_memory_valid=max_in_memory_valid,
                               worker_batch_size=worker_batch_size, backend=backend)
        with run_lock(output):
            return self._run_optimization(output, search_settings, progress_every, history_mode, resume)

    def _run_optimization(self, output, search_settings, progress_every, history_mode, resume):
        """Own one run's locked files and synchronize search checkpoints with its history."""
        from time import monotonic
        from .checkpoint import RestartStore
        from .restart import (atomic_yaml, check_compatible, verify_mechanisms,
                              write_restart_script, import_legacy, scientific_settings)
        def save(name, data):
            """Atomically replace a complete run artifact."""
            atomic_yaml(output / name, data)
        current = dict(model=str(self.model_path), reference=str(self.reference_path),
             znd_conditions=self._condition_dict(), flame_conditions=self._condition_dict('flame'),
             ignition_conditions=self._condition_dict('ignition'), ignition_solver=asdict(self.ignition_solver),
             fuel=self.fuel, oxidizer=self.oxidizer, flame_solver=asdict(self.flame_solver),
             znd_solver=asdict(self.znd_solver), rates=asdict(self._rates),
             stoichiometry=[asdict(b.spec) for b in self._balances], orders=[asdict(s) for s in self._orders],
             trend=asdict(self._trend) if self._trend is not None else None,
             objectives={k: {'type': type(o).__name__, **asdict(o)} for k, o in self._objectives.items()},
             backend=execution_backend(search_settings.get('backend', 'process')), search=search_settings, progress_every=progress_every,
             history={'format': 'jsonl', 'mode': history_mode, 'flush_every': 1024, 'flush_seconds': 5.0})
        if resume:
            saved = yaml.safe_load((output / 'settings.yaml').read_text())
            check_compatible(saved, current)
            verify_mechanisms(output, self)
        else:
            # Validate history options before reference work; existing runs are never silently reused.
            HistoryWriter(output, mode=history_mode)
            save('settings.yaml', current)
            for name, scheme in [('original_model.yaml', self.scheme), ('reference_model.yaml', self.reference_scheme)]:
                save(name, scheme)
            save('mechanism_hashes.yaml', {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                                         for name in ('original_model.yaml','reference_model.yaml')})
            import importlib.metadata as metadata
            import platform
            versions = {'python': platform.python_version()}
            for name in ('numpy','scipy','cantera','PyYAML','matplotlib'):
                versions[name] = metadata.version(name)
            if mpi_active():
                from mpi4py import MPI
                versions['mpi4py'] = metadata.version('mpi4py')
                versions['mpi_library'] = MPI.Get_library_version().strip()
            save('versions.yaml', versions)
        write_restart_script(output)
        checkpoint = RestartStore(output / 'restart.sqlite', len(self.bounds), search_settings['sample_batch_size'])
        try:
            signature = scientific_settings(current)
            stored_signature = checkpoint.get('scientific_settings')
            if stored_signature is not None and stored_signature != signature:
                raise ValueError('Saved settings were changed after the checkpoint was created.')
            checkpoint.set('scientific_settings', signature)
            hashes = yaml.safe_load((output / 'mechanism_hashes.yaml').read_text())
            if checkpoint.get('mechanism_hashes', hashes) != hashes:
                raise ValueError('Mechanism snapshots differ from the saved checkpoint.')
            checkpoint.set('mechanism_hashes', hashes)
            state = checkpoint.runner_state
            if not state and resume:
                state = import_legacy(output, checkpoint, self.bounds, search_settings) or {}
            reference_path = output / 'reference_results.yaml'
            if state.get('reference') is not None:
                reference = state['reference']
            elif resume and reference_path.exists():
                reference = yaml.safe_load(reference_path.read_text())
            else:
                print('Calculating reference cases...', flush=True)
                with (output/'solver.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
                    reference = self.evaluate_reference()
            for family, cases, keys in self._groups():
                validate_outputs(reference[family], len(cases.P), keys)
            save('reference_results.yaml', reference)
            best = state.get('best')
            if best is not None:
                # JSON checkpoint keys are strings; keep the public reaction-number mapping.
                for name in ('order_sums', 'order_sum_penalties'):
                    best[name] = {int(k): v for k, v in best.get(name, {}).items()}
            nfev, failed = int(state.get('nfev', 0)), int(state.get('failed', 0))
            def save_best(record):
                """Rebuild exported best files from the checkpoint's authoritative record."""
                parameters = self.decode(record['vector'])
                save('best_model.yaml', self.mechanism(parameters))
                save('best_parameters.yaml', parameters)
                save('best_results.yaml', record['results'])
                save('best_loss.yaml', {k: record[k] for k in ('loss','fit_loss','regularization',
                     'weighted_regularization','objectives','order_sums','order_sum_penalties',
                     'order_sum_penalty','trend_slopes','trend_loss','weighted_trend_loss')})
            if best is not None:
                save_best(best)
            if resume:
                print(f'Restarting | saved evaluations {nfev} | failed {failed}', flush=True)
            last_count, last_time = nfev, monotonic()
            history_writer = HistoryWriter(output, mode=history_mode, resume_state=state.get('history'))
            with history_writer as history:
                def persist(force=False):
                    """Commit received work and flushed log offsets as one consistent checkpoint."""
                    nonlocal last_count, last_time
                    if not force and nfev-last_count < 1024 and monotonic()-last_time < 5:
                        return
                    snapshot = dict(nfev=nfev, failed=failed, best=best,
                                    reference=reference, history=history.snapshot())
                    checkpoint.commit(snapshot)
                    save('restart.yaml', dict(schema_version=1, database='restart.sqlite',
                         entry_point='restart.py', phase=checkpoint.get('phase', 'sampling'),
                         evaluations=nfev, failed=failed, completed_samples=checkpoint.completed_count,
                         completed_local_runs=len(checkpoint.local_runs()),
                         best_loss=None if best is None else best['loss']))
                    last_count, last_time = nfev, monotonic()
                def receive(record):
                    """Append one evaluation and immediately export any improvement."""
                    nonlocal best, nfev, failed
                    nfev += 1
                    valid = np.isfinite(record['loss'])
                    failed += int(not valid)
                    history.record(record, nfev)
                    if valid and (best is None or record['loss'] < best['loss']):
                        # Diagnostics can be large; they are not needed to restore the best candidate.
                        best = deepcopy({k: v for k, v in record.items() if k not in ('diagnostics','error')})
                        save_best(best)
                    if nfev % progress_every == 0:
                        value = 'none' if best is None else f'{best["loss"]:.6g}'
                        print(f'Eval {nfev} | best {value} | failed {failed}', flush=True)
                persist(force=True)
                try:
                    summary = search(CandidateObjective(self, reference), self.bounds, on_record=receive,
                         sample_directory=output, compact_errors=(history_mode == 'compact'),
                         checkpoint=checkpoint, on_checkpoint=persist, **search_settings)
                except BaseException as error:
                    save('status.yaml', {'status':'stopped', 'error':f'{type(error).__name__}: {error}',
                         'nfev':nfev, 'failed':failed, 'restart':'restart.py'})
                    raise
            summary.update(nfev=nfev, failed=failed, loss=best['loss'])
            save('search_summary.yaml', summary)
            save('status.yaml', {'status':'finished', 'restart':'restart.py'})
            print(f'Finished | best {best["loss"]:.6g} | results: {output}', flush=True)
            return TuningResult(self, best, reference, output, summary)
        finally:
            checkpoint.close()


class CandidateObjective:
    """Picklable callable; each process uses isolated files and captured output."""
    def __init__(self, problem, reference):
        """Store the configured problem and reference results for worker evaluation."""
        self.problem, self.reference = problem, reference

    def __call__(self, vector):
        """Evaluate one vector, capturing diagnostics and assigning infinity to expected failures."""
        import cantera as ct
        diagnostics = io.StringIO()
        record = {'vector': np.asarray(vector, dtype=float).tolist()}
        with redirect_stdout(diagnostics), redirect_stderr(diagnostics), warnings.catch_warnings():
            warnings.simplefilter('always')
            try:
                parameters = self.problem.decode(vector)
                results = self.problem.evaluate(parameters)
                record.update(self.problem.score(parameters, results, self.reference), results=results)
            # Expected candidate failures receive infinite loss; preserve the reason.
            except (ct.CanteraError, RuntimeError, ValueError, FloatingPointError) as error:
                record.update(loss=float('inf'), error=str(error))
        record['diagnostics'] = diagnostics.getvalue()
        return record


class TuningResult:
    """Best candidate, decoded parameters, reference results, and search/output metadata."""
    def __init__(self, problem, best, reference, output_dir, summary):
        """Store the best record and decode its parameter vector for later reuse."""
        self.problem, self.best, self.reference = problem, best, reference
        self.output_dir, self.summary = output_dir, summary
        self.parameters = problem.decode(best['vector'])
        self.loss = best['loss']

    def compare(self):
        """Return per-case predictions, references, and signed percentage errors."""
        comparison = {}
        for family, rows in self.best['results'].items():
            comparison[family] = [
                {'case': i + 1, **{key: {'model': row[key], 'reference': ref[key],
                  'error_percent': 100 * (row[key] / ref[key] - 1)} for key in row}}
                for i, (row, ref) in enumerate(zip(rows, self.reference[family]))
            ]
        return comparison

    def save_model(self, file_name):
        """Write the best candidate as a standalone mechanism YAML file."""
        path = Path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.problem.mechanism(self.parameters), sort_keys=False))
