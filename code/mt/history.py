"""Buffered evaluation history with bounded failure diagnostics."""
import json
import math
import os
from numbers import Integral, Real
from pathlib import Path
from time import monotonic

import yaml


ERROR_CATEGORIES = frozenset(('plog_validation', 'parameter_bounds', 'reaction_orders',
    'invalid_outputs', 'integration', 'convergence', 'znd_outputs',
    'mechanism_validation', 'numerical', 'cantera_other', 'other'))


def error_category(error):
    """Map changing solver messages to a small, fixed set of failure categories."""
    message = str(error or '')[:16384].lower()
    if 'plog' in message or 'p-log' in message:
        return 'plog_validation'
    if 'out of bounds' in message or 'outside bounds' in message:
        return 'parameter_bounds'
    if 'order' in message and any(word in message for word in ('negative', 'sum', 'bounds', 'non-reactant')):
        return 'reaction_orders'
    if any(word in message for word in ('positive finite outputs', 'missing or invalid outputs', 'nonfinite loss')):
        return 'invalid_outputs'
    if any(word in message for word in ('cvodes', 'idasolve', 'integrator', 'integration failed', 'required step size')):
        return 'integration'
    if any(word in message for word in ('converg', 'newton', 'no solution')):
        return 'convergence'
    if any(word in message for word in ('znd', 'induction', 'exothermic', 'thermicity')):
        return 'znd_outputs'
    if any(word in message for word in ('not balanced', 'reaction equation', 'unknown species', 'inputfileerror')):
        return 'mechanism_validation'
    if any(word in message for word in ('overflow', 'divide by zero', 'floating point', 'nonfinite', 'not finite')):
        return 'numerical'
    if 'cantera' in message:
        return 'cantera_other'
    return 'other'


def json_value(value):
    """Convert NumPy values and nonfinite numbers without modifying the source record."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, 'tolist'):
        return json_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize history value of type {type(value).__name__}.')


class HistoryWriter:
    """Write one JSON line per evaluation and only a few detailed failure examples."""

    def __init__(self, output, mode='compact', flush_every=1024, flush_seconds=5.0,
                 examples_per_error=3, resume_state=None):
        """Configure buffered files and fixed-size failure counters for one run."""
        if mode not in ('compact', 'full'):
            raise ValueError('history mode must be compact or full.')
        if isinstance(flush_every, bool) or not isinstance(flush_every, int) or flush_every < 1:
            raise ValueError('flush_every must be a positive integer.')
        if isinstance(flush_seconds, bool) or not isinstance(flush_seconds, Real) or not math.isfinite(flush_seconds) or flush_seconds <= 0:
            raise ValueError('flush_seconds must be positive and finite.')
        if isinstance(examples_per_error, bool) or not isinstance(examples_per_error, int) or examples_per_error < 0:
            raise ValueError('examples_per_error must be a nonnegative integer.')
        self.output, self.mode = Path(output), mode
        self.flush_every, self.flush_seconds = flush_every, float(flush_seconds)
        self.examples_per_error = examples_per_error
        self.evaluations = self.failed = self.diagnostic_examples = 0
        self.failure_counts = {}
        self.failure_examples = {}
        self._history = self._solver = None
        self._unflushed = 0
        self._last_flush = monotonic()
        self._resume_state = resume_state
        if resume_state is not None:
            self.evaluations = int(resume_state['evaluations'])
            self.failed = int(resume_state['failed'])
            self.failure_counts = dict(resume_state['failure_counts'])
            self.failure_examples = dict(resume_state.get('failure_examples', {}))
            self.diagnostic_examples = int(resume_state.get('diagnostic_examples', 0))

    def __enter__(self):
        """Open bounded file buffers; the surrounding runner creates the run directory."""
        if self._history is not None:
            raise RuntimeError('HistoryWriter is already open.')
        if self._resume_state is not None:
            # Discard only uncommitted tails; the database is the checkpoint authority.
            for name, key in (('history.jsonl', 'history_offset'), ('solver.log', 'solver_offset')):
                path = self.output / name
                offset = self._resume_state[key]
                if not path.exists() or path.stat().st_size < offset:
                    raise ValueError(f'{name} is shorter than its saved restart position.')
            for name, key in (('history.jsonl', 'history_offset'), ('solver.log', 'solver_offset')):
                with (self.output / name).open('r+b') as stream:
                    stream.truncate(self._resume_state[key])
        mode = 'a' if self._resume_state is not None else 'w'
        self._history = (self.output / 'history.jsonl').open(mode, buffering=262144)
        try:
            self._solver = (self.output / 'solver.log').open('a', buffering=65536)
        except BaseException:
            self._history.close()
            self._history = None
            raise
        self._last_flush = monotonic()
        return self

    def _example(self, record, evaluation, category):
        """Append one selected failure or successful diagnostic to the solver log."""
        self._solver.write(f'\nEvaluation {evaluation} | stage={record.get("stage")} '
                           f'| sample_index={record.get("sample_index")} | {category}\n')
        for key in ('error', 'diagnostics'):
            if record.get(key):
                self._solver.write(f'{key}:\n{record[key]}\n')

    def record(self, record, evaluation):
        """Write one immutable snapshot and aggregate failures without retaining records."""
        if self._history is None:
            raise RuntimeError('Open HistoryWriter with a with statement before recording.')
        valid = math.isfinite(float(record['loss']))
        category = None
        if not valid:
            category = record.get('error_code')
            if category not in ERROR_CATEGORIES:
                category = error_category(record.get('error'))
        excluded = ('results', 'diagnostics', 'error') if self.mode == 'compact' else ('results',)
        entry = {key: value for key, value in record.items() if key not in excluded}
        entry.update(evaluation=evaluation, loss=record['loss'] if valid else None,
                     error_code=category)
        self._history.write(json.dumps(json_value(entry), separators=(',', ':'), allow_nan=False) + '\n')
        self.evaluations += 1
        self._unflushed += 1
        if not valid:
            self.failed += 1
            self.failure_counts[category] = self.failure_counts.get(category, 0) + 1
            saved = self.failure_examples.get(category, 0)
            if saved < self.examples_per_error and (record.get('error') or record.get('diagnostics')):
                self._example(record, evaluation, category)
                self.failure_examples[category] = saved + 1
        elif self.mode == 'compact' and record.get('diagnostics') and self.diagnostic_examples < self.examples_per_error:
            self._example(record, evaluation, 'successful_diagnostics')
            self.diagnostic_examples += 1
        if self._unflushed >= self.flush_every or monotonic() - self._last_flush >= self.flush_seconds:
            self.flush()

    def flush(self):
        """Flush buffered history and example logs while leaving file ownership here."""
        if self._history is not None:
            self._history.flush()
            self._solver.flush()
            self._unflushed = 0
            self._last_flush = monotonic()

    def snapshot(self):
        """Flush durable log bytes before committing their offsets to the restart database."""
        self.flush()
        for stream in (self._history, self._solver):
            os.fsync(stream.fileno())
        return {
            'evaluations': self.evaluations, 'failed': self.failed,
            'failure_counts': dict(self.failure_counts),
            'failure_examples': dict(self.failure_examples),
            'diagnostic_examples': self.diagnostic_examples,
            'history_offset': self._history.tell(), 'solver_offset': self._solver.tell(),
        }

    def close(self):
        """Flush on normal or exceptional exit and save small counts and schema metadata."""
        if self._history is None:
            return
        try:
            self.flush()
        finally:
            self._history.close()
            self._solver.close()
            self._history = self._solver = None
        summary = {
            'schema_version': 1, 'mode': self.mode,
            'evaluations': self.evaluations, 'successful': self.evaluations - self.failed,
            'failed': self.failed, 'failure_counts': self.failure_counts,
            'failure_examples': self.failure_examples,
            'successful_diagnostic_examples': self.diagnostic_examples,
            'flush_every': self.flush_every, 'flush_seconds': self.flush_seconds,
            'examples_per_error': self.examples_per_error,
            'schema': {
                'history.jsonl': 'One JSON object per evaluation, in receipt order; evaluation is one-based.',
                'sample_index': 'Original zero-based sample index; local records identify their starting sample.',
                'loss': 'Finite loss, or null for a failed candidate.',
                'error_code': 'Fixed failure category, or null for a successful candidate.',
                'vector': 'Full candidate parameter vector; objective fields are retained when available.',
                'nonfinite_values': 'Encoded as null for standard JSON compatibility.',
                'solver.log': 'Reference diagnostics and bounded full examples for each failure category.',
                'full_mode': 'Also retains every error and diagnostic string in history.jsonl.',
                'results': 'Per-case results are omitted from history; improvements are saved as best_results.yaml.',
            },
        }
        (self.output / 'history_summary.yaml').write_text(yaml.safe_dump(summary, sort_keys=False))

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the files without suppressing a solver or search exception."""
        self.close()
        return False
