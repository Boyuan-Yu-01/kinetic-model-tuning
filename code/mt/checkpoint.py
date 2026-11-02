"""Transactional search state for restarting interrupted optimization runs."""
import json
import sqlite3
import numpy as np

from .sample_store import SampleStore


class RestartStore(SampleStore):
    """Keep sampled minima and completed work in a durable, bounded-memory database."""

    def __init__(self, path, dimensions, batch_size):
        """Open an existing checkpoint or create its tables without committing progress."""
        self.dimensions, self.batch_size = dimensions, batch_size
        self.safe_to_commit = True
        self.connection = sqlite3.connect(str(path))
        self.connection.execute('PRAGMA cache_size=-8192')
        self.connection.execute('PRAGMA temp_store=FILE')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute(
            'CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, loss REAL, unit BLOB)')
        self.connection.execute(
            'CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.connection.execute(
            'CREATE TABLE IF NOT EXISTS completed_samples '
            '(start INTEGER PRIMARY KEY, stop INTEGER NOT NULL)')
        self.connection.execute(
            'CREATE TABLE IF NOT EXISTS local_runs (seed_index INTEGER PRIMARY KEY, result TEXT NOT NULL)')
        version = self.get('schema_version', 1)
        if version != 1:
            self.close()
            raise ValueError(f'Unsupported restart database schema: {version}')
        self.set('schema_version', 1)
        self.count = self.connection.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
        self.completed_count = self.connection.execute(
            'SELECT COALESCE(SUM(stop-start), 0) FROM completed_samples').fetchone()[0]
        stored_dimensions = self.get('dimensions')
        if stored_dimensions is not None and stored_dimensions != dimensions:
            self.close()
            raise ValueError('Restart checkpoint dimensions do not match this problem.')
        self.set('dimensions', dimensions)

    def get(self, key, default=None):
        """Read JSON metadata, including values pending in the current transaction."""
        row = self.connection.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def set(self, key, value):
        """Stage JSON metadata for the next joint search/history checkpoint."""
        self.connection.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)',
                                (key, json.dumps(value, separators=(',', ':'))))

    @property
    def runner_state(self):
        """Return counters, best result, and history offsets from the last checkpoint."""
        return self.get('runner_state', {})

    def commit(self, runner_state):
        """Atomically persist search progress alongside the matching runner state."""
        self.set('runner_state', runner_state)
        self.connection.commit()

    def close(self):
        """Discard uncheckpointed progress; the runner decides when history is durable."""
        self.connection.rollback()
        self.connection.close()

    def add(self, index, loss, unit):
        """Stage one finite sample without SampleStore's independent auto-commit."""
        self.connection.execute('INSERT INTO samples VALUES (?, ?, ?)',
                                (int(index), float(loss), np.asarray(unit, dtype=np.float64).tobytes()))
        self.count += 1

    def import_sample(self, index, unit, loss):
        """Mark one received sample complete, merging adjacent ranges on disk."""
        index = int(index)
        previous = self.connection.execute(
            'SELECT start, stop FROM completed_samples WHERE start<=? ORDER BY start DESC LIMIT 1',
            (index,)).fetchone()
        if previous is not None and index < previous[1]:
            return False
        following = self.connection.execute(
            'SELECT start, stop FROM completed_samples WHERE start=?', (index + 1,)).fetchone()
        start, stop = index, index + 1
        if previous is not None and previous[1] == index:
            start = previous[0]
            self.connection.execute('DELETE FROM completed_samples WHERE start=?', (start,))
        if following is not None:
            stop = following[1]
            self.connection.execute('DELETE FROM completed_samples WHERE start=?', (following[0],))
        self.connection.execute('INSERT INTO completed_samples VALUES (?, ?)', (start, stop))
        if np.isfinite(loss):
            self.add(index, loss, unit)
        self.completed_count += 1
        return True

    def missing_ranges(self, samples):
        """Stream unfinished index intervals without loading all completed IDs."""
        position = 0
        while position < samples:
            previous = self.connection.execute(
                'SELECT stop FROM completed_samples WHERE start<=? ORDER BY start DESC LIMIT 1',
                (position,)).fetchone()
            if previous is not None and previous[0] > position:
                position = previous[0]
                continue
            following = self.connection.execute(
                'SELECT start, stop FROM completed_samples WHERE start>? ORDER BY start LIMIT 1',
                (position,)).fetchone()
            stop = min(samples, following[0]) if following else samples
            if position < stop:
                yield position, stop
            position = following[1] if following else samples

    def finish(self):
        """Index finite losses for selection without committing the runner's transaction."""
        self.connection.execute('CREATE INDEX IF NOT EXISTS loss_order ON samples(loss, id)')

    def set_selection(self, selected, method):
        """Stage the chosen seeds so a restart retains exactly the same local starts."""
        self.set('selection', {'method': method,
                               'seeds': [[int(index), unit.tolist()] for index, unit in selected]})

    def get_selection(self):
        """Return saved normalized seeds and their selection method, if already selected."""
        saved = self.get('selection')
        if saved is None:
            return None
        return ([(int(index), np.asarray(unit, dtype=float)) for index, unit in saved['seeds']],
                saved['method'])

    def save_local_run(self, result):
        """Mark a returned Powell run complete; interrupted runs are restarted from their seed."""
        self.connection.execute('INSERT OR REPLACE INTO local_runs VALUES (?, ?)',
                                (int(result['seed_index']), json.dumps(result, separators=(',', ':'))))

    def local_runs(self):
        """Read the small set of completed local summaries in seed-index order."""
        return [json.loads(row[0]) for row in self.connection.execute(
            'SELECT result FROM local_runs ORDER BY seed_index')]
