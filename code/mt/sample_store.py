"""Disk-backed valid samples and bounded-memory, exact seed selection."""
import sqlite3
import numpy as np
from scipy.spatial import cKDTree


class SampleStore:
    """Keep finite-loss samples in SQLite rather than retaining all candidates in RAM."""
    def __init__(self, path, dimensions, batch_size):
        """Create a private disk database with a small, bounded SQLite page cache."""
        self.dimensions, self.batch_size = dimensions, batch_size
        self.count = 0
        self.connection = sqlite3.connect(str(path))
        self.connection.execute('PRAGMA cache_size=-8192')  # 8 MiB page cache.
        self.connection.execute('PRAGMA temp_store=FILE')
        self.connection.execute('CREATE TABLE samples (id INTEGER PRIMARY KEY, loss REAL, unit BLOB)')

    def close(self):
        """Close the database before its temporary directory is removed."""
        self.connection.close()

    def add(self, index, loss, unit):
        """Store one successful sample; callers omit failed candidates."""
        self.connection.execute('INSERT INTO samples VALUES (?, ?, ?)',
                                (int(index), float(loss), np.asarray(unit, dtype=np.float64).tobytes()))
        self.count += 1
        if self.count % self.batch_size == 0:
            self.connection.commit()

    def finish(self):
        """Commit samples and index loss ordering on disk for greedy seed selection."""
        self.connection.commit()
        self.connection.execute('CREATE INDEX loss_order ON samples(loss, id)')
        self.connection.commit()

    def _batches(self, by_loss=False):
        """Read bounded batches in a deterministic order and close the cursor."""
        order = 'loss, id' if by_loss else 'id'
        cursor = self.connection.execute(f'SELECT id, loss, unit FROM samples ORDER BY {order}')
        try:
            while batch := cursor.fetchmany(self.batch_size):
                yield batch
        finally:
            cursor.close()

    def rows(self, by_loss=False):
        """Stream sample coordinates without retaining the database in memory."""
        for batch in self._batches(by_loss):
            for index, loss, blob in batch:
                yield index, loss, np.frombuffer(blob, dtype=np.float64)

    def blocks(self):
        """Read normalized coordinates in small blocks for disk-based neighbor scans."""
        for rows in self._batches():
            yield (np.array([r[0] for r in rows], dtype=np.int64),
                   np.array([r[1] for r in rows]),
                   np.stack([np.frombuffer(r[2], dtype=np.float64) for r in rows]))

    def select(self, starts, neighbors, min_distance, max_in_memory):
        """Select low-loss separated sampled minima using all finite-loss samples.

        Small valid sets use a KD tree; larger sets use exact blockwise disk scans.
        The fallback is slower but never drops samples or approximates neighbors.
        Distance ties are broken by original sample index in both paths.
        """
        if not self.count:
            raise RuntimeError('All sampled candidates failed; inspect evaluation history and solver.log.')
        k = min(neighbors, self.count - 1)
        tree = None
        if self.count <= max_in_memory:
            ids = np.empty(self.count, dtype=np.int64)
            losses = np.empty(self.count)
            units = np.empty((self.count, self.dimensions))
            for j, (index, loss, unit) in enumerate(self.rows()):
                ids[j], losses[j], units[j] = index, loss, unit
            tree = cKDTree(units)
        method = 'kdtree' if tree is not None else 'disk_exact'
        print(f'Selecting seeds from {self.count} valid samples ({method}).', flush=True)
        selected = []
        for index, loss, unit in self.rows(by_loss=True):
            # A point already too close cannot become a selected seed.
            if any(np.linalg.norm(unit - other[1]) < min_distance for other in selected):
                continue
            if k and tree is not None:
                distances, _ = tree.query(unit, k=k + 1)
                radius = np.nextafter(float(np.max(distances)), np.inf)
                near = np.asarray(tree.query_ball_point(unit, radius), dtype=int)
                near = near[ids[near] != index]
                d2 = np.sum((units[near] - unit)**2, axis=1)
                near = near[np.lexsort((ids[near], d2))[:k]]
                minimum = np.all(loss <= losses[near])
            elif k:
                # Merge only the nearest k entries from each bounded block.
                best = []
                for block_ids, block_losses, block_units in self.blocks():
                    keep = block_ids != index
                    d2 = np.sum((block_units - unit)**2, axis=1)
                    positions = np.flatnonzero(keep)
                    nearest = positions[np.lexsort((block_ids[positions], d2[positions]))[:k]]
                    best.extend((float(d2[j]), int(block_ids[j]), float(block_losses[j])) for j in nearest)
                    best = sorted(best)[:k]
                minimum = all(loss <= item[2] for item in best)
            else:
                minimum = True
            if minimum:
                selected.append((int(index), unit.copy()))
                if len(selected) == starts:
                    break
        return selected, method
