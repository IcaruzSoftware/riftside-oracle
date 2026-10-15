"""Shared test fixtures: import path and a fake Supabase client."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fake Supabase — records writes, answers the query chains evaluate/predict use
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class _Query:
    def __init__(self, table):
        self.t = table
        self.kind = "select"
        self.cols = "*"
        self.payload = None
        self.on_conflict = None
        self.filters = []
        self._negate = False
        self.count = None
        self.head = False

    def select(self, cols="*", count=None, head=None):
        self.kind, self.cols, self.count, self.head = "select", cols, count, bool(head)
        return self

    @property
    def not_(self):
        self._negate = True
        return self

    def is_(self, col, val):
        self.filters.append(("is", col, val, self._negate)); self._negate = False
        return self

    def eq(self, col, val):
        self.filters.append(("eq", col, val, self._negate)); self._negate = False
        return self

    def gte(self, col, val):
        self.filters.append(("gte", col, val, self._negate)); self._negate = False
        return self

    def update(self, payload):
        self.kind, self.payload = "update", payload
        return self

    def upsert(self, payload, on_conflict=None):
        self.kind, self.payload, self.on_conflict = "upsert", payload, on_conflict
        return self

    def insert(self, payload):
        self.kind, self.payload = "insert", payload
        return self

    def delete(self):
        self.kind = "delete"
        return self

    def _match(self, row):
        for kind, col, val, neg in self.filters:
            if kind == "is" and val == "null":
                ok = row.get(col) is None
            elif kind == "eq":
                ok = row.get(col) == val
            elif kind == "gte":
                ok = row.get(col) is not None and str(row.get(col)) >= str(val)
            else:
                ok = True
            if neg:
                ok = not ok
            if not ok:
                return False
        return True

    def execute(self):
        t = self.t
        if self.kind == "select":
            rows = [r for r in t.rows if self._match(r)]
            return _Result([] if self.head else [dict(r) for r in rows],
                           len(rows) if self.count == "exact" else None)
        if self.kind == "update":
            for r in t.rows:
                if self._match(r):
                    r.update(self.payload)
            return _Result([])
        if self.kind == "upsert":
            key = self.on_conflict or "id"
            row = dict(self.payload)
            existing = next((r for r in t.rows if r.get(key) == row.get(key)), None)
            if existing:
                existing.update(row)
                out = existing
            else:
                row.setdefault("id", f"id-{len(t.rows)+1}")
                t.rows.append(row)
                out = row
            t.db.writes.append((t.name, "upsert", dict(out)))
            return _Result([dict(out)])
        if self.kind == "insert":
            payload = self.payload if isinstance(self.payload, list) else [self.payload]
            for row in payload:
                row = dict(row)
                row.setdefault("id", f"id-{len(t.rows)+1}")
                t.rows.append(row)
                t.db.writes.append((t.name, "insert", row))
            return _Result([dict(r) for r in payload])
        if self.kind == "delete":
            kept = [r for r in t.rows if not self._match(r)]
            removed = len(t.rows) - len(kept)
            t.rows[:] = kept
            t.db.writes.append((t.name, "delete", {"removed": removed}))
            return _Result([])
        return _Result([])


class _Table:
    def __init__(self, name, db):
        self.name = name
        self.db = db
        self.rows = db.seed.get(name, [])

    def __getattr__(self, item):
        return getattr(_Query(self), item)


class FakeSupabase:
    def __init__(self, seed=None):
        self.seed = {k: [dict(r) for r in v] for k, v in (seed or {}).items()}
        self._tables = {}
        self.writes = []

    def table(self, name):
        if name not in self._tables:
            self._tables[name] = _Table(name, self)
        return self._tables[name]


@pytest.fixture
def fake_supabase():
    return FakeSupabase

