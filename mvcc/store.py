"""Snapshot-isolation MVCC store kept in local memory / local files.

Version model
-------------
Every logical record (identified by a string key) owns a chain of
versions. Each version carries:

* ``commit_ts`` -- the commit timestamp of the transaction that created it
* ``value``     -- the stored payload, or the ``TOMBSTONE`` sentinel when
  the creating transaction deleted the record
* ``tx_id``     -- the id of the transaction that created the version

Visibility (snapshot isolation)
-------------------------------
A transaction reads a fixed snapshot taken at ``begin``. A version is
visible to snapshot ``S`` iff it is the newest version of its key with
``commit_ts <= S``. A transaction always sees its own uncommitted writes
on top of the snapshot. Versions committed after the snapshot are never
visible.

Conflict rule (first committer wins)
------------------------------------
If any version of a key the transaction wrote has ``commit_ts`` greater
than the transaction's snapshot, the write is rejected and the
transaction is aborted with :class:`WriteConflictError`. Two concurrent
transactions writing the same key can therefore never both succeed.

Garbage collection
------------------
The GC watermark is the oldest snapshot timestamp among active
transactions (or the latest commit timestamp when none are active).
Versions older than the newest version visible at the watermark can
never be read by any present or future transaction and are removed.
"""

import bisect
import threading

from .errors import TransactionStateError, WriteConflictError
from .wal import WriteAheadLog


class _Tombstone:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "TOMBSTONE"


TOMBSTONE = _Tombstone()

_ACTIVE = "active"
_COMMITTED = "committed"
_ABORTED = "aborted"


class _Version:
    __slots__ = ("commit_ts", "value", "tx_id")

    def __init__(self, commit_ts, value, tx_id):
        self.commit_ts = commit_ts
        self.value = value
        self.tx_id = tx_id

    def __repr__(self):
        return "Version(ts=%r, value=%r, tx=%r)" % (
            self.commit_ts,
            self.value,
            self.tx_id,
        )


class Transaction:
    """Handle for one running transaction. Created by ``MVCCStore.begin``."""

    def __init__(self, store, tx_id, snapshot_ts):
        self._store = store
        self.id = tx_id
        self.snapshot_ts = snapshot_ts
        self.state = _ACTIVE
        self.write_set = {}

    def read(self, key):
        """Read ``key`` from this transaction's fixed snapshot."""
        return self._store.read(self, key)

    def write(self, key, value):
        """Buffer a write of ``value`` to ``key`` (visible to self only)."""
        self._store.write(self, key, value)

    def delete(self, key):
        """Buffer a versioned tombstone for ``key``."""
        self._store.delete(self, key)

    def commit(self):
        """Commit, raising :class:`WriteConflictError` on conflict."""
        return self._store.commit(self)

    def rollback(self):
        """Abort the transaction and discard all buffered writes."""
        self._store.rollback(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.state == _ACTIVE:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
        return False


class MVCCStore:
    """A local, thread-safe MVCC key-value store with snapshot isolation."""

    def __init__(self, wal_path=None):
        self._lock = threading.RLock()
        self._commit_ts = 0
        self._txn_seq = 0
        # key -> list[_Version] sorted ascending by commit_ts
        self._versions = {}
        self._active = {}
        self._wal = WriteAheadLog(wal_path) if wal_path else None
        if self._wal:
            self._recover()

    # ------------------------------------------------------------------
    # recovery
    # ------------------------------------------------------------------
    def _recover(self):
        for commit_ts, key, op, value in WriteAheadLog.replay(self._wal.path):
            version = _Version(commit_ts, TOMBSTONE if op == "del" else value, -1)
            chain = self._versions.setdefault(key, [])
            chain.append(version)
            self._commit_ts = max(self._commit_ts, commit_ts)
        for chain in self._versions.values():
            chain.sort(key=lambda v: v.commit_ts)

    # ------------------------------------------------------------------
    # transaction lifecycle
    # ------------------------------------------------------------------
    def begin(self):
        """Start a transaction reading a fixed snapshot of the store."""
        with self._lock:
            self._txn_seq += 1
            txn = Transaction(self, self._txn_seq, self._commit_ts)
            self._active[txn.id] = txn
            return txn

    def commit(self, txn):
        """Commit ``txn``; abort it and raise on write-write conflict."""
        with self._lock:
            self._check_active(txn)
            for key in txn.write_set:
                if self._has_conflict(txn, key):
                    self._abort_locked(txn)
                    raise WriteConflictError(
                        "transaction %d conflicts on key %r: a newer version "
                        "was committed after its snapshot" % (txn.id, key)
                    )
            self._commit_ts += 1
            commit_ts = self._commit_ts
            for key, value in txn.write_set.items():
                version = _Version(commit_ts, value, txn.id)
                chain = self._versions.setdefault(key, [])
                chain.append(version)
                if self._wal:
                    if value is TOMBSTONE:
                        self._wal.append(commit_ts, key, "del", None)
                    else:
                        self._wal.append(commit_ts, key, "put", value)
            txn.state = _COMMITTED
            del self._active[txn.id]
            return commit_ts

    def rollback(self, txn):
        """Abort ``txn``; all of its buffered writes are discarded."""
        with self._lock:
            self._check_active(txn)
            self._abort_locked(txn)

    def _abort_locked(self, txn):
        txn.state = _ABORTED
        txn.write_set.clear()
        del self._active[txn.id]

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------
    def read(self, txn, key):
        """Return the value visible to ``txn``'s snapshot, or ``None``."""
        with self._lock:
            self._check_active(txn)
            if key in txn.write_set:
                value = txn.write_set[key]
                return None if value is TOMBSTONE else value
            version = self._visible_version(key, txn.snapshot_ts)
            if version is None or version.value is TOMBSTONE:
                return None
            return version.value

    def write(self, txn, key, value):
        """Buffer ``value`` for ``key``; fail fast on a known conflict."""
        with self._lock:
            self._check_active(txn)
            self._fail_fast_on_conflict(txn, key)
            txn.write_set[key] = value

    def delete(self, txn, key):
        """Buffer a tombstone for ``key``; fail fast on a known conflict."""
        with self._lock:
            self._check_active(txn)
            self._fail_fast_on_conflict(txn, key)
            txn.write_set[key] = TOMBSTONE

    # ------------------------------------------------------------------
    # garbage collection
    # ------------------------------------------------------------------
    def gc(self):
        """Collect versions no active or future transaction can read.

        Returns a dict with the number of versions and keys removed.
        """
        with self._lock:
            if self._active:
                watermark = min(t.snapshot_ts for t in self._active.values())
            else:
                watermark = self._commit_ts
            removed_versions = 0
            removed_keys = 0
            for key in list(self._versions):
                chain = self._versions[key]
                # Index of the newest version visible at the watermark.
                idx = bisect.bisect_right(
                    [v.commit_ts for v in chain], watermark
                ) - 1
                if idx < 0:
                    continue
                if idx > 0:
                    removed_versions += idx
                    del chain[:idx]
                # If the only version anyone can still see is a tombstone
                # and nothing newer exists, the whole key can disappear.
                if len(chain) == 1 and chain[0].value is TOMBSTONE:
                    del self._versions[key]
                    removed_versions += 1
                    removed_keys += 1
            return {"versions_removed": removed_versions,
                    "keys_removed": removed_keys}

    # ------------------------------------------------------------------
    # introspection helpers (used by tests and debugging)
    # ------------------------------------------------------------------
    def version_count(self, key):
        with self._lock:
            return len(self._versions.get(key, ()))

    def active_transactions(self):
        with self._lock:
            return sorted(self._active)

    @property
    def last_commit_ts(self):
        with self._lock:
            return self._commit_ts

    def close(self):
        with self._lock:
            if self._wal:
                self._wal.close()
                self._wal = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _check_active(self, txn):
        if txn.state != _ACTIVE:
            raise TransactionStateError(
                "transaction %d is %s" % (txn.id, txn.state)
            )

    def _visible_version(self, key, snapshot_ts):
        chain = self._versions.get(key)
        if not chain:
            return None
        idx = bisect.bisect_right(
            [v.commit_ts for v in chain], snapshot_ts
        ) - 1
        if idx < 0:
            return None
        return chain[idx]

    def _has_conflict(self, txn, key):
        chain = self._versions.get(key)
        return bool(chain) and chain[-1].commit_ts > txn.snapshot_ts

    def _fail_fast_on_conflict(self, txn, key):
        if key not in txn.write_set and self._has_conflict(txn, key):
            self._abort_locked(txn)
            raise WriteConflictError(
                "transaction %d conflicts on key %r: a newer version was "
                "committed after its snapshot" % (txn.id, key)
            )
