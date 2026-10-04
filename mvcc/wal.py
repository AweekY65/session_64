"""Write-ahead log persisted to a local file.

The log is an append-only sequence of JSON lines, one per committed
version. It is the only persistence mechanism and lives entirely on the
local filesystem; no external service is involved.
"""

import json
import os
import threading


class WriteAheadLog:
    """Append-only local file log of committed versions."""

    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    @property
    def path(self):
        return self._path

    def append(self, commit_ts, key, op, value):
        """Append one committed version record and flush it to disk."""
        record = {"ts": commit_ts, "key": key, "op": op, "value": value}
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self):
        with self._lock:
            self._file.close()

    @staticmethod
    def replay(path):
        """Yield (commit_ts, key, op, value) records from an existing log."""
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                yield record["ts"], record["key"], record["op"], record["value"]
