"""Terminal-runnable test suite for the local MVCC store.

Run with:  python3 -m unittest discover -s tests -v
No database, service or GUI is required.
"""

import os
import random
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvcc import MVCCStore, TransactionStateError, WriteConflictError


class SnapshotIsolationTest(unittest.TestCase):
    def setUp(self):
        self.store = MVCCStore()

    def test_reads_at_different_points_in_time(self):
        """Distinct snapshots observe distinct committed states."""
        t1 = self.store.begin()
        t1.write("k", "v1")
        t1.commit()

        reader_old = self.store.begin()  # snapshot pinned at v1

        t2 = self.store.begin()
        t2.write("k", "v2")
        t2.commit()

        reader_new = self.store.begin()  # snapshot after v2 committed

        self.assertEqual(reader_old.read("k"), "v1")
        self.assertEqual(reader_new.read("k"), "v2")

    def test_snapshot_is_stable_repeatable_read(self):
        """A transaction never sees data committed after its snapshot."""
        t1 = self.store.begin()
        t1.write("k", "v1")
        t1.commit()

        reader = self.store.begin()
        self.assertEqual(reader.read("k"), "v1")

        t2 = self.store.begin()
        t2.write("k", "v2")
        t2.commit()

        # Same key read again inside the same transaction: unchanged.
        self.assertEqual(reader.read("k"), "v1")
        # A brand new transaction sees the newer commit.
        self.assertEqual(self.store.begin().read("k"), "v2")

    def test_transaction_sees_own_writes(self):
        txn = self.store.begin()
        txn.write("a", 1)
        self.assertEqual(txn.read("a"), 1)
        txn.delete("a")
        self.assertIsNone(txn.read("a"))
        txn.rollback()

    def test_uncommitted_writes_are_invisible_to_others(self):
        writer = self.store.begin()
        writer.write("k", "dirty")
        other = self.store.begin()
        self.assertIsNone(other.read("k"))
        writer.rollback()


class ConflictTest(unittest.TestCase):
    def setUp(self):
        self.store = MVCCStore()

    def test_write_write_conflict_second_committer_fails(self):
        t1 = self.store.begin()
        t2 = self.store.begin()
        t1.write("k", "from-t1")
        t2.write("k", "from-t2")
        t1.commit()
        with self.assertRaises(WriteConflictError):
            t2.commit()
        # t2 was aborted by the failed commit; the winner's value stands.
        self.assertEqual(self.store.begin().read("k"), "from-t1")
        self.assertEqual(t2.state, "aborted")

    def test_conflict_detected_fail_fast_at_write_time(self):
        t1 = self.store.begin()
        t2 = self.store.begin()
        t1.write("k", 1)
        t1.commit()
        with self.assertRaises(WriteConflictError):
            t2.write("k", 2)
        self.assertEqual(t2.state, "aborted")

    def test_disjoint_keys_do_not_conflict(self):
        t1 = self.store.begin()
        t2 = self.store.begin()
        t1.write("a", 1)
        t2.write("b", 2)
        t1.commit()
        t2.commit()  # must not raise
        reader = self.store.begin()
        self.assertEqual(reader.read("a"), 1)
        self.assertEqual(reader.read("b"), 2)

    def test_concurrent_updates_exactly_one_winner(self):
        """Many threads racing on one key: exactly one commit succeeds."""
        for _ in range(20):
            store = MVCCStore()
            seed = store.begin()
            seed.write("counter", 0)
            seed.commit()

            results = []
            lock = threading.Lock()
            # All transactions begin before anyone commits, so they share
            # the same snapshot and genuinely race on the same key.
            txns = [store.begin() for _ in range(8)]

            def worker(txn, value):
                try:
                    txn.write("counter", value)
                    txn.commit()
                    outcome = "committed"
                except WriteConflictError:
                    outcome = "conflict"
                with lock:
                    results.append(outcome)

            threads = [threading.Thread(target=worker, args=(txns[i], i))
                       for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(results.count("committed"), 1)
            self.assertEqual(results.count("conflict"), 7)


class RollbackTest(unittest.TestCase):
    def setUp(self):
        self.store = MVCCStore()

    def test_rollback_discards_all_writes(self):
        t1 = self.store.begin()
        t1.write("a", 1)
        t1.write("b", 2)
        t1.rollback()
        reader = self.store.begin()
        self.assertIsNone(reader.read("a"))
        self.assertIsNone(reader.read("b"))

    def test_rollback_restores_previous_version(self):
        t1 = self.store.begin()
        t1.write("k", "stable")
        t1.commit()
        t2 = self.store.begin()
        t2.write("k", "broken")
        t2.rollback()
        self.assertEqual(self.store.begin().read("k"), "stable")

    def test_operations_after_commit_or_rollback_fail(self):
        txn = self.store.begin()
        txn.write("k", 1)
        txn.commit()
        with self.assertRaises(TransactionStateError):
            txn.read("k")
        with self.assertRaises(TransactionStateError):
            txn.write("k", 2)
        txn2 = self.store.begin()
        txn2.rollback()
        with self.assertRaises(TransactionStateError):
            txn2.delete("k")


class DeleteVisibilityTest(unittest.TestCase):
    def setUp(self):
        self.store = MVCCStore()

    def test_delete_hides_record_from_new_snapshots(self):
        t1 = self.store.begin()
        t1.write("k", "v")
        t1.commit()
        t2 = self.store.begin()
        t2.delete("k")
        t2.commit()
        self.assertIsNone(self.store.begin().read("k"))

    def test_old_snapshot_still_reads_deleted_record(self):
        t1 = self.store.begin()
        t1.write("k", "v")
        t1.commit()

        old_reader = self.store.begin()

        t2 = self.store.begin()
        t2.delete("k")
        t2.commit()

        # Tombstone is a version: the old snapshot reads right past it.
        self.assertEqual(old_reader.read("k"), "v")
        self.assertIsNone(self.store.begin().read("k"))

    def test_delete_then_recreate(self):
        t1 = self.store.begin()
        t1.write("k", "v1")
        t1.commit()
        t2 = self.store.begin()
        t2.delete("k")
        t2.commit()
        t3 = self.store.begin()
        t3.write("k", "v2")
        t3.commit()
        self.assertEqual(self.store.begin().read("k"), "v2")

    def test_concurrent_delete_conflicts_with_update(self):
        t1 = self.store.begin()
        t1.write("k", "v")
        t1.commit()
        deleter = self.store.begin()
        updater = self.store.begin()
        deleter.delete("k")
        updater.write("k", "v2")
        deleter.commit()
        with self.assertRaises(WriteConflictError):
            updater.commit()


class LongTransactionTest(unittest.TestCase):
    def setUp(self):
        self.store = MVCCStore()

    def test_long_transaction_pins_its_snapshot(self):
        t1 = self.store.begin()
        t1.write("k", "gen0")
        t1.commit()

        long_txn = self.store.begin()

        for i in range(1, 50):
            t = self.store.begin()
            t.write("k", "gen%d" % i)
            t.commit()

        # 49 commits later the long transaction still reads gen0.
        self.assertEqual(long_txn.read("k"), "gen0")

        # GC must not collect the version the long transaction can read.
        stats = self.store.gc()
        self.assertEqual(stats["versions_removed"], 0)
        self.assertEqual(long_txn.read("k"), "gen0")

        long_txn.rollback()
        stats = self.store.gc()
        self.assertEqual(stats["versions_removed"], 49)
        self.assertEqual(self.store.version_count("k"), 1)


class GarbageCollectionTest(unittest.TestCase):
    def setUp(self):
        self.store = MVCCStore()

    def test_gc_removes_stale_versions_when_no_readers(self):
        for i in range(10):
            t = self.store.begin()
            t.write("k", i)
            t.commit()
        self.assertEqual(self.store.version_count("k"), 10)
        stats = self.store.gc()
        self.assertEqual(stats["versions_removed"], 9)
        self.assertEqual(self.store.version_count("k"), 1)
        self.assertEqual(self.store.begin().read("k"), 9)

    def test_gc_respects_oldest_active_snapshot(self):
        for i in range(5):
            t = self.store.begin()
            t.write("k", i)
            t.commit()

        reader = self.store.begin()  # snapshot sees value 4

        for i in range(5, 10):
            t = self.store.begin()
            t.write("k", i)
            t.commit()

        self.store.gc()
        # Versions 0..3 are unreadable by anyone; 4..9 must stay.
        self.assertEqual(self.store.version_count("k"), 6)
        self.assertEqual(reader.read("k"), 4)
        reader.rollback()

        self.store.gc()
        self.assertEqual(self.store.version_count("k"), 1)

    def test_gc_removes_fully_deleted_keys(self):
        t1 = self.store.begin()
        t1.write("dead", "x")
        t1.commit()
        t2 = self.store.begin()
        t2.delete("dead")
        t2.commit()
        stats = self.store.gc()
        self.assertEqual(stats["keys_removed"], 1)
        self.assertEqual(self.store.version_count("dead"), 0)

    def test_gc_keeps_tombstone_visible_to_active_snapshot(self):
        t1 = self.store.begin()
        t1.write("k", "v")
        t1.commit()

        old_reader = self.store.begin()  # can still see "v"

        t2 = self.store.begin()
        t2.delete("k")
        t2.commit()

        stats = self.store.gc()
        self.assertEqual(stats["keys_removed"], 0)
        self.assertEqual(old_reader.read("k"), "v")

        old_reader.rollback()
        stats = self.store.gc()
        self.assertEqual(stats["keys_removed"], 1)


class RandomizedTransactionTest(unittest.TestCase):
    def test_many_random_transactions_keep_snapshot_invariants(self):
        """Random workloads must preserve snapshot isolation invariants."""
        rng = random.Random(20261003)
        keys = ["k%d" % i for i in range(6)]

        for _ in range(15):
            store = MVCCStore()
            committed_values = {}  # key -> latest committed value
            open_txns = []

            for _ in range(rng.randint(50, 200)):
                action = rng.choice(
                    ["begin", "read", "write", "delete", "commit", "rollback"]
                )
                if action == "begin" or not open_txns:
                    txn = store.begin()
                    # Expected snapshot: committed state at begin time.
                    open_txns.append((txn, dict(committed_values), {}))
                    continue

                txn, snapshot, writes = rng.choice(open_txns)
                key = rng.choice(keys)

                if action == "read":
                    expected = writes.get(key, snapshot.get(key))
                    self.assertEqual(txn.read(key), expected)
                elif action == "write":
                    value = rng.randint(0, 1000)
                    try:
                        txn.write(key, value)
                        writes[key] = value
                    except WriteConflictError:
                        open_txns = [e for e in open_txns if e[0] is not txn]
                elif action == "delete":
                    try:
                        txn.delete(key)
                        writes[key] = None
                    except WriteConflictError:
                        open_txns = [e for e in open_txns if e[0] is not txn]
                elif action == "commit":
                    open_txns.remove((txn, snapshot, writes))
                    try:
                        txn.commit()
                    except WriteConflictError:
                        continue
                    for k, v in writes.items():
                        committed_values[k] = v
                    # Verify a fresh snapshot sees the committed writes.
                    fresh = store.begin()
                    for k, v in writes.items():
                        self.assertEqual(fresh.read(k), v)
                    fresh.rollback()
                else:  # rollback
                    open_txns.remove((txn, snapshot, writes))
                    txn.rollback()

            for txn, _, _ in open_txns:
                txn.rollback()


class WalPersistenceTest(unittest.TestCase):
    def test_commit_log_survives_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mvcc.log")
            store = MVCCStore(wal_path=path)
            t1 = store.begin()
            t1.write("a", "1")
            t1.write("b", "2")
            t1.commit()
            t2 = store.begin()
            t2.delete("a")
            t2.commit()
            store.close()

            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as handle:
                self.assertEqual(len(handle.read().strip().splitlines()), 3)

            reopened = MVCCStore(wal_path=path)
            reader = reopened.begin()
            self.assertIsNone(reader.read("a"))
            self.assertEqual(reader.read("b"), "2")
            # Commit timestamps continue where the log left off.
            t3 = reopened.begin()
            t3.write("c", "3")
            ts = t3.commit()
            self.assertEqual(ts, 3)
            reopened.close()

    def test_pure_in_memory_mode_writes_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                store = MVCCStore()
                t = store.begin()
                t.write("k", "v")
                t.commit()
            finally:
                os.chdir(cwd)
            self.assertEqual(os.listdir(tmp), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
