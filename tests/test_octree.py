"""Automated tests for the local 3D Octree.

Every query result is cross-checked against a brute-force reference
implementation and the accuracy is printed to the terminal.
"""

import math
import os
import random
import tempfile

import pytest

from octree import Octree

BOX_LO = (0.0, 0.0, 0.0)
BOX_HI = (100.0, 100.0, 100.0)


# --------------------------------------------------------------------- #
# brute-force reference
# --------------------------------------------------------------------- #
def bf_range(points, lo, hi):
    return [p for p in points if all(lo[a] <= p[a] <= hi[a] for a in range(3))]


def bf_radius(points, c, r):
    return [p for p in points if math.dist(p, c) <= r + 1e-12]


def bf_nn(points, q):
    if not points:
        return None
    best = min(points, key=lambda p: math.dist(p, q))
    return best, math.dist(best, q)


def as_multiset(points):
    return sorted(tuple(p) for p in points)


def report(name, correct, total):
    pct = 100.0 * correct / total if total else 100.0
    print(f"\n[accuracy] {name}: {correct}/{total} correct ({pct:.2f}%)")
    return pct


def random_points(rng, n, lo=BOX_LO, hi=BOX_HI):
    return [
        tuple(rng.uniform(lo[a], hi[a]) for a in range(3))
        for _ in range(n)
    ]


# --------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------- #
@pytest.fixture()
def rng():
    return random.Random(20241002)


@pytest.fixture()
def cloud(rng):
    pts = random_points(rng, 3000)
    tree = Octree(BOX_LO, BOX_HI, capacity=8, max_depth=12)
    for p in pts:
        tree.insert(p)
    return pts, tree


# --------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------- #
def test_insert_and_size(cloud):
    pts, tree = cloud
    assert len(tree) == len(pts)
    assert as_multiset(iter(tree)) == as_multiset(pts)


def test_range_query_vs_bruteforce(cloud, rng):
    pts, tree = cloud
    trials, correct = 200, 0
    for _ in range(trials):
        a = random_points(rng, 1)[0]
        b = random_points(rng, 1)[0]
        lo = tuple(min(a[i], b[i]) for i in range(3))
        hi = tuple(max(a[i], b[i]) for i in range(3))
        got = tree.range_query(lo, hi)
        want = bf_range(pts, lo, hi)
        if as_multiset(got) == as_multiset(want):
            correct += 1
    assert report("range_query", correct, trials) == 100.0


def test_radius_search_vs_bruteforce(cloud, rng):
    pts, tree = cloud
    trials, correct = 200, 0
    for _ in range(trials):
        c = random_points(rng, 1)[0]
        r = rng.uniform(0.0, 30.0)
        got = tree.radius_search(c, r)
        want = bf_radius(pts, c, r)
        if as_multiset(got) == as_multiset(want):
            correct += 1
    assert report("radius_search", correct, trials) == 100.0


def test_nearest_neighbor_vs_bruteforce(cloud, rng):
    pts, tree = cloud
    trials, correct = 200, 0
    for _ in range(trials):
        q = random_points(rng, 1)[0]
        got = tree.nearest_neighbor(q)
        want = bf_nn(pts, q)
        assert got is not None and want is not None
        # distances must match; point itself may differ on exact ties
        if abs(got[1] - want[1]) <= 1e-9:
            correct += 1
    assert report("nearest_neighbor", correct, trials) == 100.0


def test_boundary_points_on_split_planes():
    # Points exactly on the root split plane (50,50,50) and on box faces.
    pts = [
        (50.0, 50.0, 50.0),
        (50.0, 10.0, 90.0),
        (10.0, 50.0, 90.0),
        (10.0, 90.0, 50.0),
        (0.0, 0.0, 0.0),
        (100.0, 100.0, 100.0),
        (0.0, 50.0, 100.0),
        (25.0, 25.0, 25.0),
        (75.0, 75.0, 75.0),
    ]
    tree = Octree(BOX_LO, BOX_HI, capacity=1, max_depth=8)
    for p in pts:
        tree.insert(p)
    # full-box query must return every point exactly once
    got = tree.range_query(BOX_LO, BOX_HI)
    assert as_multiset(got) == as_multiset(pts)
    # query box whose face lies exactly on the split plane
    got = tree.range_query((50.0, 50.0, 50.0), BOX_HI)
    want = bf_range(pts, (50.0, 50.0, 50.0), BOX_HI)
    assert as_multiset(got) == as_multiset(want)
    # radius search centered exactly on a split-plane point
    got = tree.radius_search((50.0, 50.0, 50.0), 0.0)
    assert got == [(50.0, 50.0, 50.0)]
    print("\n[accuracy] boundary_points: all checks passed (100.00%)")


def test_duplicate_points():
    pts = [(1.0, 2.0, 3.0)] * 50 + [(1.0, 2.0, 3.0000001)] * 5
    tree = Octree(BOX_LO, BOX_HI, capacity=4, max_depth=10)
    for p in pts:
        tree.insert(p)
    assert len(tree) == 55
    got = tree.range_query((0.0, 0.0, 0.0), (2.0, 3.0, 4.0))
    assert as_multiset(got) == as_multiset(pts)
    got = tree.radius_search((1.0, 2.0, 3.0), 0.0)
    assert len(got) == 50
    p, d = tree.nearest_neighbor((1.0, 2.0, 3.0))
    assert d == pytest.approx(0.0)
    assert p == (1.0, 2.0, 3.0)
    print("\n[accuracy] duplicate_points: all checks passed (100.00%)")


def test_empty_tree():
    tree = Octree(BOX_LO, BOX_HI)
    assert len(tree) == 0
    assert tree.range_query(BOX_LO, BOX_HI) == []
    assert tree.radius_search((1.0, 1.0, 1.0), 5.0) == []
    assert tree.nearest_neighbor((1.0, 1.0, 1.0)) is None
    print("\n[accuracy] empty_tree: all checks passed (100.00%)")


def test_max_depth_limit(rng):
    pts = random_points(rng, 500)
    tree = Octree(BOX_LO, BOX_HI, capacity=1, max_depth=2)
    for p in pts:
        tree.insert(p)

    def max_node_depth(node):
        if node.is_leaf:
            return node.depth
        return max(max_node_depth(c) for c in node.children)

    assert max_node_depth(tree.root) <= 2
    # queries still agree with brute force even with oversized leaves
    lo, hi = (10.0, 10.0, 10.0), (60.0, 60.0, 60.0)
    assert as_multiset(tree.range_query(lo, hi)) == as_multiset(bf_range(pts, lo, hi))
    c, r = (50.0, 50.0, 50.0), 25.0
    assert as_multiset(tree.radius_search(c, r)) == as_multiset(bf_radius(pts, c, r))
    got, want = tree.nearest_neighbor(c), bf_nn(pts, c)
    assert got[1] == pytest.approx(want[1], abs=1e-9)
    print("\n[accuracy] max_depth_limit: all checks passed (100.00%)")


def test_save_and_load_roundtrip(cloud, rng):
    pts, tree = cloud
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tree.json")
        tree.save(path)
        loaded = Octree.load(path)
    assert len(loaded) == len(tree)
    trials, correct = 100, 0
    for _ in range(trials):
        q = random_points(rng, 1)[0]
        r = rng.uniform(0.0, 30.0)
        lo = tuple(v - 10 for v in q)
        hi = tuple(v + 10 for v in q)
        same = (
            as_multiset(loaded.range_query(lo, hi)) == as_multiset(tree.range_query(lo, hi))
            and as_multiset(loaded.radius_search(q, r)) == as_multiset(tree.radius_search(q, r))
            and abs(loaded.nearest_neighbor(q)[1] - tree.nearest_neighbor(q)[1]) <= 1e-9
        )
        # and both must match brute force
        same = same and as_multiset(loaded.range_query(lo, hi)) == as_multiset(
            bf_range(pts, lo, hi)
        )
        if same:
            correct += 1
    assert report("save_load_roundtrip", correct, trials) == 100.0


def test_insert_outside_root_box_raises():
    tree = Octree(BOX_LO, BOX_HI)
    with pytest.raises(ValueError):
        tree.insert((-1.0, 50.0, 50.0))
    with pytest.raises(ValueError):
        tree.insert((50.0, 50.0, 100.1))


def test_nn_uses_pruning_not_full_scan(cloud):
    """NN must not visit every point: bounding-box pruning is required."""
    pts, tree = cloud
    visited = 0
    orig_dist2 = None
    import octree.octree as mod

    orig_dist2 = mod._dist2

    def counting_dist2(a, b):
        nonlocal visited
        visited += 1
        return orig_dist2(a, b)

    mod._dist2 = counting_dist2
    try:
        tree.nearest_neighbor((50.0, 50.0, 50.0))
    finally:
        mod._dist2 = orig_dist2
    print(f"\n[pruning] NN visited {visited}/{len(pts)} points")
    assert 0 < visited < len(pts)
