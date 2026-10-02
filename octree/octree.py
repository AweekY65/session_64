"""Pure-local 3D Octree spatial index.

Everything lives in memory or in local files (JSON). No PCL, ROS,
databases, network, or any external service is involved.

Conventions
-----------
* A point is a ``(x, y, z)`` tuple of floats. Duplicate points are
  allowed and stored independently.
* Each node owns an axis-aligned bounding box ``[lo, hi]`` per axis.
* Subdivision splits a node at its center. A point goes to the high
  child on axis ``a`` iff ``p[a] >= center[a]``. This half-open rule
  guarantees points exactly on a split plane land in exactly one
  child -- never duplicated, never lost.
* Range/box queries use inclusive bounds on all sides, so points on
  the query boundary are returned.
"""

from __future__ import annotations

import heapq
import json
import math
from typing import Iterator, List, Optional, Sequence, Tuple

Point3 = Tuple[float, float, float]

_EPS = 1e-12


def _as_point(p: Sequence[float]) -> Point3:
    if len(p) != 3:
        raise ValueError(f"point must have 3 coordinates, got {p!r}")
    return (float(p[0]), float(p[1]), float(p[2]))


def _dist2(a: Point3, b: Point3) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _box_min_dist2(lo: Point3, hi: Point3, p: Point3) -> float:
    """Squared distance from point ``p`` to the AABB [lo, hi] (0 if inside)."""
    d2 = 0.0
    for i in range(3):
        if p[i] < lo[i]:
            d2 += (lo[i] - p[i]) ** 2
        elif p[i] > hi[i]:
            d2 += (p[i] - hi[i]) ** 2
    return d2


def _boxes_intersect(
    alo: Point3, ahi: Point3, blo: Point3, bhi: Point3
) -> bool:
    for i in range(3):
        if ahi[i] < blo[i] or bhi[i] < alo[i]:
            return False
    return True


class OctreeNode:
    """One node of the octree. Internal nodes have 8 children, leaves hold points."""

    __slots__ = ("lo", "hi", "depth", "points", "children")

    def __init__(self, lo: Point3, hi: Point3, depth: int = 0) -> None:
        self.lo: Point3 = lo
        self.hi: Point3 = hi
        self.depth: int = depth
        self.points: List[Point3] = []
        self.children: Optional[List["OctreeNode"]] = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    def center(self) -> Point3:
        return (
            (self.lo[0] + self.hi[0]) * 0.5,
            (self.lo[1] + self.hi[1]) * 0.5,
            (self.lo[2] + self.hi[2]) * 0.5,
        )

    def octant_index(self, p: Point3, center: Point3) -> int:
        idx = 0
        for axis in range(3):
            if p[axis] >= center[axis]:
                idx |= 1 << axis
        return idx

    def subdivide(self) -> None:
        lo, hi = self.lo, self.hi
        c = self.center()
        self.children = []
        for idx in range(8):
            clo = tuple(c[a] if (idx >> a) & 1 else lo[a] for a in range(3))
            chi = tuple(hi[a] if (idx >> a) & 1 else c[a] for a in range(3))
            self.children.append(OctreeNode(clo, chi, self.depth + 1))  # type: ignore[arg-type]

    def child_for(self, p: Point3) -> "OctreeNode":
        assert self.children is not None
        return self.children[self.octant_index(p, self.center())]

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        data = {
            "lo": list(self.lo),
            "hi": list(self.hi),
            "depth": self.depth,
            "points": [list(p) for p in self.points],
        }
        if self.children is not None:
            data["children"] = [ch.to_dict() for ch in self.children]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "OctreeNode":
        node = cls(_as_point(data["lo"]), _as_point(data["hi"]), int(data["depth"]))
        node.points = [_as_point(p) for p in data["points"]]
        if "children" in data:
            node.children = [cls.from_dict(ch) for ch in data["children"]]
        return node


class Octree:
    """3D Octree over a fixed axis-aligned root region.

    Parameters
    ----------
    lo, hi:
        Root bounding box corners. Points inserted outside this box
        raise ``ValueError``.
    capacity:
        Maximum number of points a leaf holds before it subdivides.
    max_depth:
        Maximum tree depth. Nodes at this depth never subdivide, so
        they may hold more than ``capacity`` points.
    """

    def __init__(
        self,
        lo: Sequence[float] = (0.0, 0.0, 0.0),
        hi: Sequence[float] = (1.0, 1.0, 1.0),
        capacity: int = 8,
        max_depth: int = 16,
    ) -> None:
        lo_t, hi_t = _as_point(lo), _as_point(hi)
        for a in range(3):
            if not lo_t[a] < hi_t[a]:
                raise ValueError("root box must have positive extent on every axis")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        self.capacity = capacity
        self.max_depth = max_depth
        self.root = OctreeNode(lo_t, hi_t, 0)
        self._size = 0

    # ------------------------------------------------------------------ #
    # basic API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._size

    def _contains(self, p: Point3) -> bool:
        lo, hi = self.root.lo, self.root.hi
        return all(lo[a] - _EPS <= p[a] <= hi[a] + _EPS for a in range(3))

    def insert(self, point: Sequence[float]) -> None:
        p = _as_point(point)
        if not self._contains(p):
            raise ValueError(f"point {p} outside root box {self.root.lo}..{self.root.hi}")
        node = self.root
        while True:
            if node.is_leaf:
                node.points.append(p)
                if len(node.points) > self.capacity and node.depth < self.max_depth:
                    self._split(node)
                break
            node = node.child_for(p)
        self._size += 1

    def _split(self, node: OctreeNode) -> None:
        node.subdivide()
        old, node.points = node.points, []
        for p in old:
            node.child_for(p).points.append(p)

    def __iter__(self) -> Iterator[Point3]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                yield from node.points
            else:
                stack.extend(node.children)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def range_query(self, qlo: Sequence[float], qhi: Sequence[float]) -> List[Point3]:
        """All points inside the inclusive AABB [qlo, qhi]."""
        lo, hi = _as_point(qlo), _as_point(qhi)
        for a in range(3):
            if lo[a] > hi[a]:
                raise ValueError("query box lo must be <= hi on every axis")
        out: List[Point3] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not _boxes_intersect(node.lo, node.hi, lo, hi):
                continue
            if node.is_leaf:
                for p in node.points:
                    if all(lo[a] <= p[a] <= hi[a] for a in range(3)):
                        out.append(p)
            else:
                stack.extend(node.children)  # type: ignore[arg-type]
        return out

    def radius_search(self, center: Sequence[float], radius: float) -> List[Point3]:
        """All points within Euclidean distance ``radius`` of ``center``."""
        c = _as_point(center)
        if radius < 0:
            raise ValueError("radius must be >= 0")
        r2 = radius * radius
        out: List[Point3] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if _box_min_dist2(node.lo, node.hi, c) > r2 + _EPS:
                continue
            if node.is_leaf:
                for p in node.points:
                    if _dist2(p, c) <= r2:
                        out.append(p)
            else:
                stack.extend(node.children)  # type: ignore[arg-type]
        return out

    def nearest_neighbor(
        self, query: Sequence[float]
    ) -> Optional[Tuple[Point3, float]]:
        """Nearest point to ``query`` using best-first search with
        bounding-box pruning. Returns ``(point, distance)`` or ``None``
        for an empty tree."""
        q = _as_point(query)
        if self._size == 0:
            return None
        best_p: Optional[Point3] = None
        best_d2 = math.inf
        # Priority queue keyed by lower-bound distance to a node's box.
        heap: List[Tuple[float, int, OctreeNode]] = []
        counter = 0
        heapq.heappush(heap, (_box_min_dist2(self.root.lo, self.root.hi, q), 0, self.root))
        while heap:
            lb, _, node = heapq.heappop(heap)
            if lb > best_d2 + _EPS:
                break  # everything left in the queue is farther than best
            if node.is_leaf:
                for p in node.points:
                    d2 = _dist2(p, q)
                    if d2 < best_d2 - _EPS:
                        best_d2 = d2
                        best_p = p
            else:
                for child in node.children or []:
                    counter += 1
                    heapq.heappush(
                        heap, (_box_min_dist2(child.lo, child.hi, q), counter, child)
                    )
        if best_p is None:
            return None
        return best_p, math.sqrt(best_d2)

    # ------------------------------------------------------------------ #
    # persistence (local files only)
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Serialize the full tree structure to a local JSON file."""
        payload = {
            "capacity": self.capacity,
            "max_depth": self.max_depth,
            "size": self._size,
            "root": self.root.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "Octree":
        """Rebuild a tree from a local JSON file saved with :meth:`save`."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        tree = cls.__new__(cls)
        tree.capacity = int(payload["capacity"])
        tree.max_depth = int(payload["max_depth"])
        tree.root = OctreeNode.from_dict(payload["root"])
        tree._size = int(payload["size"])
        return tree
