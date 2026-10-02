"""Pure-local 3D Octree spatial index.

No PCL, ROS, databases, or external services. Points, tree nodes,
query state, and serialized data live only in memory or local files.
"""

from .octree import Octree, OctreeNode, Point3

__all__ = ["Octree", "OctreeNode", "Point3"]
