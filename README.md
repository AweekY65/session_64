# Local 3D Octree Spatial Index

纯本地三维 Octree 空间索引。所有点云、树节点、查询状态和序列化数据只存在于
内存或本地文件中——不依赖 PCL 服务、数据库、ROS 或任何外部服务，仅用 Python
标准库实现。

## 功能

- 三维点插入，按节点容量（`capacity`）与最大深度（`max_depth`）递归划分为 8 个子空间
- Axis-aligned box range query（闭区间，边界上的点会被返回）
- Radius search（欧氏距离球形邻域）
- Nearest-neighbor 查询（best-first + 包围盒剪枝，非全量扫描）
- 树结构保存 / 重新加载（本地 JSON 文件），加载后查询结果一致
- 正确处理划分平面上的点、空间边界点、完全相同位置的重复点

## 空间划分规则

- 每个节点拥有轴对齐包围盒 `[lo, hi]`，在中心点 `center` 处沿三轴划分为 8 个 octant。
- 点 `p` 在轴 `a` 上满足 `p[a] >= center[a]` 时进入高位子节点（半开区间规则）。
  因此恰好落在划分平面上的点只会进入**唯一**一个子节点——不重复、不遗漏。
- 叶子节点点数超过 `capacity` 且深度小于 `max_depth` 时分裂；达到 `max_depth`
  的叶子不再分裂（可持有超过 `capacity` 的点）。
- 重复点（完全相同坐标）作为独立条目存储，查询时全部返回。

## 剪枝规则

- **Range query**：节点包围盒与查询盒不相交则整棵子树跳过；叶子内逐点做闭区间包含测试。
- **Radius search**：查询球心到节点包围盒的最小距离平方 `> r²` 则跳过该子树
  （包围盒级判断使用 1e-12 容差防止浮点误剪，点级判断为精确比较）。
- **Nearest neighbor**：用优先队列按「查询点到节点包围盒的距离下界」做 best-first
  搜索；当队列中最小下界大于当前最优距离时终止，剩余子树全部被剪枝。

## 复杂度

| 操作 | 平均 | 最坏 |
| --- | --- | --- |
| 插入 | O(log n) | O(max_depth) |
| Range query | O(log n + k) | O(n) |
| Radius search | O(log n + k) | O(n) |
| Nearest neighbor | O(log n) | O(n) |
| 空间 | O(n) | O(n) |

`k` 为返回的点数。点均匀分布时树高为 O(log n)；退化分布（如全部重复点）由
`max_depth` 限制树高，查询退化为对大叶子的线性扫描。

## 使用示例

```python
from octree import Octree

tree = Octree(lo=(0, 0, 0), hi=(100, 100, 100), capacity=8, max_depth=16)
tree.insert((10.0, 20.0, 30.0))

tree.range_query((0, 0, 0), (50, 50, 50))   # AABB 范围查询
tree.radius_search((10, 20, 30), 5.0)       # 半径 5 球形邻域
tree.nearest_neighbor((12, 18, 31))         # -> ((10.0, 20.0, 30.0), dist)

tree.save("tree.json")                      # 保存到本地文件
loaded = Octree.load("tree.json")           # 重新加载，查询结果一致
```

插入根包围盒之外的点会抛出 `ValueError`。

## 运行测试

```bash
python3 -m pytest tests/ -v -s
```

测试覆盖：随机点云（3000 点）range/radius/NN 与 brute-force 对照、划分平面与
空间边界点、重复点、空树、深度限制、序列化 round-trip，以及 NN 剪枝有效性
（断言最近邻搜索访问的点数严格小于全量）。`-s` 选项会在终端显示每项对照的
正确率（`[accuracy] ... 100.00%`）。
