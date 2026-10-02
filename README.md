# pyramid_tool — 本地图像金字塔与切片生成工具

一个纯本地的图像金字塔（image pyramid）与 tile 切片工具。所有输入图片、
缩放层、tile、manifest 和缓存**只存储在本地文件或内存中**，不依赖地图
服务器、CDN、云存储或任何外部服务。

## 功能概览

- 读取常见 PNG / JPEG 输入（经 Pillow 解码，统一转为 RGB）。
- 按配置逐级缩小生成金字塔，直到达到最小尺寸。
- 每级按固定 tile size 切片，边缘采用**裁剪（crop）**规则。
- 支持 `nearest` 与 `bilinear` 两种缩放方式，结果确定（相同输入字节级一致）。
- 生成 `manifest.json`，记录原图尺寸、层级、tile 坐标、尺寸与 SHA-256 哈希。
- 增量重建：输入与配置未变化时不重复生成 tile；损坏或缺失的 tile 自动重新生成。
- 临时文件 + 原子替换：生成中断不会留下被 manifest 误认为有效的半文件。

## 层级算法

- 第 0 级为原图 `(W, H)`。
- 每一级宽高各减半，**奇数尺寸向上取整**：`w' = max(1, ceil(w/2))`，
  保证奇数尺寸的覆盖区域不丢失。
- 当某一级满足 `max(w, h) <= min_size` 时，该级为最后一级。
- 每一级由**原图**直接缩放到该级尺寸（而非逐级串联缩放），避免误差累积，
  并保证结果只取决于输入与配置。

示例：`101x77`、`min_size=8` → `(101,77) (51,39) (26,20) (13,10) (7,5)`。

## 坐标与边缘规则

- tile 以 `(level, x, y)` 寻址，原点为左上角，`x` 向右、`y` 向下增长。
- 每级网格：`cols = ceil(w / tile_size)`，`rows = ceil(h / tile_size)`。
- **边缘规则：裁剪，不 padding**。tile 文件恰好包含其区域像素，右边缘 /
  下边缘的 tile 尺寸可能小于 `tile_size`（如 300 宽、tile 128 时，最右列
  tile 宽为 44）。不生成任何填充像素。
- tile 以无损 PNG 存储，路径为 `tiles/<level>/<x>/<y>.png`。

## 缩放方式与稳定性

- `nearest`：对齐规则为 align-corners=False，输出像素 `(i,j)` 取自源像素
  `floor((i+0.5) * src/dst)`（越界截断）。
- `bilinear`：同一映射下做双线性插值，float64 累加，最后 `floor(v+0.5)`
  四舍五入到 uint8。
- 两种算法均在本地用 numpy 实现，舍入规则固定；tile 用无损 PNG 编码，
  因此**相同输入 + 相同配置必然得到字节级一致的结果**。

## manifest

`manifest.json` 结构：

```json
{
  "version": 1,
  "edge_rule": "crop",
  "input":  {"path": "...", "sha256": "...", "width": W, "height": H},
  "config": {"tile_size": 256, "resample": "bilinear", "min_size": 256},
  "levels": [
    {"level": 0, "width": W, "height": H, "cols": C, "rows": R,
     "tiles": [{"x": 0, "y": 0, "width": 256, "height": 256,
                "file": "tiles/0/0/0.png", "sha256": "..."}]}
  ]
}
```

## 增量重建

构建时读取已有 manifest：若输入文件 SHA-256 与配置均一致，则逐个检查
tile——文件存在且哈希与 manifest 记录相符则**跳过**；缺失或哈希不符
（损坏）的 tile 会被重新生成。输入变化或配置变化会触发全量重建。

## 原子性

tile 与 manifest 均先写入同目录的 `*.tmp-*` 临时文件，`fsync` 后用
`os.replace` 原子替换。生成中断只会留下：完整的旧文件、完整的新文件，
或不被 manifest 引用的临时文件（下次构建时自动清理）——绝不会留下被
manifest 误认为有效的半文件。manifest 最后写入，因此中断时旧 manifest
保持原样，其引用的 tile 不会被误认为有效。

## 使用

```bash
# 构建（或增量重建）
python -m pyramid_tool input.jpg out_dir --tile-size 256 --resample bilinear --min-size 256

# 校验已有金字塔与 manifest 的一致性
python -m pyramid_tool --verify out_dir
```

Python API：

```python
from pyramid_tool import BuildConfig, build_pyramid, verify_pyramid

stats = build_pyramid("input.png", "out_dir", BuildConfig(tile_size=256, resample="nearest"))
print(stats.tiles_written, stats.tiles_reused)
problems = verify_pyramid("out_dir")  # [] 表示一致
```

## 测试

测试在终端运行并输出校验结果，不会打开任何图片窗口。测试图像全部由
代码生成，覆盖：奇数尺寸、边缘 tile 裁剪、nearest/bilinear 两种缩放、
结果确定性、增量复用、tile 损坏 / 缺失重生成、中途失败的原子性等场景。

```bash
python -m unittest discover -s tests -v
```

## 依赖

- Python >= 3.9
- Pillow
- numpy
