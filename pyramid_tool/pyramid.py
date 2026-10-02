"""Core logic for the local image pyramid / tile generator.

Everything lives on the local filesystem or in memory. No map server, CDN,
cloud storage or any other external service is used.

Coordinate / tiling rules
-------------------------
* Level 0 is the original image. Each following level halves both
  dimensions (rounding up for odd sizes) until ``max(width, height)``
  is <= ``min_size``.
* Tiles are addressed as ``(level, x, y)`` with the origin at the
  top-left corner. ``x`` grows to the right, ``y`` grows downwards.
* Edge handling rule: **crop, never pad**. A tile always contains exactly
  the pixels of its region, so tiles on the right / bottom edge may be
  smaller than ``tile_size``. No padding pixels are ever invented.
* Tiles are stored as PNG files at ``tiles/<level>/<x>/<y>.png``.

Determinism
-----------
Resampling (nearest / bilinear) is implemented locally in numpy with a
fixed "align corners = False" mapping, and tiles are encoded as PNG
(lossless), so the same input always produces byte-identical output.

Atomicity
---------
Tiles and ``manifest.json`` are written to a temporary file in the same
directory and then moved into place with ``os.replace`` (atomic on POSIX
and Windows). A crash can therefore only leave a complete old file, a
complete new file, or an unreferenced ``*.tmp-*`` file -- never a half
file that the manifest would consider valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

MANIFEST_NAME = "manifest.json"
TILES_DIRNAME = "tiles"
RESAMPLE_MODES = ("nearest", "bilinear")
MANIFEST_VERSION = 1


# ---------------------------------------------------------------------------
# Configuration / statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for one pyramid build."""

    tile_size: int = 256
    resample: str = "bilinear"
    min_size: int = 256

    def __post_init__(self) -> None:
        if self.tile_size < 1:
            raise ValueError("tile_size must be >= 1")
        if self.min_size < 1:
            raise ValueError("min_size must be >= 1")
        if self.resample not in RESAMPLE_MODES:
            raise ValueError(
                f"resample must be one of {RESAMPLE_MODES}, got {self.resample!r}"
            )

    def as_dict(self) -> dict:
        return {
            "tile_size": self.tile_size,
            "resample": self.resample,
            "min_size": self.min_size,
        }


@dataclass
class BuildStats:
    """Summary of a build run."""

    levels: int = 0
    tiles_total: int = 0
    tiles_written: int = 0
    tiles_reused: int = 0
    manifest_path: str = ""


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------


def compute_level_dims(width: int, height: int, min_size: int) -> List[Tuple[int, int]]:
    """Return ``[(w0, h0), (w1, h1), ...]`` starting at the original size.

    Each level halves both dimensions, rounding up so odd sizes stay
    covered, and stops once ``max(w, h) <= min_size`` (that level is the
    last one included).
    """
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be >= 1")
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    dims = [(width, height)]
    while max(dims[-1]) > min_size:
        w, h = dims[-1]
        dims.append((max(1, (w + 1) // 2), max(1, (h + 1) // 2)))
    return dims


def compute_tile_grid(width: int, height: int, tile_size: int) -> Tuple[int, int]:
    """Return ``(cols, rows)`` of tiles needed to cover ``width x height``."""
    if tile_size < 1:
        raise ValueError("tile_size must be >= 1")
    cols = (width + tile_size - 1) // tile_size
    rows = (height + tile_size - 1) // tile_size
    return cols, rows


def tile_region(width: int, height: int, tile_size: int, x: int, y: int) -> Tuple[int, int, int, int]:
    """Return ``(left, top, tw, th)`` for tile ``(x, y)`` (crop rule)."""
    left = x * tile_size
    top = y * tile_size
    tw = min(tile_size, width - left)
    th = min(tile_size, height - top)
    if tw <= 0 or th <= 0:
        raise ValueError(f"tile ({x}, {y}) is outside a {width}x{height} level")
    return left, top, tw, th


# ---------------------------------------------------------------------------
# Deterministic resampling (numpy, fixed rounding rules)
# ---------------------------------------------------------------------------


def _resize_nearest(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    src_h, src_w = arr.shape[:2]
    ys = np.minimum(
        ((np.arange(out_h, dtype=np.float64) + 0.5) * src_h / out_h).astype(np.int64),
        src_h - 1,
    )
    xs = np.minimum(
        ((np.arange(out_w, dtype=np.float64) + 0.5) * src_w / out_w).astype(np.int64),
        src_w - 1,
    )
    return arr[ys][:, xs]


def _resize_axis(img: np.ndarray, out_n: int, axis: int) -> np.ndarray:
    src_n = img.shape[axis]
    pos = (np.arange(out_n, dtype=np.float64) + 0.5) * (src_n / out_n) - 0.5
    pos = np.clip(pos, 0.0, src_n - 1.0)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, src_n - 1)
    w1 = pos - i0
    w0 = 1.0 - w1
    if axis == 0:
        return img[i0] * w0[:, None, None] + img[i1] * w1[:, None, None]
    return img[:, i0] * w0[None, :, None] + img[:, i1] * w1[None, :, None]


def _resize_bilinear(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    img = arr.astype(np.float64)
    img = _resize_axis(img, out_h, axis=0)
    img = _resize_axis(img, out_w, axis=1)
    return np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8)


def resize_image(arr: np.ndarray, out_w: int, out_h: int, resample: str) -> np.ndarray:
    """Resize an HxWxC uint8 array deterministically."""
    if arr.shape[1] == out_w and arr.shape[0] == out_h:
        return arr.copy()
    if resample == "nearest":
        return _resize_nearest(arr, out_w, out_h)
    if resample == "bilinear":
        return _resize_bilinear(arr, out_w, out_h)
    raise ValueError(f"unknown resample mode: {resample!r}")


# ---------------------------------------------------------------------------
# Hashing / atomic IO
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".tmp-", suffix=".part"
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _encode_tile_png(tile: np.ndarray) -> bytes:
    """Encode a tile array as PNG bytes (lossless, deterministic)."""
    import io

    buf = io.BytesIO()
    Image.fromarray(tile).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _tile_relpath(level: int, x: int, y: int) -> str:
    return f"{TILES_DIRNAME}/{level}/{x}/{y}.png"


def _load_reusable_manifest(output_dir: Path, input_hash: str, config: BuildConfig) -> Dict[Tuple[int, int, int], dict]:
    """Return {(level, x, y): tile_entry} from a previous compatible manifest."""
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            old = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if old.get("version") != MANIFEST_VERSION:
        return {}
    if old.get("input", {}).get("sha256") != input_hash:
        return {}
    if old.get("config") != config.as_dict():
        return {}
    reusable: Dict[Tuple[int, int, int], dict] = {}
    for lvl in old.get("levels", []):
        level = lvl.get("level")
        for tile in lvl.get("tiles", []):
            reusable[(level, tile["x"], tile["y"])] = tile
    return reusable


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_pyramid(input_path: os.PathLike | str, output_dir: os.PathLike | str, config: BuildConfig | None = None) -> BuildStats:
    """Build (or incrementally rebuild) a local image pyramid.

    Incremental behaviour: if the previous manifest matches the current
    input hash and config, tiles whose file still exists and still hashes
    to the recorded value are left untouched. Missing or corrupted tiles
    are regenerated.
    """
    config = config or BuildConfig()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_bytes = input_path.read_bytes()
    input_hash = sha256_bytes(input_bytes)

    import io

    with Image.open(io.BytesIO(input_bytes)) as im:
        rgb = im.convert("RGB")
        src = np.asarray(rgb, dtype=np.uint8).copy()
    src_h, src_w = src.shape[:2]

    level_dims = compute_level_dims(src_w, src_h, config.min_size)
    reusable = _load_reusable_manifest(output_dir, input_hash, config)

    stats = BuildStats(levels=len(level_dims))
    manifest_levels: List[dict] = []

    for level, (lw, lh) in enumerate(level_dims):
        if level == 0:
            level_arr = src
        else:
            level_arr = resize_image(src, lw, lh, config.resample)

        cols, rows = compute_tile_grid(lw, lh, config.tile_size)
        tile_entries: List[dict] = []

        for y in range(rows):
            for x in range(cols):
                stats.tiles_total += 1
                rel = _tile_relpath(level, x, y)
                abs_path = output_dir / rel
                old_entry = reusable.get((level, x, y))

                if (
                    old_entry is not None
                    and old_entry.get("file") == rel
                    and abs_path.is_file()
                    and sha256_file(abs_path) == old_entry.get("sha256")
                ):
                    stats.tiles_reused += 1
                    tile_entries.append(dict(old_entry))
                    continue

                left, top, tw, th = tile_region(lw, lh, config.tile_size, x, y)
                tile = level_arr[top : top + th, left : left + tw]
                png_bytes = _encode_tile_png(tile)
                digest = sha256_bytes(png_bytes)
                _atomic_write_bytes(abs_path, png_bytes)
                stats.tiles_written += 1
                tile_entries.append(
                    {
                        "x": x,
                        "y": y,
                        "width": tw,
                        "height": th,
                        "file": rel,
                        "sha256": digest,
                    }
                )

        manifest_levels.append(
            {
                "level": level,
                "width": lw,
                "height": lh,
                "cols": cols,
                "rows": rows,
                "tiles": tile_entries,
            }
        )

    manifest = {
        "version": MANIFEST_VERSION,
        "generator": "pyramid_tool (local only, no external services)",
        "edge_rule": "crop",
        "input": {
            "path": str(input_path),
            "sha256": input_hash,
            "width": src_w,
            "height": src_h,
        },
        "config": config.as_dict(),
        "levels": manifest_levels,
    }
    manifest_path = output_dir / MANIFEST_NAME
    _atomic_write_bytes(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8"),
    )
    stats.manifest_path = str(manifest_path)

    _cleanup_stale_temp_files(output_dir)
    return stats


def _cleanup_stale_temp_files(output_dir: Path) -> None:
    """Remove leftover ``*.tmp-*`` files from interrupted previous runs."""
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            if ".tmp-" in name:
                try:
                    os.unlink(Path(root) / name)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_pyramid(output_dir: os.PathLike | str) -> List[str]:
    """Check that every tile referenced by the manifest exists and matches
    its recorded hash. Returns a list of human-readable problems (empty
    means the pyramid is fully consistent)."""
    output_dir = Path(output_dir)
    problems: List[str] = []
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unreadable manifest: {exc}"]

    for lvl in manifest.get("levels", []):
        level = lvl.get("level")
        for tile in lvl.get("tiles", []):
            rel = tile.get("file", "")
            path = output_dir / rel
            tag = f"level {level} tile ({tile.get('x')},{tile.get('y')})"
            if not path.is_file():
                problems.append(f"{tag}: missing file {rel}")
                continue
            actual = sha256_file(path)
            if actual != tile.get("sha256"):
                problems.append(f"{tag}: hash mismatch for {rel}")
    return problems
