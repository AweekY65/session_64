"""Automated tests for pyramid_tool.

All tests generate small synthetic images in code, run entirely in the
terminal and print check results via the unittest runner. No image
windows are ever opened.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyramid_tool import (  # noqa: E402
    BuildConfig,
    build_pyramid,
    compute_level_dims,
    compute_tile_grid,
    resize_image,
    tile_region,
    verify_pyramid,
)
from pyramid_tool import pyramid as pyramid_mod  # noqa: E402


def make_gradient_array(width: int, height: int) -> np.ndarray:
    """Deterministic non-uniform RGB test pattern."""
    yy, xx = np.mgrid[0:height, 0:width]
    arr = np.stack(
        [
            (xx * 7 + yy * 3) % 256,
            (xx * 5 + yy * 11) % 256,
            (xx * 13 + yy * 2) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    return arr


def save_image(arr: np.ndarray, path: Path, fmt: str = "PNG") -> None:
    Image.fromarray(arr).save(path, format=fmt)


def read_manifest(out_dir: Path) -> dict:
    with open(out_dir / "manifest.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


class TempWorkspaceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work = Path(self._tmp.name)
        self.out = self.work / "pyramid"

    def make_input(self, width: int, height: int, fmt: str = "PNG", name: str = "input.png") -> Path:
        path = self.work / name
        save_image(make_gradient_array(width, height), path, fmt)
        return path


class TestLevelAlgorithm(unittest.TestCase):
    def test_odd_size_levels_round_up(self):
        dims = compute_level_dims(101, 77, min_size=8)
        self.assertEqual(dims[0], (101, 77))
        self.assertEqual(dims[1], (51, 39))
        self.assertEqual(dims[2], (26, 20))
        self.assertEqual(dims[3], (13, 10))
        self.assertEqual(dims[4], (7, 5))
        self.assertEqual(len(dims), 5)  # stops once max(w,h) <= min_size

    def test_exact_power_of_two(self):
        dims = compute_level_dims(256, 256, min_size=64)
        self.assertEqual(dims, [(256, 256), (128, 128), (64, 64)])

    def test_tile_grid(self):
        self.assertEqual(compute_tile_grid(256, 256, 256), (1, 1))
        self.assertEqual(compute_tile_grid(300, 200, 128), (3, 2))
        self.assertEqual(compute_tile_grid(1, 1, 256), (1, 1))

    def test_tile_region_crop_rule(self):
        # right edge tile is cropped, not padded
        self.assertEqual(tile_region(300, 200, 128, 2, 0), (256, 0, 44, 128))
        # bottom edge
        self.assertEqual(tile_region(300, 200, 128, 0, 1), (0, 128, 128, 72))
        # corner
        self.assertEqual(tile_region(300, 200, 128, 2, 1), (256, 128, 44, 72))
        with self.assertRaises(ValueError):
            tile_region(300, 200, 128, 3, 0)


class TestResampling(unittest.TestCase):
    def test_nearest_picks_exact_source_pixels(self):
        arr = make_gradient_array(8, 8)
        out = resize_image(arr, 4, 4, "nearest")
        # align-corners=False mapping: out(i,j) = src at (2i+1, 2j+1)
        expected = arr[1::2, 1::2]
        np.testing.assert_array_equal(out, expected)

    def test_bilinear_averages_for_exact_2x(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[..., 0] = np.arange(16, dtype=np.uint8).reshape(4, 4) * 10
        out = resize_image(arr, 2, 2, "bilinear")
        # each output pixel is the mean of the corresponding 2x2 block
        for oy in range(2):
            for ox in range(2):
                block = arr[2 * oy : 2 * oy + 2, 2 * ox : 2 * ox + 2, 0].astype(np.float64)
                self.assertEqual(int(out[oy, ox, 0]), int(np.floor(block.mean() + 0.5)))

    def test_modes_differ_on_gradient(self):
        arr = make_gradient_array(64, 64)
        near = resize_image(arr, 32, 32, "nearest")
        bil = resize_image(arr, 32, 32, "bilinear")
        self.assertFalse(np.array_equal(near, bil))

    def test_determinism(self):
        arr = make_gradient_array(63, 45)
        for mode in ("nearest", "bilinear"):
            a = resize_image(arr, 17, 11, mode)
            b = resize_image(arr, 17, 11, mode)
            np.testing.assert_array_equal(a, b)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            resize_image(make_gradient_array(4, 4), 2, 2, "cubic")
        with self.assertRaises(ValueError):
            BuildConfig(resample="cubic")


class TestBuild(TempWorkspaceTestCase):
    def test_manifest_content_and_tiles(self):
        src = self.make_input(300, 200)
        config = BuildConfig(tile_size=128, min_size=64)
        stats = build_pyramid(src, self.out, config)

        self.assertEqual(stats.levels, 4)  # 300x200, 150x100, 75x50, 38x25
        manifest = read_manifest(self.out)
        self.assertEqual(manifest["input"]["width"], 300)
        self.assertEqual(manifest["input"]["height"], 200)
        self.assertEqual(manifest["config"], config.as_dict())
        self.assertEqual(manifest["edge_rule"], "crop")

        level0 = manifest["levels"][0]
        self.assertEqual((level0["cols"], level0["rows"]), (3, 2))
        # corner tile cropped to 44x72
        corner = [t for t in level0["tiles"] if t["x"] == 2 and t["y"] == 1][0]
        self.assertEqual((corner["width"], corner["height"]), (44, 72))

        # every tile file exists, matches its hash, and has the recorded size
        self.assertEqual(verify_pyramid(self.out), [])
        for lvl in manifest["levels"]:
            for t in lvl["tiles"]:
                with Image.open(self.out / t["file"]) as im:
                    self.assertEqual(im.size, (t["width"], t["height"]))

    def test_jpeg_input_supported(self):
        src = self.make_input(120, 90, fmt="JPEG", name="input.jpg")
        stats = build_pyramid(src, self.out, BuildConfig(tile_size=64, min_size=32))
        self.assertGreaterEqual(stats.levels, 2)
        self.assertEqual(verify_pyramid(self.out), [])

    def test_odd_size_image_build(self):
        src = self.make_input(101, 77)
        stats = build_pyramid(src, self.out, BuildConfig(tile_size=32, min_size=16))
        manifest = read_manifest(self.out)
        dims = [(l["width"], l["height"]) for l in manifest["levels"]]
        self.assertEqual(dims, compute_level_dims(101, 77, 16))
        self.assertEqual(verify_pyramid(self.out), [])
        self.assertEqual(stats.tiles_written, stats.tiles_total)

    def test_build_is_byte_stable(self):
        src = self.make_input(150, 111)
        config = BuildConfig(tile_size=64, min_size=32)
        build_pyramid(src, self.out, config)
        first = read_manifest(self.out)
        out2 = self.work / "pyramid2"
        build_pyramid(src, out2, config)
        second = read_manifest(out2)
        # identical hashes everywhere (paths inside manifest differ only by input path field)
        self.assertEqual(
            [[t["sha256"] for t in l["tiles"]] for l in first["levels"]],
            [[t["sha256"] for t in l["tiles"]] for l in second["levels"]],
        )


class TestIncremental(TempWorkspaceTestCase):
    def _build(self, width=200, height=150, fmt="PNG"):
        src = self.make_input(width, height, fmt=fmt)
        config = BuildConfig(tile_size=64, min_size=32)
        stats = build_pyramid(src, self.out, config)
        return src, config, stats

    def test_second_build_reuses_everything(self):
        src, config, first = self._build()
        mtimes_before = {
            p: p.stat().st_mtime_ns for p in self.out.rglob("*.png")
        }
        second = build_pyramid(src, self.out, config)
        self.assertEqual(second.tiles_written, 0)
        self.assertEqual(second.tiles_reused, first.tiles_total)
        mtimes_after = {p: p.stat().st_mtime_ns for p in self.out.rglob("*.png")}
        self.assertEqual(mtimes_before, mtimes_after)  # no tile was rewritten

    def test_corrupted_tile_is_regenerated(self):
        src, config, _ = self._build()
        victim = next(self.out.rglob("tiles/0/*/*.png"))
        good_bytes = victim.read_bytes()
        victim.write_bytes(b"corrupted!")
        self.assertTrue(any("hash mismatch" in p for p in verify_pyramid(self.out)))

        stats = build_pyramid(src, self.out, config)
        self.assertEqual(stats.tiles_written, 1)
        self.assertEqual(victim.read_bytes(), good_bytes)
        self.assertEqual(verify_pyramid(self.out), [])

    def test_missing_tile_is_regenerated(self):
        src, config, _ = self._build()
        victim = next(self.out.rglob("tiles/1/*/*.png"))
        saved = victim.read_bytes()
        victim.unlink()
        stats = build_pyramid(src, self.out, config)
        self.assertEqual(stats.tiles_written, 1)
        self.assertEqual(victim.read_bytes(), saved)
        self.assertEqual(verify_pyramid(self.out), [])

    def test_changed_input_triggers_full_rebuild(self):
        src, config, first = self._build()
        # overwrite input with different pixels (same size)
        arr = make_gradient_array(200, 150)
        arr = (arr.astype(np.uint16) + 40).clip(0, 255).astype(np.uint8)
        save_image(arr, src)
        stats = build_pyramid(src, self.out, config)
        self.assertEqual(stats.tiles_written, stats.tiles_total)
        self.assertEqual(stats.tiles_reused, 0)
        self.assertEqual(verify_pyramid(self.out), [])


class TestAtomicity(TempWorkspaceTestCase):
    def test_interrupted_build_keeps_old_manifest_valid(self):
        src = self.make_input(200, 150)
        config = BuildConfig(tile_size=64, min_size=32)
        build_pyramid(src, self.out, config)
        old_manifest = (self.out / "manifest.json").read_bytes()
        self.assertEqual(verify_pyramid(self.out), [])

        # change the input so the next build must rewrite tiles
        arr = (make_gradient_array(200, 150).astype(np.uint16) + 90).clip(0, 255).astype(np.uint8)
        save_image(arr, src)

        # simulate a crash part-way through the rebuild
        real_write = pyramid_mod._atomic_write_bytes
        calls = {"n": 0}

        def flaky_write(path, data):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("simulated crash (disk full / killed)")
            return real_write(path, data)

        original = pyramid_mod._atomic_write_bytes
        pyramid_mod._atomic_write_bytes = flaky_write
        try:
            with self.assertRaises(RuntimeError):
                build_pyramid(src, self.out, config)
        finally:
            pyramid_mod._atomic_write_bytes = original

        # the old manifest is untouched; the two tiles written before the
        # crash are complete files (atomic replace) whose hashes do NOT match
        # the old manifest, so they are detected -- never mistaken as valid
        self.assertEqual((self.out / "manifest.json").read_bytes(), old_manifest)
        problems = verify_pyramid(self.out)
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("hash mismatch" in p for p in problems))
        for rel in ("tiles/0/0/0.png", "tiles/0/1/0.png"):
            with Image.open(self.out / rel) as im:
                im.load()  # decodes fully: a complete file, not a half file

        # a normal rebuild recovers and produces a fully valid pyramid
        stats = build_pyramid(src, self.out, config)
        self.assertGreater(stats.tiles_written, 0)
        self.assertEqual(verify_pyramid(self.out), [])
        # stale temp files from the crash are cleaned up
        leftovers = [p for p in self.out.rglob("*") if ".tmp-" in p.name]
        self.assertEqual(leftovers, [])

    def test_no_partial_tile_file_after_failed_write(self):
        src = self.make_input(100, 80)
        config = BuildConfig(tile_size=64, min_size=32)

        def failing_write(path, data):
            # write a truncated temp file, then "crash" before os.replace
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), prefix="x.tmp-")
            os.write(fd, data[:10])
            os.close(fd)
            raise RuntimeError("simulated crash")

        original = pyramid_mod._atomic_write_bytes
        pyramid_mod._atomic_write_bytes = failing_write
        try:
            with self.assertRaises(RuntimeError):
                build_pyramid(src, self.out, config)
        finally:
            pyramid_mod._atomic_write_bytes = original

        # no manifest was written, so nothing can be considered valid
        self.assertFalse((self.out / "manifest.json").exists())
        self.assertNotEqual(verify_pyramid(self.out), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
