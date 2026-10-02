"""Local image pyramid and tile generation tool.

Everything is stored on the local filesystem or in memory. No map server,
CDN, cloud storage or any external service is involved.
"""

from .pyramid import (
    BuildConfig,
    BuildStats,
    build_pyramid,
    compute_level_dims,
    compute_tile_grid,
    resize_image,
    tile_region,
    verify_pyramid,
)

__all__ = [
    "BuildConfig",
    "BuildStats",
    "build_pyramid",
    "compute_level_dims",
    "compute_tile_grid",
    "resize_image",
    "tile_region",
    "verify_pyramid",
]

__version__ = "1.0.0"
