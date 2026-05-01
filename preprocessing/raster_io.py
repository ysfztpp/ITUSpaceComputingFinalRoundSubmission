from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import (
    INVALID_FILL_VALUE,
    VALID_REFLECTANCE_MAX_INCLUSIVE,
    VALID_REFLECTANCE_MIN_EXCLUSIVE,
)

try:
    import rasterio
    from rasterio.windows import Window
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "rasterio is required for this project. PIL reads these float GeoTIFFs incorrectly. "
        "Install with `python3 -m pip install rasterio`."
    ) from exc


@dataclass(frozen=True)
class RasterMeta:
    path: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: int
    height: int
    pixel_size_x: float
    pixel_size_y: float
    crs: str
    transform: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class PatchExtraction:
    patch: np.ndarray
    pixel_x: int
    pixel_y: int
    pixel_x_float: float
    pixel_y_float: float
    border_margin_pixels: int
    center_clamped: bool


@dataclass(frozen=True)
class PatchWindow:
    pixel_x: int
    pixel_y: int
    pixel_x_float: float
    pixel_y_float: float
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    border_margin_pixels: int
    center_clamped: bool


def raster_meta_from_src(path: Path, src) -> RasterMeta:
    transform = src.transform
    bounds = src.bounds
    return RasterMeta(
        path=str(path),
        xmin=float(bounds.left),
        ymin=float(bounds.bottom),
        xmax=float(bounds.right),
        ymax=float(bounds.top),
        width=int(src.width),
        height=int(src.height),
        pixel_size_x=float(abs(transform.a)),
        pixel_size_y=float(abs(transform.e)),
        crs=str(src.crs) if src.crs else "",
        transform=(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f),
    )


def read_raster_meta(path: Path) -> RasterMeta:
    with rasterio.open(path) as src:
        return raster_meta_from_src(path, src)


def contains_point(meta: RasterMeta, lon: float, lat: float) -> bool:
    return meta.xmin <= lon < meta.xmax and meta.ymin <= lat < meta.ymax


def bbox_distance_deg(meta: RasterMeta, lon: float, lat: float) -> float:
    dx = 0.0
    if lon < meta.xmin:
        dx = meta.xmin - lon
    elif lon > meta.xmax:
        dx = lon - meta.xmax

    dy = 0.0
    if lat < meta.ymin:
        dy = meta.ymin - lat
    elif lat > meta.ymax:
        dy = lat - meta.ymax
    return math.hypot(dx, dy)


def lonlat_to_pixel(meta: RasterMeta, lon: float, lat: float) -> tuple[float, float, int, int]:
    pixel_x_float = (lon - meta.xmin) / meta.pixel_size_x
    pixel_y_float = (meta.ymax - lat) / meta.pixel_size_y
    return pixel_x_float, pixel_y_float, int(round(pixel_x_float)), int(round(pixel_y_float))


def patch_fits_without_padding(meta: RasterMeta, pixel_x: int, pixel_y: int, patch_size: int) -> bool:
    half = patch_size // 2
    return half <= pixel_x < (meta.width - half) and half <= pixel_y < (meta.height - half)


def border_margin_pixels(meta: RasterMeta, pixel_x: int, pixel_y: int) -> int:
    return min(pixel_x, pixel_y, meta.width - 1 - pixel_x, meta.height - 1 - pixel_y)


def _patch_window(src, meta: RasterMeta, lon: float, lat: float, patch_size: int) -> PatchWindow:
    half = patch_size // 2
    pixel_x_float, pixel_y_float, pixel_x, pixel_y = lonlat_to_pixel(meta, lon, lat)
    center_clamped = False

    if pixel_x < 0:
        pixel_x = 0
        center_clamped = True
    elif pixel_x >= src.width:
        pixel_x = src.width - 1
        center_clamped = True

    if pixel_y < 0:
        pixel_y = 0
        center_clamped = True
    elif pixel_y >= src.height:
        pixel_y = src.height - 1
        center_clamped = True

    x0 = pixel_x - half
    x1 = pixel_x + half + 1
    y0 = pixel_y - half
    y1 = pixel_y + half + 1

    crop_left = max(0, x0)
    crop_top = max(0, y0)
    crop_right = min(src.width, x1)
    crop_bottom = min(src.height, y1)
    if crop_right <= crop_left or crop_bottom <= crop_top:
        raise ValueError("empty patch after coordinate clamping")

    pad_left = crop_left - x0
    pad_top = crop_top - y0
    pad_right = x1 - crop_right
    pad_bottom = y1 - crop_bottom
    return PatchWindow(
        pixel_x=int(pixel_x),
        pixel_y=int(pixel_y),
        pixel_x_float=float(pixel_x_float),
        pixel_y_float=float(pixel_y_float),
        crop_left=int(crop_left),
        crop_top=int(crop_top),
        crop_right=int(crop_right),
        crop_bottom=int(crop_bottom),
        pad_left=int(pad_left),
        pad_top=int(pad_top),
        pad_right=int(pad_right),
        pad_bottom=int(pad_bottom),
        border_margin_pixels=int(border_margin_pixels(meta, pixel_x, pixel_y)),
        center_clamped=bool(center_clamped),
    )


def _patch_extraction_from_array(patch: np.ndarray, spec: PatchWindow, patch_size: int) -> PatchExtraction:
    if any(value > 0 for value in (spec.pad_left, spec.pad_top, spec.pad_right, spec.pad_bottom)):
        patch = np.pad(patch, ((spec.pad_top, spec.pad_bottom), (spec.pad_left, spec.pad_right)), mode="edge")

    if patch.shape != (patch_size, patch_size):
        raise ValueError(f"unexpected patch shape {patch.shape}, expected {(patch_size, patch_size)}")

    return PatchExtraction(
        patch=patch.astype(np.float32, copy=False),
        pixel_x=spec.pixel_x,
        pixel_y=spec.pixel_y,
        pixel_x_float=spec.pixel_x_float,
        pixel_y_float=spec.pixel_y_float,
        border_margin_pixels=spec.border_margin_pixels,
        center_clamped=spec.center_clamped,
    )


def _extract_patch_from_open_src(src, meta: RasterMeta, lon: float, lat: float, patch_size: int) -> PatchExtraction:
    spec = _patch_window(src, meta, lon, lat, patch_size)
    window = Window(spec.crop_left, spec.crop_top, spec.crop_right - spec.crop_left, spec.crop_bottom - spec.crop_top)
    patch = src.read(1, window=window, out_dtype="float32", masked=False)
    patch = np.asarray(patch, dtype=np.float32)
    return _patch_extraction_from_array(patch, spec, patch_size)


def extract_patch_edge(path: Path, lon: float, lat: float, patch_size: int) -> PatchExtraction:
    with rasterio.open(path) as src:
        meta = raster_meta_from_src(path, src)
        return _extract_patch_from_open_src(src, meta, lon, lat, patch_size)


def extract_patch_edge_from_src(src, meta: RasterMeta, lon: float, lat: float, patch_size: int) -> PatchExtraction:
    return _extract_patch_from_open_src(src, meta, lon, lat, patch_size)


def extract_patches_edge_batched_from_src(
    src,
    meta: RasterMeta,
    points: list[tuple[float, float]],
    patch_size: int,
    max_union_pixels: int = 262144,
    max_overread_ratio: float = 6.0,
) -> tuple[list[PatchExtraction], bool, int]:
    """Extract same edge-padded patches, optionally from one union read.

    Returns ``(patches, used_batch_read, pixels_read)``. If the union window
    would read too much extra data, this falls back to exact per-patch reads.
    """

    specs = [_patch_window(src, meta, lon, lat, patch_size) for lon, lat in points]
    if not specs:
        return [], False, 0

    union_left = min(spec.crop_left for spec in specs)
    union_top = min(spec.crop_top for spec in specs)
    union_right = max(spec.crop_right for spec in specs)
    union_bottom = max(spec.crop_bottom for spec in specs)
    union_pixels = (union_right - union_left) * (union_bottom - union_top)
    exact_pixels = sum((spec.crop_right - spec.crop_left) * (spec.crop_bottom - spec.crop_top) for spec in specs)

    if union_pixels > max_union_pixels or union_pixels > int(max_overread_ratio * max(exact_pixels, 1)):
        out: list[PatchExtraction] = []
        pixels_read = 0
        for spec in specs:
            window = Window(spec.crop_left, spec.crop_top, spec.crop_right - spec.crop_left, spec.crop_bottom - spec.crop_top)
            patch = src.read(1, window=window, out_dtype="float32", masked=False)
            patch = np.asarray(patch, dtype=np.float32)
            pixels_read += int(patch.size)
            out.append(_patch_extraction_from_array(patch, spec, patch_size))
        return out, False, pixels_read

    window = Window(union_left, union_top, union_right - union_left, union_bottom - union_top)
    union = src.read(1, window=window, out_dtype="float32", masked=False)
    union = np.asarray(union, dtype=np.float32)
    out = []
    for spec in specs:
        r0 = spec.crop_top - union_top
        r1 = spec.crop_bottom - union_top
        c0 = spec.crop_left - union_left
        c1 = spec.crop_right - union_left
        out.append(_patch_extraction_from_array(union[r0:r1, c0:c1], spec, patch_size))
    return out, True, int(union.size)


def _default_batch_tile_size(src, max_union_pixels: int, patch_size: int) -> int:
    """Choose a practical spatial bin size for clustered window reads.

    Native TIFF blocks are preferred when they are small enough to be useful.
    Striped TIFFs often report very wide blocks; in that case use a square tile
    derived from the configured union-pixel cap.
    """

    fallback = max(patch_size, int(math.sqrt(max(max_union_pixels, patch_size * patch_size))))
    try:
        block_h, block_w = src.block_shapes[0]
        block_pixels = int(block_h) * int(block_w)
        if patch_size <= block_h <= fallback and patch_size <= block_w <= fallback and block_pixels <= max_union_pixels:
            return int(max(block_h, block_w))
    except Exception:
        pass
    return int(fallback)


def extract_patches_edge_clustered_from_src(
    src,
    meta: RasterMeta,
    points: list[tuple[float, float]],
    patch_size: int,
    max_union_pixels: int = 262144,
    max_overread_ratio: float = 6.0,
    tile_size: int | None = None,
) -> tuple[list[PatchExtraction], dict[str, int]]:
    """Extract patches using small spatial union windows.

    Unlike ``extract_patches_edge_batched_from_src``, this does not try to read
    every point in a region/date/band through one large union. It bins nearby
    patch centers into raster-sized tiles, reads one bounded union per bin, and
    falls back to exact per-patch reads only for bins whose union would overread
    too much.
    """

    specs = [_patch_window(src, meta, lon, lat, patch_size) for lon, lat in points]
    stats = {
        "read_calls": 0,
        "batch_read_calls": 0,
        "fallback_patch_read_calls": 0,
        "pixels_read": 0,
        "cluster_count": 0,
        "batched_patch_count": 0,
        "fallback_patch_count": 0,
    }
    if not specs:
        return [], stats

    tile = int(tile_size or _default_batch_tile_size(src, max_union_pixels, patch_size))
    groups: dict[tuple[int, int], list[tuple[int, PatchWindow]]] = {}
    for index, spec in enumerate(specs):
        key = (spec.pixel_x // tile, spec.pixel_y // tile)
        groups.setdefault(key, []).append((index, spec))
    stats["cluster_count"] = len(groups)

    out: list[PatchExtraction | None] = [None] * len(specs)
    for items in groups.values():
        group_specs = [spec for _, spec in items]
        union_left = min(spec.crop_left for spec in group_specs)
        union_top = min(spec.crop_top for spec in group_specs)
        union_right = max(spec.crop_right for spec in group_specs)
        union_bottom = max(spec.crop_bottom for spec in group_specs)
        union_pixels = (union_right - union_left) * (union_bottom - union_top)
        exact_pixels = sum((spec.crop_right - spec.crop_left) * (spec.crop_bottom - spec.crop_top) for spec in group_specs)
        use_union = len(items) > 1 and union_pixels <= max_union_pixels and union_pixels <= int(max_overread_ratio * max(exact_pixels, 1))

        if use_union:
            window = Window(union_left, union_top, union_right - union_left, union_bottom - union_top)
            union = src.read(1, window=window, out_dtype="float32", masked=False)
            union = np.asarray(union, dtype=np.float32)
            stats["read_calls"] += 1
            stats["batch_read_calls"] += 1
            stats["pixels_read"] += int(union.size)
            stats["batched_patch_count"] += len(items)
            for index, spec in items:
                r0 = spec.crop_top - union_top
                r1 = spec.crop_bottom - union_top
                c0 = spec.crop_left - union_left
                c1 = spec.crop_right - union_left
                out[index] = _patch_extraction_from_array(union[r0:r1, c0:c1], spec, patch_size)
            continue

        for index, spec in items:
            window = Window(spec.crop_left, spec.crop_top, spec.crop_right - spec.crop_left, spec.crop_bottom - spec.crop_top)
            patch = src.read(1, window=window, out_dtype="float32", masked=False)
            patch = np.asarray(patch, dtype=np.float32)
            stats["read_calls"] += 1
            stats["fallback_patch_read_calls"] += 1
            stats["fallback_patch_count"] += 1
            stats["pixels_read"] += int(patch.size)
            out[index] = _patch_extraction_from_array(patch, spec, patch_size)

    return [item for item in out if item is not None], stats


def clean_patch_values(
    patch: np.ndarray,
    min_exclusive: float = VALID_REFLECTANCE_MIN_EXCLUSIVE,
    max_inclusive: float = VALID_REFLECTANCE_MAX_INCLUSIVE,
    fill_value: float = INVALID_FILL_VALUE,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(patch) & (patch > min_exclusive) & (patch <= max_inclusive)
    cleaned = np.where(valid, patch, fill_value).astype(np.float32, copy=False)
    return cleaned, valid.astype(bool, copy=False)
