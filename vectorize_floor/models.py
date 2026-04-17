"""Typed data models used across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any


Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class PipelineConfig:
    """Configuration for the vectorization pipeline.

    All thresholds are expressed either in pixels (absolute) or as fractions
    of the image size (relative). Pick values carefully — they control the
    trade-off between noise-robustness and geometric fidelity.
    """

    # --- Segmentation ---
    n_color_clusters: int = 6
    min_cluster_fraction: float = 0.005
    ignore_near_white_threshold: int = 235
    ignore_near_black_threshold: int = 40

    # --- Mask cleanup ---
    closing_kernel_size: int = 15
    min_region_area_fraction: float = 0.0015
    max_region_area_fraction: float = 0.6
    min_solidity: float = 0.55

    # --- Geometry simplification ---
    approx_epsilon_fraction: float = 0.005
    simplify_tolerance: float = 3.0
    axis_snap_angle_deg: float = 4.0

    # --- OCR ---
    ocr_fallback: bool = False
    ocr_lot_id_regex: str = r"(?:F\d+|R|OF|P|B)[-\s]?\d{2,4}"
    ocr_padding_px: int = 20

    # --- Output ---
    floor_id: str = "floor"
    debug: bool = False


@dataclass
class RegionCandidate:
    """A candidate room region extracted from the mask."""

    index: int
    polygon_points: List[Point]
    bbox: BBox
    centroid: Point
    area_px: float
    raw_contour_points: List[Point] = field(default_factory=list)


@dataclass
class LotMeta:
    """Final lot metadata ready for SVG / JSON serialization."""

    lot_id: str
    polygon_points: List[Point]
    bbox: BBox
    centroid: Point
    area_px: float
    label: Optional[str] = None
    status: Optional[str] = None
    area: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["polygon_points"] = [list(p) for p in self.polygon_points]
        d["bbox"] = list(self.bbox)
        d["centroid"] = list(self.centroid)
        return d


@dataclass
class ValidationReport:
    total_regions_found: int = 0
    total_lots_mapped: int = 0
    unmapped_regions: List[str] = field(default_factory=list)
    missing_lot_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
