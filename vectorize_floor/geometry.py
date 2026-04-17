"""Contour extraction and geometric simplification."""
from __future__ import annotations

import logging
import math
from typing import List

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

from .models import PipelineConfig, RegionCandidate, Point

logger = logging.getLogger(__name__)


def extract_regions(
    mask: np.ndarray, cfg: PipelineConfig
) -> List[RegionCandidate]:
    """Extract candidate room regions from a cleaned binary mask."""
    h, w = mask.shape[:2]
    total_area = float(h * w)
    min_area = cfg.min_region_area_fraction * total_area
    max_area = cfg.max_region_area_fraction * total_area

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    regions: List[RegionCandidate] = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        if len(cnt) < 4:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < cfg.min_solidity:
            logger.debug("Region %d rejected: low solidity %.2f", i, solidity)
            continue

        simplified = _simplify_contour(cnt, cfg)
        if len(simplified) < 3:
            continue

        simplified = _snap_to_axes(simplified, cfg.axis_snap_angle_deg)

        poly = Polygon(simplified)
        if not poly.is_valid:
            poly = make_valid(poly)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)
            if poly.geom_type != "Polygon":
                continue
        poly = poly.simplify(cfg.simplify_tolerance, preserve_topology=True)
        if poly.is_empty or poly.area < min_area:
            continue

        coords = list(poly.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]

        x, y, bw, bh = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] else x + bw / 2
        cy = M["m01"] / M["m00"] if M["m00"] else y + bh / 2

        regions.append(
            RegionCandidate(
                index=len(regions),
                polygon_points=[(float(px), float(py)) for px, py in coords],
                bbox=(int(x), int(y), int(bw), int(bh)),
                centroid=(float(cx), float(cy)),
                area_px=float(area),
                raw_contour_points=[tuple(p[0]) for p in cnt] if cfg.debug else [],
            )
        )

    regions.sort(key=lambda r: -r.area_px)
    for new_i, r in enumerate(regions):
        r.index = new_i

    logger.info("Extracted %d region candidates", len(regions))
    return regions


def _simplify_contour(cnt: np.ndarray, cfg: PipelineConfig) -> List[Point]:
    perimeter = cv2.arcLength(cnt, closed=True)
    epsilon = cfg.approx_epsilon_fraction * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def _snap_to_axes(points: List[Point], max_angle_deg: float) -> List[Point]:
    """Snap near-horizontal/vertical segments to the axes."""
    if len(points) < 3:
        return points

    pts = [list(p) for p in points]
    n = len(pts)
    thr = math.tan(math.radians(max_angle_deg))

    for i in range(n):
        j = (i + 1) % n
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        if abs(dx) > 1e-6 and abs(dy / dx) < thr:
            avg_y = (pts[i][1] + pts[j][1]) / 2
            pts[i][1] = avg_y
            pts[j][1] = avg_y
        elif abs(dy) > 1e-6 and abs(dx / dy) < thr:
            avg_x = (pts[i][0] + pts[j][0]) / 2
            pts[i][0] = avg_x
            pts[j][0] = avg_x

    cleaned: List[Point] = []
    for p in pts:
        tp = (float(p[0]), float(p[1]))
        if not cleaned or (abs(cleaned[-1][0] - tp[0]) > 0.5 or
                           abs(cleaned[-1][1] - tp[1]) > 0.5):
            cleaned.append(tp)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    return cleaned
