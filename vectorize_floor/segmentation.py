"""Room-fill color segmentation and mask cleanup."""
from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from .models import PipelineConfig

logger = logging.getLogger(__name__)


def _dominant_fill_colors(
    img_bgr: np.ndarray, cfg: PipelineConfig
) -> List[Tuple[int, int, int]]:
    """Cluster non-background pixels to find dominant fill colors.

    Ignores near-white (paper) and near-black (walls/text) pixels.
    Returns a list of BGR centroids ordered by coverage (descending).
    """
    flat = img_bgr.reshape(-1, 3).astype(np.float32)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).ravel()
    mask = (gray < cfg.ignore_near_white_threshold) & (
        gray > cfg.ignore_near_black_threshold
    )
    candidate_pixels = flat[mask]
    if len(candidate_pixels) < 1000:
        logger.warning("Very few candidate pixels for clustering: %d", len(candidate_pixels))
        return []

    max_samples = 100_000
    if len(candidate_pixels) > max_samples:
        idx = np.random.choice(len(candidate_pixels), max_samples, replace=False)
        candidate_pixels = candidate_pixels[idx]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        candidate_pixels, cfg.n_color_clusters, None, criteria, 3,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.ravel()

    counts = np.bincount(labels, minlength=cfg.n_color_clusters)
    total = counts.sum()
    ordered_idx = np.argsort(-counts)

    result = []
    for i in ordered_idx:
        fraction = counts[i] / total
        if fraction < cfg.min_cluster_fraction:
            continue
        bgr = tuple(int(v) for v in centers[i])
        result.append(bgr)

    logger.info("Found %d dominant fill colors", len(result))
    return result


def build_fill_mask(overlay_bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Build a binary mask of room fills."""
    colors = _dominant_fill_colors(overlay_bgr, cfg)
    if not colors:
        raise RuntimeError(
            "Could not detect any dominant fill colors. "
            "Check overlay image or tune n_color_clusters / min_cluster_fraction."
        )

    lab = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2LAB)
    mask = np.zeros(overlay_bgr.shape[:2], dtype=np.uint8)

    for bgr in colors:
        target_lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0]
        lo = np.array([max(0, int(target_lab[0]) - 40),
                       max(0, int(target_lab[1]) - 12),
                       max(0, int(target_lab[2]) - 12)], dtype=np.uint8)
        hi = np.array([min(255, int(target_lab[0]) + 40),
                       min(255, int(target_lab[1]) + 12),
                       min(255, int(target_lab[2]) + 12)], dtype=np.uint8)
        partial = cv2.inRange(lab, lo, hi)
        mask = cv2.bitwise_or(mask, partial)

    coverage = float(mask.mean()) / 255.0
    logger.info("Raw fill mask coverage: %.3f", coverage)
    if coverage < 0.01:
        logger.warning("Mask coverage is very low — colors likely mis-detected.")
    return mask


def clean_mask(mask: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Clean the mask: seal door gaps, fill text/column holes, drop noise."""
    k = cfg.closing_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, small_kernel, iterations=1)

    filled = _fill_holes_per_component(opened)

    logger.info(
        "Mask cleanup: raw=%d, closed=%d, filled=%d (nonzero px)",
        int((mask > 0).sum()), int((closed > 0).sum()), int((filled > 0).sum()),
    )
    return filled


def _fill_holes_per_component(binary: np.ndarray) -> np.ndarray:
    """Fill internal holes in each connected component."""
    num, labels = cv2.connectedComponents(binary)
    out = np.zeros_like(binary)
    for lbl in range(1, num):
        comp = (labels == lbl).astype(np.uint8) * 255
        padded = cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        ff = padded.copy()
        ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(ff, ff_mask, (0, 0), 255)
        holes = cv2.bitwise_not(ff)
        filled_padded = cv2.bitwise_or(padded, holes)
        filled = filled_padded[1:-1, 1:-1]
        out = cv2.bitwise_or(out, filled)
    return out
