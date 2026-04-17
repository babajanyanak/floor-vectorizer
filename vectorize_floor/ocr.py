"""Optional OCR-based lot_id recovery (fallback only)."""
from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

from .models import PipelineConfig, RegionCandidate

logger = logging.getLogger(__name__)

try:
    import pytesseract  # type: ignore
    _HAS_TESSERACT = True
except ImportError:  # pragma: no cover
    _HAS_TESSERACT = False


def extract_lot_id_from_region(
    plan_bgr: np.ndarray, region: RegionCandidate, cfg: PipelineConfig
) -> Optional[str]:
    """Try to OCR a lot_id like 'F2-401' from the region's bbox.

    Returns None if OCR is unavailable, fails, or nothing matches the regex.
    """
    if not _HAS_TESSERACT:
        logger.warning("pytesseract not installed; OCR fallback disabled.")
        return None

    x, y, w, h = region.bbox
    pad = cfg.ocr_padding_px
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(plan_bgr.shape[1], x + w + pad)
    y1 = min(plan_bgr.shape[0], y + h + pad)
    crop = plan_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale = max(1.0, 40.0 / max(1, gray.shape[0] // 10))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    _, thr = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        text = pytesseract.image_to_string(
            thr, config="--psm 6 -c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-",
        )
    except Exception as e:  # pragma: no cover
        logger.warning("Tesseract error: %s", e)
        return None

    pattern = re.compile(cfg.ocr_lot_id_regex)
    for match in pattern.finditer(text):
        candidate = match.group(0).upper().replace(" ", "")
        logger.debug("OCR matched '%s' in region %d", candidate, region.index)
        return candidate
    return None
