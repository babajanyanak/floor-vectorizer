"""I/O helpers: image loading, mapping parsing, file validation."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    """Load image as BGR uint8. Raises FileNotFoundError / ValueError on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    if img.size == 0:
        raise ValueError(f"Empty image: {path}")
    logger.info("Loaded image %s: shape=%s", path.name, img.shape)
    return img


def align_overlay_to_plan(plan: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Align overlay to plan dimensions.

    If shapes match, returns overlay as-is. Otherwise performs a simple resize.
    For production use with skewed scans, replace with ORB/SIFT-based homography.
    """
    if plan.shape[:2] == overlay.shape[:2]:
        return overlay
    logger.warning(
        "Overlay shape %s != plan shape %s; resizing overlay.",
        overlay.shape[:2], plan.shape[:2],
    )
    h, w = plan.shape[:2]
    return cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)


def load_mapping(path: Optional[Path]) -> List[Dict[str, Any]]:
    """Load lot mapping from CSV or JSON. Returns [] if path is None."""
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported mapping format: {suffix}")

    if "lot_id" not in df.columns:
        raise ValueError("Mapping file must contain a 'lot_id' column.")

    records = df.to_dict(orient="records")
    cleaned = []
    for r in records:
        cleaned.append({k: (None if pd.isna(v) else v) for k, v in r.items()})
    logger.info("Loaded %d lot mappings from %s", len(cleaned), path.name)
    return cleaned


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", path)
