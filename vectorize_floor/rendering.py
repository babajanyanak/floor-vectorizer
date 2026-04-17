"""SVG rendering (clean + preview with base64-embedded plan)."""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from xml.etree import ElementTree as ET

from .models import LotMeta

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def _points_to_svg(points: List[Tuple[float, float]]) -> str:
    """Format points for the SVG `points` attribute with 2-decimal precision."""
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _build_svg_root(width: int, height: int) -> ET.Element:
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    root = ET.Element(f"{{{SVG_NS}}}svg", {
        "viewBox": f"0 0 {width} {height}",
        "width": str(width),
        "height": str(height),
        "xmlns": SVG_NS,
        "xmlns:xlink": XLINK_NS,
    })
    return root


def _append_lot_polygons(root: ET.Element, lots: List[LotMeta],
                         css_class: str = "lot") -> None:
    group = ET.SubElement(root, f"{{{SVG_NS}}}g", {"class": "lots"})
    for lot in lots:
        if len(lot.polygon_points) < 3:
            logger.warning("Skipping lot %s: insufficient points", lot.lot_id)
            continue
        attrs = {
            "id": lot.lot_id,
            "data-lot-id": lot.lot_id,
            "class": css_class,
            "points": _points_to_svg(lot.polygon_points),
            "fill": "rgba(80,160,220,0.35)",
            "stroke": "#1b4d6b",
            "stroke-width": "1.5",
            "vector-effect": "non-scaling-stroke",
        }
        if lot.status:
            attrs["data-status"] = str(lot.status)
        if lot.area is not None:
            attrs["data-area"] = str(lot.area)
        if lot.label:
            attrs["data-label"] = str(lot.label)
        ET.SubElement(group, f"{{{SVG_NS}}}polygon", attrs)


def write_clean_svg(lots: List[LotMeta], width: int, height: int,
                    out_path: Path) -> None:
    root = _build_svg_root(width, height)
    _append_lot_polygons(root, lots)
    _write_tree(root, out_path)
    logger.info("Wrote clean SVG: %s", out_path)


def write_preview_svg(lots: List[LotMeta], plan_bgr: np.ndarray,
                      out_path: Path) -> None:
    h, w = plan_bgr.shape[:2]
    root = _build_svg_root(w, h)
    ok, buf = cv2.imencode(".png", plan_bgr)
    if not ok:
        raise RuntimeError("Failed to encode plan image as PNG for preview.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    ET.SubElement(root, f"{{{SVG_NS}}}image", {
        "x": "0", "y": "0",
        "width": str(w), "height": str(h),
        "xlink:href": f"data:image/png;base64,{b64}",
        "href": f"data:image/png;base64,{b64}",
        "preserveAspectRatio": "none",
    })
    _append_lot_polygons(root, lots, css_class="lot preview")
    _write_tree(root, out_path)
    logger.info("Wrote preview SVG: %s", out_path)


def _write_tree(root: ET.Element, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def save_debug_artifacts(debug_dir: Path,
                         mask_raw: Optional[np.ndarray] = None,
                         mask_clean: Optional[np.ndarray] = None,
                         contours_img: Optional[np.ndarray] = None) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    if mask_raw is not None:
        cv2.imwrite(str(debug_dir / "mask_raw.png"), mask_raw)
    if mask_clean is not None:
        cv2.imwrite(str(debug_dir / "mask_clean.png"), mask_clean)
    if contours_img is not None:
        cv2.imwrite(str(debug_dir / "contours.png"), contours_img)
