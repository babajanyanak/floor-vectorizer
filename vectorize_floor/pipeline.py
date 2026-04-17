"""Orchestrates the full vectorization flow."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import io_utils, rendering, segmentation
from .geometry import extract_regions
from .mapping import assign_lot_ids
from .models import PipelineConfig, ValidationReport

logger = logging.getLogger(__name__)


@dataclass
class PipelineInputs:
    plan_path: Path
    overlay_path: Optional[Path]
    mapping_path: Optional[Path]
    out_dir: Path


class Pipeline:
    """Orchestrates the stages of floor-plan vectorization.

    Designed to be called programmatically (e.g., from a FastAPI worker) or
    via the CLI wrapper. Each public method is idempotent given the same
    inputs and config.
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.report = ValidationReport()

    def run(self, inputs: PipelineInputs) -> dict:
        """Run the full pipeline end-to-end and return a summary dict."""
        out_dir = inputs.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        plan = io_utils.load_image(inputs.plan_path)
        overlay_path = inputs.overlay_path or inputs.plan_path
        overlay = io_utils.load_image(overlay_path)
        overlay = io_utils.align_overlay_to_plan(plan, overlay)

        mapping = io_utils.load_mapping(inputs.mapping_path)

        try:
            mask_raw = segmentation.build_fill_mask(overlay, self.cfg)
        except RuntimeError as e:
            self.report.errors.append(str(e))
            io_utils.save_json(self.report.to_dict(),
                               out_dir / "validation_report.json")
            raise

        mask_clean = segmentation.clean_mask(mask_raw, self.cfg)

        regions = extract_regions(mask_clean, self.cfg)
        if not regions:
            self.report.errors.append("No room regions extracted from mask.")
            io_utils.save_json(self.report.to_dict(),
                               out_dir / "validation_report.json")
            raise RuntimeError("No regions found — check segmentation settings.")

        lots = assign_lot_ids(regions, mapping, plan, self.cfg, self.report)
        self._sanity_check_lots(lots)

        h, w = plan.shape[:2]
        rendering.write_clean_svg(lots, w, h, out_dir / "lots.svg")
        rendering.write_preview_svg(lots, plan, out_dir / "lots_preview.svg")

        lots_json = {
            "floor_id": self.cfg.floor_id,
            "lots": [lot.to_dict() for lot in lots],
        }
        io_utils.save_json(lots_json, out_dir / "lots.json")
        io_utils.save_json(self.report.to_dict(),
                           out_dir / "validation_report.json")

        if self.cfg.debug:
            contours_img = plan.copy()
            for lot in lots:
                pts = np.array(lot.polygon_points, dtype=np.int32)
                cv2.polylines(contours_img, [pts], True, (0, 0, 255), 2)
                cv2.putText(contours_img, lot.lot_id,
                            (int(lot.centroid[0]), int(lot.centroid[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            rendering.save_debug_artifacts(
                out_dir / "debug",
                mask_raw=mask_raw,
                mask_clean=mask_clean,
                contours_img=contours_img,
            )

        summary = {
            "lots_found": len(lots),
            "lots_mapped": self.report.total_lots_mapped,
            "warnings": len(self.report.warnings),
            "errors": len(self.report.errors),
        }
        logger.info("Pipeline summary: %s", summary)
        return summary

    def _sanity_check_lots(self, lots) -> None:
        """Add warnings for suspicious polygons."""
        for lot in lots:
            if len(lot.polygon_points) < 4:
                self.report.warnings.append(
                    f"Lot {lot.lot_id} has only {len(lot.polygon_points)} vertices."
                )
            if len(lot.polygon_points) > 80:
                self.report.warnings.append(
                    f"Lot {lot.lot_id} has {len(lot.polygon_points)} vertices — "
                    f"possibly under-simplified."
                )
            _x, _y, bw, bh = lot.bbox
            if min(bw, bh) < 10:
                self.report.warnings.append(
                    f"Lot {lot.lot_id} bbox is very thin: {bw}x{bh}."
                )
