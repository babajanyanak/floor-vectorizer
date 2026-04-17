"""Assignment of lot_id values to detected regions."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import LotMeta, PipelineConfig, RegionCandidate, ValidationReport
from .ocr import extract_lot_id_from_region

logger = logging.getLogger(__name__)


def assign_lot_ids(
    regions: List[RegionCandidate],
    mapping: List[Dict],
    plan_bgr: Optional[np.ndarray],
    cfg: PipelineConfig,
    report: ValidationReport,
) -> List[LotMeta]:
    """Assign lot_id to each region using priority: mapping > OCR > unknown."""
    lots: List[LotMeta] = []
    used_indices: set = set()

    if mapping:
        lots.extend(
            _apply_mapping(regions, mapping, used_indices, report)
        )

    for r in regions:
        if r.index in used_indices:
            continue
        lot_id: Optional[str] = None
        if cfg.ocr_fallback and plan_bgr is not None:
            lot_id = extract_lot_id_from_region(plan_bgr, r, cfg)
            if lot_id:
                logger.info("Region %d assigned via OCR: %s", r.index, lot_id)
        if not lot_id:
            lot_id = f"unknown_{r.index:02d}"
            report.warnings.append(
                f"Region {r.index} got fallback id '{lot_id}' "
                f"(centroid={r.centroid})."
            )
            report.unmapped_regions.append(lot_id)

        lots.append(_region_to_lot(r, lot_id))
        used_indices.add(r.index)

    report.total_regions_found = len(regions)
    report.total_lots_mapped = sum(
        1 for lot in lots if not lot.lot_id.startswith("unknown_")
    )

    expected_ids = {str(m["lot_id"]) for m in mapping if m.get("lot_id")}
    produced_ids = {lot.lot_id for lot in lots}
    report.missing_lot_ids = sorted(expected_ids - produced_ids)

    return lots


def _apply_mapping(
    regions: List[RegionCandidate],
    mapping: List[Dict],
    used_indices: set,
    report: ValidationReport,
) -> List[LotMeta]:
    out: List[LotMeta] = []

    anchor_records = [
        m for m in mapping
        if m.get("anchor_x") is not None and m.get("anchor_y") is not None
    ]
    if anchor_records:
        out.extend(_assign_by_anchors(regions, anchor_records, used_indices))

    index_records = [m for m in mapping if m.get("region_index") is not None]
    for m in index_records:
        idx = int(m["region_index"])
        region = next((r for r in regions if r.index == idx), None)
        if region is None:
            report.warnings.append(
                f"Mapping references region_index={idx} but region not found."
            )
            continue
        if idx in used_indices:
            continue
        out.append(_region_to_lot(region, str(m["lot_id"]), m))
        used_indices.add(idx)

    plain_records = [
        m for m in mapping
        if m not in anchor_records and m not in index_records
    ]
    free_regions = [r for r in regions if r.index not in used_indices]
    for m, r in zip(plain_records, free_regions):
        out.append(_region_to_lot(r, str(m["lot_id"]), m))
        used_indices.add(r.index)
    if len(plain_records) > len(free_regions):
        leftover = plain_records[len(free_regions):]
        for m in leftover:
            report.warnings.append(
                f"No region available for mapping entry lot_id={m.get('lot_id')}"
            )

    return out


def _assign_by_anchors(
    regions: List[RegionCandidate],
    anchor_records: List[Dict],
    used_indices: set,
) -> List[LotMeta]:
    """Greedy nearest-neighbor between anchor points and region centroids."""
    out: List[LotMeta] = []
    pairs: List[Tuple[float, int, Dict]] = []
    for m in anchor_records:
        ax, ay = float(m["anchor_x"]), float(m["anchor_y"])
        for r in regions:
            if r.index in used_indices:
                continue
            cx, cy = r.centroid
            d2 = (ax - cx) ** 2 + (ay - cy) ** 2
            pairs.append((d2, r.index, m))
    pairs.sort(key=lambda t: t[0])

    assigned_records: set = set()
    for _d2, r_idx, m in pairs:
        m_key = id(m)
        if r_idx in used_indices or m_key in assigned_records:
            continue
        region = next(r for r in regions if r.index == r_idx)
        out.append(_region_to_lot(region, str(m["lot_id"]), m))
        used_indices.add(r_idx)
        assigned_records.add(m_key)
    return out


def _region_to_lot(
    r: RegionCandidate, lot_id: str, mapping_entry: Optional[Dict] = None
) -> LotMeta:
    label = None
    status = None
    area = None
    if mapping_entry:
        label = mapping_entry.get("label")
        status = mapping_entry.get("status")
        area = mapping_entry.get("area")
    return LotMeta(
        lot_id=lot_id,
        polygon_points=r.polygon_points,
        bbox=r.bbox,
        centroid=r.centroid,
        area_px=r.area_px,
        label=label,
        status=status,
        area=area,
    )
