"""End-to-end smoke test using a synthetic floor plan."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures import make_synthetic_plan
from vectorize_floor.models import PipelineConfig
from vectorize_floor.pipeline import Pipeline, PipelineInputs


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.png"
    make_synthetic_plan(plan_path)

    out_dir = tmp_path / "out"
    cfg = PipelineConfig(
        floor_id="test_floor",
        closing_kernel_size=9,
        min_region_area_fraction=0.01,
    )
    summary = Pipeline(cfg).run(
        PipelineInputs(
            plan_path=plan_path,
            overlay_path=None,
            mapping_path=None,
            out_dir=out_dir,
        )
    )

    assert summary["lots_found"] >= 2, "should detect the synthetic rooms"

    lots_json = json.loads((out_dir / "lots.json").read_text())
    assert lots_json["floor_id"] == "test_floor"
    assert len(lots_json["lots"]) == summary["lots_found"]
    for lot in lots_json["lots"]:
        assert len(lot["polygon_points"]) >= 3
        assert lot["area_px"] > 0

    svg = (out_dir / "lots.svg").read_text()
    assert "<polygon" in svg
    assert 'data-lot-id=' in svg

    preview = (out_dir / "lots_preview.svg").read_text()
    assert "base64," in preview

    report = json.loads((out_dir / "validation_report.json").read_text())
    assert report["total_regions_found"] == summary["lots_found"]


def test_pipeline_fails_on_missing_plan(tmp_path: Path) -> None:
    cfg = PipelineConfig()
    with pytest.raises(FileNotFoundError):
        Pipeline(cfg).run(
            PipelineInputs(
                plan_path=tmp_path / "nope.png",
                overlay_path=None,
                mapping_path=None,
                out_dir=tmp_path / "out",
            )
        )
