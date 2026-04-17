"""Command-line interface."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .models import PipelineConfig
from .pipeline import Pipeline, PipelineInputs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vectorize_floor",
        description="Vectorize a floor-plan image into SVG lots.",
    )
    p.add_argument("--plan", type=Path, required=True,
                   help="Path to the base floor-plan image.")
    p.add_argument("--overlay", type=Path, default=None,
                   help="Path to the overlay image with fills. Defaults to --plan.")
    p.add_argument("--mapping", type=Path, default=None,
                   help="CSV or JSON file with lot_id mapping.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for SVGs and JSON.")
    p.add_argument("--floor-id", type=str, default="floor",
                   help="Logical floor identifier written into lots.json.")
    p.add_argument("--preview", action="store_true",
                   help="(Kept for compatibility — preview is always produced.)")
    p.add_argument("--ocr-fallback", action="store_true",
                   help="Use OCR when mapping does not cover a region.")
    p.add_argument("--closing-kernel", type=int, default=None,
                   help="Override closing kernel size (px).")
    p.add_argument("--simplify-tolerance", type=float, default=None,
                   help="Override shapely simplify tolerance (px).")
    p.add_argument("--debug", action="store_true",
                   help="Dump intermediate masks and overlays.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = PipelineConfig(
        floor_id=args.floor_id,
        ocr_fallback=args.ocr_fallback,
        debug=args.debug,
    )
    if args.closing_kernel is not None:
        cfg.closing_kernel_size = args.closing_kernel
    if args.simplify_tolerance is not None:
        cfg.simplify_tolerance = args.simplify_tolerance

    inputs = PipelineInputs(
        plan_path=args.plan,
        overlay_path=args.overlay,
        mapping_path=args.mapping,
        out_dir=args.out_dir,
    )

    try:
        Pipeline(cfg).run(inputs)
    except FileNotFoundError as e:
        logging.error("Input file missing: %s", e)
        return 2
    except ValueError as e:
        logging.error("Invalid input: %s", e)
        return 3
    except RuntimeError as e:
        logging.error("Pipeline failure: %s", e)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
