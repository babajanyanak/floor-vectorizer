"""Tiny synthetic floor-plan generator for tests.

Produces a 600x400 plan with three rectangular rooms filled with different
colors, plus a "wall" outline. Good enough to exercise the full pipeline.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_synthetic_plan(path: Path) -> None:
    h, w = 400, 600
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Outer wall.
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (0, 0, 0), 3)

    rooms = [
        ((30, 30, 200, 180), (210, 220, 240)),   # light blue
        ((240, 30, 410, 180), (220, 240, 215)),  # light green
        ((30, 200, 410, 370), (230, 220, 210)),  # beige
    ]
    for (x1, y1, x2, y2), color in rooms:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # Simulated door gap on the interior wall between rooms 1 and 3:
    # draw white over a small piece of the wall to mimic a door break.
    cv2.line(img, (100, 200), (150, 200), (255, 255, 255), 3)

    # A tiny "column" inside room 3.
    cv2.circle(img, (220, 290), 8, (0, 0, 0), -1)

    # A text label inside room 1.
    cv2.putText(img, "F2-401", (60, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
