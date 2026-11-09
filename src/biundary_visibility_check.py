
from pathlib import Path
import csv
import math

ROOT = Path(__file__).resolve().parents[1]
ANNO_DIR = ROOT / "data" / "annotations"

WIDTH = 1920
HEIGHT = 1080

total_partial = 0
total_outside = 0

for clip_no in range(1, 11):

    path = ANNO_DIR / f"Clip_{clip_no}_refined.txt"

    if not path.exists():
        print(f"Clip {clip_no}: MISSING")
        continue

    partial = 0
    completely_outside = 0
    invalid = 0

    with path.open(
        "r", encoding="utf-8-sig"
    ) as f:

        for row in csv.reader(f):
            if len(row) < 6:
                invalid += 1
                continue

            try:
                x, y, w, h = map(float, row[2:6])
            except ValueError:
                invalid += 1
                continue

            if (
                not all(math.isfinite(v) for v in (x, y, w, h))
                or w <= 0
                or h <= 0
            ):
                invalid += 1
                continue

            # Original bounding box
            x2 = x + w
            y2 = y + h

            # Clip coordinates to the image
            cx1 = max(0, min(WIDTH, x))
            cy1 = max(0, min(HEIGHT, y))
            cx2 = max(0, min(WIDTH, x2))
            cy2 = max(0, min(HEIGHT, y2))

            visible_w = cx2 - cx1
            visible_h = cy2 - cy1

            if visible_w <= 0 or visible_h <= 0:
                completely_outside += 1

            elif (
                x < 0 or y < 0
                or x2 > WIDTH or y2 > HEIGHT
            ):
                partial += 1

    total_partial += partial
    total_outside += completely_outside

    print(
        f"Clip {clip_no}: "
        f"partially visible={partial}, "
        f"completely outside={completely_outside}, "
        f"invalid={invalid}"
    )

print("\nTOTAL PARTIALLY VISIBLE:", total_partial)
print("TOTAL COMPLETELY OUTSIDE:", total_outside)
print("\nOriginal annotation files unchanged.")
