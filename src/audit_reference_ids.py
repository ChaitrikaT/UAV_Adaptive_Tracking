"""Summarize identity continuity in EXISTING annotations only.

Does not track objects or assess predicted tracking accuracy.
"""
from collections import defaultdict
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reference_id_audit.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
records = []
for clip in range(1, 11):
    path = ROOT / "data" / "annotations" / f"Clip_{clip}_refined.txt"
    if not path.is_file():
        print(f"Clip {clip}: missing annotation; skipped")
        continue
    by_id = defaultdict(list)
    invalid = 0
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.reader(handle):
            try:
                frame = int(float(row[0]))
                ident = int(float(row[1]))
                if frame < 1:
                    raise ValueError("invalid frame")
                by_id[ident].append(frame)
            except (ValueError, IndexError, OverflowError):
                invalid += 1
    gaps = 0
    duplicate_id_frame = 0
    for frames in by_id.values():
        unique = sorted(set(frames))
        gaps += sum(b-a-1 for a,b in zip(unique, unique[1:]) if b-a>1)
        duplicate_id_frame += len(frames)-len(unique)
    entry = {
        "clip":clip,
        "reference_unique_ids":len(by_id),
        "annotated_rows":sum(map(len,by_id.values())),
        "missing_id_frame_steps":gaps,
        "duplicate_same_id_same_frame":duplicate_id_frame,
        "unreadable_rows":invalid,
    }
    records.append(entry)
    print(entry)
if not records:
    raise SystemExit("No annotations found")
with OUT.open("w", newline="", encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=records[0].keys())
    writer.writeheader();writer.writerows(records)
print("Saved",OUT)
print("NOTE: ground-truth ID continuity is not predicted-tracker evaluation.")
