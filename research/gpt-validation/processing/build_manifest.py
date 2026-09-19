"""Write the manifest of beginner videos the GPT validation study covers.

The study uses the 100 beginner videos the two experts already scored in the
rubric workbooks (50 smash, 50 serve). Workbook IDs and filenames do not always
match exactly -- `EG01` is the file `EG1`, `EG20` is `EG20-較佳`, `EG49` is
`EG49原CG46` -- so IDs are matched by prefix and number. A mirrored `_left`
copy is used only when it is the student's sole recording.

  python research/gpt-validation/processing/build_manifest.py \\
    --videos-root ~/dev/badminton-analysis/scoring_videos \\
    --output research/gpt-validation/processing/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# (skill, workbook relative to the videos root, sheet listing the IDs, folder)
SOURCES = (
    ("smash", "殺球/系統與專家影片評分分數-殺球.xlsx", "專家1", "殺球/初學者殺球"),
    ("serve", "發球/系統與專家影片評分分數-發球(投稿 2026 ECSS 資料).xlsx", "專家1", "發球/初學者發球"),
)

_ID = re.compile(r"^([A-Za-z]+)0*(\d+)")


def id_key(value: str) -> tuple[str, int] | None:
    """`EG01`, `EG1`, `EG1-較佳` and `EG1_left` all share the key (EG, 1)."""
    match = _ID.match(value.strip())
    return (match.group(1).upper(), int(match.group(2))) if match else None


def match_file(workbook_id: str, files: list[str]) -> str:
    key = id_key(workbook_id)
    if key is None:
        raise ValueError(f"unrecognised workbook id {workbook_id!r}")
    candidates = [name for name in files if name.lower().endswith(".mp4") and id_key(Path(name).stem) == key]
    originals = [name for name in candidates if not Path(name).stem.endswith("_left")]
    chosen = originals or candidates
    if len(chosen) != 1:
        raise ValueError(f"{workbook_id!r} matches {len(chosen)} files: {sorted(chosen)}")
    return chosen[0]


def workbook_ids(path: Path, sheet: str) -> list[str]:
    import pandas as pd

    frame = pd.read_excel(path, sheet_name=sheet, header=None)
    # Column D (index 3) lists the beginner IDs under a two-row header.
    ids = [str(value).strip() for value in frame.iloc[2:, 3].dropna()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name} lists a beginner twice")
    return ids


def build(videos_root: Path) -> list[dict[str, str]]:
    rows = []
    for skill, workbook, sheet, folder in SOURCES:
        files = sorted(entry.name for entry in (videos_root / folder).iterdir() if entry.is_file())
        for workbook_id in workbook_ids(videos_root / workbook, sheet):
            source_file = match_file(workbook_id, files)
            rows.append(
                {
                    "skill": skill,
                    "workbook_id": workbook_id,
                    "source_file": source_file,
                    "source_path": f"{folder}/{source_file}",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    rows = build(arguments.videos_root.expanduser())
    with arguments.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["skill", "workbook_id", "source_file", "source_path"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {skill: sum(row["skill"] == skill for row in rows) for skill, *_ in SOURCES}
    print(f"wrote {len(rows)} videos to {arguments.output}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
