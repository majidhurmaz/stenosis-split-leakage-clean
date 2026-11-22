"""Make a SMALL summary + sample of big datasets so they can be shared for code writing.

Usage (from the project folder, inside 'conda activate plaque'):
    python inspect_datasets.py --cadica "D:\\Datasets\\CADICA" --arcade "D:\\Datasets\\arcade"

Output: datasets_sample.zip (usually 5-30 MB) in the current folder. Upload that file.
Nothing in the original folders is changed.
"""
import argparse
import collections
import os
import zipfile
from pathlib import Path

MAX_SMALL_FILE = 20 * 1024 * 1024      # copy metadata/annotation files up to 20 MB
TEXT_EXT = {".txt", ".csv", ".json", ".xlsx", ".xls", ".mat", ".yaml", ".yml", ".md"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}


def tree_report(root: Path, out_lines, max_list=15):
    out_lines.append(f"\n################ {root} ################")
    if not root.exists():
        out_lines.append("!! PATH NOT FOUND")
        return
    ext_total = collections.Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        exts = collections.Counter(Path(f).suffix.lower() for f in filenames)
        ext_total.update(exts)
        if depth <= 3:
            ind = "    " * depth
            out_lines.append(f"{ind}{rel.name or root.name}/  dirs={len(dirnames)} files={len(filenames)} {dict(exts)}")
            for f in sorted(filenames)[:max_list]:
                out_lines.append(f"{ind}    - {f}")
            if len(filenames) > max_list:
                out_lines.append(f"{ind}    ... (+{len(filenames) - max_list} more)")
    out_lines.append(f"TOTAL by extension: {dict(ext_total)}")


def pick_samples(root: Path):
    """Small text/metadata files (all big-ish ones by type limited) + a few images."""
    picks = []
    per_ext_count = collections.Counter()
    img_dirs_done = collections.Counter()
    for dirpath, _, filenames in os.walk(root):
        for f in sorted(filenames):
            p = Path(dirpath) / f
            ext = p.suffix.lower()
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if ext in TEXT_EXT and size <= MAX_SMALL_FILE:
                # keep every metadata-like file, but only 40 of the many per-frame txt files
                if ext == ".txt" and per_ext_count[".txt"] >= 40 and "lesion" not in f.lower() and "video" not in f.lower():
                    continue
                per_ext_count[ext] += 1
                picks.append(p)
            elif ext in IMG_EXT and img_dirs_done[dirpath] < 2 and sum(img_dirs_done.values()) < 40:
                img_dirs_done[dirpath] += 1
                picks.append(p)
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadica", required=True)
    ap.add_argument("--arcade", required=True)
    ap.add_argument("--out", default="datasets_sample.zip")
    a = ap.parse_args()
    roots = {"CADICA": Path(a.cadica), "ARCADE": Path(a.arcade)}
    lines = []
    for name, r in roots.items():
        lines.append(f"\n===================== {name} =====================")
        tree_report(r, lines)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("REPORT.txt", "\n".join(lines))
        for name, r in roots.items():
            if not r.exists():
                continue
            for p in pick_samples(r):
                z.write(p, f"{name}/{p.relative_to(r)}")
    mb = os.path.getsize(a.out) / 1e6
    print("\n".join(lines[:60]))
    print(f"\nDone -> {Path(a.out).resolve()}  ({mb:.1f} MB). Upload this file.")


if __name__ == "__main__":
    main()
