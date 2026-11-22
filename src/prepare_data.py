"""Prepare ONE combined, leakage-free dataset from three public angiography sources.

Sources (any can be switched off by setting its path to null in the config)
---------------------------------------------------------------------------
1. Danilov et al. 2021  (Roboflow export: train/ valid/ test/ + COCO json)
   64 patients, bounding boxes, every frame has a stenosis.
2. ARCADE 2023 stenosis task (Zenodo original: stenosis/{train,val,test})
   1,500 images, POLYGON masks; no patient IDs -> official split kept.
3. CADICA 2024 (Mendeley: metadata.xlsx, selectedVideos/pX/vY/...)
   42 patients, cine videos, boxes with a stenosis GRADE, lesion-free videos,
   patients with no affected vessel. Gives the NORMAL / negative frames.

Labelling policy (configurable)
-------------------------------
* Danilov, ARCADE: every annotated stenosis = positive (no grade is provided).
* CADICA: a box is POSITIVE when its grade >= cadica_min_grade (default 70 %).
  Boxes graded below 20 % are clinically insignificant -> treated as background.
  Boxes from 20 % to below the threshold are AMBIGUOUS (real but not "significant"):
  cadica_ambiguous = "drop"  -> frames containing them are left out (default)
                     "negative" -> kept, the ambiguous lesion becomes background
* CADICA frames from lesion-free videos (nonlesionVideos.txt) = NEGATIVE frames.
  Only the key frames listed in pX_vY_selectedFrames.txt are used.

Splits
------
Patient level for Danilov and CADICA (70/15/15 by image count). CADICA patients are
stratified so that patients with NO affected vessel (metadata) are spread over all
splits. ARCADE keeps its official train/val/test split (patient IDs not released).

Outputs (in --out)
------------------
  images/{train,val,test}/<source>_<id>.jpg      CLAHE-enhanced
  labels/{train,val,test}/*.txt                  YOLO boxes (empty file = negative frame)
  masks/{train,val,test}/arc_*.png               ARCADE stenosis masks (for Dice/IoU)
  coco_{train,val,test}.json                     ground truth incl. source/patient/negative
  data.yaml, folds/fold_k.yaml                   training files
  split_manifest.csv                             one row per image (put in thesis appendix)
  eda/                                           counts per source/split, normal patients, figures

  python src/prepare_data.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from preprocess import enhance

SPLITS = ("train", "val", "test")
GRADE_RE = re.compile(r"p(\d+)")


# ------------------------------------------------------------------ helpers
def _num(v):
    """ARCADE/Roboflow JSON sometimes stores numbers as strings."""
    if isinstance(v, str):
        v = v.strip()
        return json.loads(v.replace("'", '"')) if v[:1] in "[{" else float(v)
    return v


def find_dir(root: Path | None, must_contain: str) -> Path | None:
    """Return the (possibly nested) folder under root that contains `must_contain`."""
    if not root:
        return None
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")
    if (root / must_contain).exists():
        return root
    for p in sorted(root.rglob(must_contain)):
        return p.parent
    raise SystemExit(f"Could not find '{must_contain}' inside {root}")


# ------------------------------------------------------------------ 1. Danilov
def load_danilov(raw: Path) -> pd.DataFrame:
    rows = []
    for sub in ("train", "valid", "test"):
        jf = raw / sub / "_annotations.coco.json"
        if not jf.exists():
            continue
        d = json.load(open(jf))
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append([float(v) for v in _num(a["bbox"])])
        for im in d["images"]:
            original = im.get("extra", {}).get("name", im["file_name"])
            p = Path(original).stem.split("_")
            pid, series, frame = (p[1], p[2], int(p[3])) if len(p) >= 4 else (Path(original).stem, "0", 0)
            rows.append(dict(src=str(raw / sub / "images" / im["file_name"]), source="danilov",
                             stem=f"dan_{Path(original).stem}", patient=f"dan_{pid}",
                             video=f"dan_{pid}_{series}", frame=frame, boxes=anns.get(im["id"], []),
                             grades=[], polys=[], negative=False, stratum="disease"))
    df = pd.DataFrame(rows)
    print(f"[Danilov] {len(df)} frames, {df.patient.nunique()} patients")
    return df


# ------------------------------------------------------------------ 2. ARCADE
def load_arcade(root: Path) -> pd.DataFrame:
    rows, missing = [], 0
    for sp_in, sp_out in (("train", "train"), ("val", "val"), ("test", "test")):
        jf = root / "stenosis" / sp_in / "annotations" / f"{sp_in}.json"
        if not jf.exists():
            continue
        d = json.load(open(jf))
        stenosis_ids = {int(_num(c["id"])) for c in d["categories"] if str(c["name"]).lower() == "stenosis"}
        boxes, polys = defaultdict(list), defaultdict(list)
        for a in d["annotations"]:
            if stenosis_ids and int(_num(a["category_id"])) not in stenosis_ids:
                continue
            iid = int(_num(a["image_id"]))
            boxes[iid].append([float(v) for v in _num(a["bbox"])])
            polys[iid].append([[float(v) for v in s] for s in _num(a["segmentation"])])
        for im in d["images"]:
            iid = int(_num(im["id"]))
            src = root / "stenosis" / sp_in / "images" / im["file_name"]
            if not src.exists():
                missing += 1
                continue
            rows.append(dict(src=str(src), source="arcade", stem=f"arc_{sp_in}_{Path(im['file_name']).stem}",
                             patient=f"arc_{sp_in}_{iid}", video=f"arc_{sp_in}_{iid}", frame=0,
                             boxes=boxes.get(iid, []), grades=[], polys=polys.get(iid, []),
                             negative=False, stratum="disease", fixed_split=sp_out))
    df = pd.DataFrame(rows)
    if missing:
        print(f"[ARCADE] warning: {missing} images listed in json but not found on disk")
    print(f"[ARCADE] {len(df)} images (official split), {sum(len(b) for b in df.boxes)} stenosis masks")
    return df


# ------------------------------------------------------------------ 3. CADICA
def _grade(label: str) -> int:
    m = GRADE_RE.match(label.strip().lower())
    return int(m.group(1)) if m else -1


def load_cadica(root: Path, min_grade: int, ambiguous: str, use_negatives: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_excel(root / "metadata.xlsx")
    meta.columns = [c.strip() for c in meta.columns]
    vcol = next(c for c in meta.columns if c.lower().startswith("number of vessels"))
    dcol = next(c for c in meta.columns if c.lower().startswith("maximum degree"))
    meta["max_degree"] = meta[dcol].astype(str).str.replace(" ", "")
    meta["category"] = np.select(
        [(meta[vcol] == 0) & (meta.max_degree == "<20%"), meta[vcol] == 0],
        ["normal (<20%)", "non-obstructive (20-50%)"], "obstructive CAD")
    stratum = {f"cad_p{int(r['Patient ID'])}": r["category"] for _, r in meta.iterrows()}

    sel = root / "selectedVideos"
    rows, dropped, below = [], 0, 0
    for pdir in sorted(sel.glob("p*"), key=lambda p: int(p.name[1:])):
        pid = f"cad_{pdir.name}"
        read = lambda f: (pdir / f).read_text().split() if (pdir / f).exists() else []
        lesion_v, nonlesion_v = set(read("lesionVideos.txt")), set(read("nonlesionVideos.txt"))
        for vdir in sorted(pdir.glob("v*")):
            v = vdir.name
            if v not in lesion_v and not (use_negatives and v in nonlesion_v):
                continue
            sf = vdir / f"{pdir.name}_{v}_selectedFrames.txt"
            frames = sf.read_text().split() if sf.exists() else [p.stem for p in sorted((vdir / "input").glob("*.png"))]
            for fr in frames:
                img = vdir / "input" / f"{fr}.png"
                if not img.exists():
                    continue
                boxes, grades = [], []
                if v in lesion_v:
                    gt = vdir / "groundtruth" / f"{fr}.txt"
                    if not gt.exists():
                        continue
                    amb = False
                    for line in gt.read_text().strip().splitlines():
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        g = _grade(parts[4])
                        if g >= min_grade:
                            boxes.append([float(x) for x in parts[:4]])
                            grades.append(parts[4])
                        elif g >= 20:
                            amb = True
                        else:
                            below += 1
                    if amb and ambiguous == "drop":
                        dropped += 1
                        continue
                rows.append(dict(src=str(img), source="cadica", stem=f"cad_{fr}", patient=pid,
                                 video=f"{pid}_{v}", frame=int(fr.split("_")[-1]), boxes=boxes, grades=grades,
                                 polys=[], negative=len(boxes) == 0, stratum=stratum.get(pid, "disease")))
    df = pd.DataFrame(rows)
    print(f"[CADICA] {len(df)} key frames from {df.patient.nunique()} patients: "
          f"{(~df.negative).sum()} positive (>= {min_grade}%), {df.negative.sum()} negative; "
          f"{dropped} frames dropped (lesion 20-{min_grade - 1}%), {below} boxes <20% ignored")
    meta_out = meta[["Patient ID", vcol, "max_degree", "category"]].rename(
        columns={vcol: "vessels_affected", "max_degree": "max_stenosis"})
    return df, meta_out


# ------------------------------------------------------------------ splitting
def patient_split(df: pd.DataFrame, frac, seed: int) -> dict:
    """Greedy patient assignment matching target image fractions, separately per stratum."""
    assign = {}
    for _, g in df.groupby("stratum"):
        counts = g.groupby("patient").size().to_dict()
        patients = list(counts)
        random.Random(seed).shuffle(patients)
        patients.sort(key=lambda p: -counts[p])
        total = sum(counts.values())
        target = {s: f * total for s, f in zip(SPLITS, frac)}
        have = {s: 0 for s in SPLITS}
        for i, p in enumerate(patients):
            # small strata (e.g. 5 normal patients): make sure test/val each get one first
            if len(patients) <= 12 and i < 3:
                s = ("test", "val", "train")[i]
            else:
                s = max(SPLITS, key=lambda s: (target[s] - have[s]) / max(target[s], 1e-9))
            assign[p] = s
            have[s] += counts[p]
    return assign


# ------------------------------------------------------------------ writing
def write_all(df: pd.DataFrame, out: Path, cfg: dict):
    pp = cfg["preprocess"]
    coco = {s: dict(images=[], annotations=[], categories=[{"id": 1, "name": "stenosis"}]) for s in SPLITS}
    ann_id = 1
    for i, r in enumerate(df.itertuples()):
        img_dir, lab_dir = out / "images" / r.split, out / "labels" / r.split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(r.src, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = enhance(img, pp["clahe"], pp["clahe_clip"], pp["clahe_tile"], pp["denoise"])
        h, w = img.shape[:2]
        cv2.imwrite(str(img_dir / f"{r.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        coco[r.split]["images"].append(dict(id=i, file_name=f"{r.stem}.jpg", width=w, height=h,
                                            source=r.source, patient=r.patient, video=r.video,
                                            negative=bool(r.negative)))
        for k, (x, y, bw, bh) in enumerate(r.boxes):
            x, y = max(0.0, x), max(0.0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw <= 1 or bh <= 1:
                continue
            lines.append(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            a = dict(id=ann_id, image_id=i, category_id=1, bbox=[x, y, bw, bh], area=bw * bh, iscrowd=0)
            if r.grades:
                a["grade"] = r.grades[k]
            if r.polys:
                a["segmentation"] = r.polys[k]
            coco[r.split]["annotations"].append(a)
            ann_id += 1
        (lab_dir / f"{r.stem}.txt").write_text("\n".join(lines))
        if r.polys:
            mdir = out / "masks" / r.split
            mdir.mkdir(parents=True, exist_ok=True)
            m = np.zeros((h, w), np.uint8)
            for poly in r.polys:
                for s in poly:
                    if len(s) >= 6:
                        cv2.fillPoly(m, [np.round(np.array(s).reshape(-1, 2)).astype(np.int32)], 255)
            cv2.imwrite(str(mdir / f"{r.stem}.png"), m)
        if (i + 1) % 2000 == 0:
            print(f"  written {i + 1}/{len(df)}")
    for s in SPLITS:
        json.dump(coco[s], open(out / f"coco_{s}.json", "w"))


def make_folds(df: pd.DataFrame, out: Path, k: int, seed: int):
    from sklearn.model_selection import GroupKFold

    dev = df[df.split != "test"].reset_index(drop=True)
    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fdir = out / "folds"
    fdir.mkdir(exist_ok=True)
    for f, (tr, va) in enumerate(gkf.split(dev, groups=dev.patient)):
        for name, idx in (("train", tr), ("val", va)):
            paths = [str((out / "images" / dev.split[j] / f"{dev.stem[j]}.jpg").resolve()) for j in idx]
            (fdir / f"fold{f}_{name}.txt").write_text("\n".join(paths))
        yaml.safe_dump(dict(path=str(out.resolve()), train=str((fdir / f"fold{f}_train.txt").resolve()),
                            val=str((fdir / f"fold{f}_val.txt").resolve()), test="images/test",
                            names={0: "stenosis"}), open(fdir / f"fold{f}.yaml", "w"), sort_keys=False)


def eda(df: pd.DataFrame, out: Path, meta: pd.DataFrame | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = out / "eda"
    e.mkdir(exist_ok=True)
    df = df.assign(n_boxes=df.boxes.map(len))
    summ = (df.groupby(["source", "split"])
              .agg(images=("stem", "size"), patients=("patient", "nunique"), videos=("video", "nunique"),
                   positive_frames=("negative", lambda s: int((~s).sum())), negative_frames=("negative", "sum"),
                   boxes=("n_boxes", "sum"))
              .reset_index())
    tot = df.groupby("split").agg(images=("stem", "size"), patients=("patient", "nunique"),
                                  videos=("video", "nunique"),
                                  positive_frames=("negative", lambda s: int((~s).sum())),
                                  negative_frames=("negative", "sum"), boxes=("n_boxes", "sum")).reset_index()
    tot.insert(0, "source", "ALL")
    summ = pd.concat([summ, tot]).reset_index(drop=True)
    # ARCADE has no patient IDs: do not count its images as patients
    summ["patients"] = summ["patients"].astype(object)
    summ.loc[summ.source == "arcade", "patients"] = "n/a"
    na = df[df.source != "arcade"].groupby("split").patient.nunique()
    for sp_ in SPLITS:
        summ.loc[(summ.source == "ALL") & (summ.split == sp_), "patients"] = f"{na.get(sp_, 0)} (+ARCADE)" \
            if (df.source == "arcade").any() else na.get(sp_, 0)
    summ["split"] = pd.Categorical(summ.split, SPLITS)
    summ = summ.sort_values(["source", "split"])
    summ.to_csv(e / "split_summary.csv", index=False)

    if meta is not None:
        cad = df[df.source == "cadica"].groupby("patient").agg(split=("split", "first"),
                                                               frames=("stem", "size"),
                                                               negative_frames=("negative", "sum"))
        meta = meta.assign(patient="cad_p" + meta["Patient ID"].astype(int).astype(str)).merge(
            cad, left_on="patient", right_index=True, how="left")
        meta.to_csv(e / "cadica_patients.csv", index=False)
        ct = pd.crosstab(meta.category, meta.split.fillna("not used"), margins=True)
        ct.to_csv(e / "cadica_patient_categories.csv")

    b = pd.DataFrame([dict(source=r.source, w=bx[2], h=bx[3]) for r in df.itertuples() for bx in r.boxes])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for s, c in zip(sorted(b.source.unique()), ("#3b6ea5", "#c0504d", "#4f9a5a")):
        sb = b[b.source == s]
        ax[0].hist(np.sqrt(sb.w * sb.h), 40, alpha=0.6, label=s, color=c)
    ax[0].set_title("Lesion size, sqrt(box area) px"); ax[0].legend(frameon=False)
    sp = summ[summ.source != "ALL"].pivot_table(index="split", columns="source", values="images",
                                                aggfunc="sum", observed=False)
    sp.plot.bar(stacked=True, ax=ax[1], color=["#4f9a5a", "#3b6ea5", "#c0504d"][: sp.shape[1]], rot=0)
    ax[1].set_title("Images per split and source"); ax[1].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(e / "eda.png", dpi=150)
    return summ


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: keep only N images per source")
    ap.add_argument("--split-mode", choices=["patient", "frame"], default="patient",
                    help="patient = correct, leakage-free split (default). frame = random FRAME-level split, "
                         "reproduces the leaky protocol of many published studies; use ONLY for the "
                         "data-leakage experiment, with --out dataset_frame")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dc = cfg["data"]
    out = Path(args.out or dc["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    parts, meta = [], None
    if dc.get("raw_dir"):
        d = load_danilov(Path(dc["raw_dir"]))
        d["split"] = d.patient.map(patient_split(d, dc["split"], dc["seed"]))
        parts.append(d)
    if dc.get("arcade_dir"):
        a = load_arcade(find_dir(dc["arcade_dir"], "stenosis"))
        a["split"] = a.pop("fixed_split")
        parts.append(a)
    if dc.get("cadica_dir"):
        c, meta = load_cadica(find_dir(dc["cadica_dir"], "metadata.xlsx"), int(dc.get("cadica_min_grade", 70)),
                              dc.get("cadica_ambiguous", "drop"), bool(dc.get("cadica_use_negatives", True)))
        c["split"] = c.patient.map(patient_split(c, dc["split"], dc["seed"]))
        parts.append(c)
    if not parts:
        raise SystemExit("No data source configured.")
    df = pd.concat(parts, ignore_index=True)
    if args.limit:
        df = pd.concat([g.sample(min(len(g), args.limit), random_state=0)
                        for _, g in df.groupby("source")], ignore_index=True)

    leak = df.groupby("patient").split.nunique()
    if args.split_mode == "frame":
        # LEAKY protocol: frames assigned at random, so one patient's near-identical cine frames
        # end up in train AND test. ARCADE keeps its official split (no patient IDs anyway).
        rng = np.random.default_rng(dc["seed"])
        m = df.source != "arcade"
        df.loc[m, "split"] = rng.choice(list(SPLITS), size=int(m.sum()), p=dc["split"])
        leak = df[m].groupby("patient").split.nunique()
        print(f"[frame split] patients appearing in more than one split: {(leak > 1).sum()} / {len(leak)} "
              f"(intentional leakage for the comparison experiment)")
    else:
        # safety: no patient may appear in two splits
        assert (leak == 1).all(), f"patient leakage: {leak[leak > 1].index.tolist()[:5]}"

    write_all(df, out, cfg)
    df.drop(columns=["boxes", "polys"]).assign(n_boxes=df.boxes.map(len)).to_csv(out / "split_manifest.csv", index=False)
    yaml.safe_dump(dict(path=str(out.resolve()), train="images/train", val="images/val", test="images/test",
                        names={0: "stenosis"}), open(out / "data.yaml", "w"), sort_keys=False)
    make_folds(df, out, dc["kfolds"], dc["seed"])
    summ = eda(df, out, meta)
    pd.set_option("display.width", 160)
    print("\n" + summ.to_string(index=False))
    if meta is not None:
        print("\nCADICA patients by category and split:")
        print(pd.read_csv(out / "eda" / "cadica_patient_categories.csv").to_string(index=False))
    print(f"\nDataset ready at {out.resolve()}")


if __name__ == "__main__":
    if not Path("configs").exists():
        os.chdir(Path(__file__).resolve().parents[1])
    main()
