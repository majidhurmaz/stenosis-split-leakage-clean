"""Objective 1 — train the stenosis/plaque detector.

Proposed model : RT-DETR (Real-Time DEtection TRansformer, Zhao et al., CVPR 2024).
                 Hybrid architecture: CNN backbone (HGNetv2 / ResNet) for local features
                 + transformer encoder (intra-scale self-attention, AIFI) and transformer
                 decoder with object queries. This is the "CNN + transformer hybrid"
                 described in the proposal, and it is designed for real-time inference.
Baseline       : YOLO11 (pure CNN) — same data, same split, for comparison.

Examples:
  python src/train.py --model rtdetr-l.pt                 # proposed model
  python src/train.py --model rtdetr-x.pt --name rtdetr_x # larger transformer
  python src/train.py --model yolo11m.pt  --name yolo11m  # CNN baseline
  python src/train.py --model rtdetr-l.pt --cv            # 5-fold patient-grouped CV
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def build(model_path: str):
    from ultralytics import RTDETR, YOLO
    return RTDETR(model_path) if "rtdetr" in Path(model_path).name.lower() else YOLO(model_path)


def make_fast_lists(out_dir: Path, fc: dict) -> Path:
    """Thin out near-duplicate cine frames: keep every k-th frame per video (per source).

    Consecutive cine frames are almost identical, so dropping some costs little accuracy
    but cuts every epoch proportionally. Val is thinned too (only used during training);
    the TEST set is never touched, so final results stay comparable.
    """
    import pandas as pd

    man = pd.read_csv(out_dir / "split_manifest.csv")
    stride = fc.get("frame_stride", {})

    def pick(df, scale=1):
        keep = []
        for (src, neg), g in df.groupby(["source", "negative"]):
            key = f"{src}_neg" if (src == "cadica" and neg) else src
            k = max(1, int(stride.get(key, 1)) * scale)
            for _, v in g.sort_values("frame").groupby("video"):
                keep.append(v.iloc[::k])
        return pd.concat(keep)

    lists = {}
    for split, scale in (("train", 1), ("val", int(fc.get("val_stride", 2)))):
        sel = pick(man[man.split == split], scale)
        files = [str((out_dir / "images" / split / f"{s}.jpg").resolve()) for s in sel.stem]
        f = out_dir / f"fast_{split}.txt"
        f.write_text("\n".join(files))
        lists[split] = f
        print(f"[fast] {split}: {len(files)} of {int((man.split == split).sum())} images "
              f"({sel.negative.sum()} normal frames)")
    y = out_dir / "fast.yaml"
    yaml.safe_dump(dict(path=str(out_dir.resolve()), train=str(lists["train"].resolve()),
                        val=str(lists["val"].resolve()), test="images/test", names={0: "stenosis"}),
                   open(y, "w"), sort_keys=False)
    return y


def train_one(model_path, data_yaml, cfg, name, project, **over):
    t = cfg["train"]
    project = str(Path(project).resolve())     # absolute, else Ultralytics nests it under runs/detect
    model = build(model_path)
    args = dict(data=str(data_yaml), imgsz=t["imgsz"], epochs=t["epochs"], batch=t["batch"],
                optimizer=t["optimizer"], lr0=t["lr0"], patience=t["patience"], device=t["device"],
                project=project, name=name, exist_ok=True, seed=cfg["data"]["seed"],
                deterministic=True, plots=True, cos_lr=True,
                # augmentation suited to grayscale X-ray (no colour jitter, mild geometry)
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=10.0, translate=0.1, scale=0.3,
                fliplr=0.5, flipud=0.0, mosaic=0.5, mixup=0.0)
    args.update({k: v for k, v in over.items() if v is not None})
    model.train(**args)
    return Path(project) / name / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default=None, help="prepared dataset folder (default: data.out_dir in config)")
    ap.add_argument("--model", default=None, help="rtdetr-l.pt | rtdetr-x.pt | yolo11m.pt | path/to/last.pt")
    ap.add_argument("--data", default=None, help="data.yaml (default: <out_dir>/data.yaml)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--cv", action="store_true", help="run k-fold patient-grouped cross-validation")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--device")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--fraction", type=float, help="debug: use a fraction of the train set")
    ap.add_argument("--freeze", type=int, help="number of first layers to freeze (0 = train all)")
    ap.add_argument("--patience", type=int, help="early-stopping patience in epochs")
    ap.add_argument("--fast", action="store_true",
                    help="fast fine-tuning: fewer duplicate frames, 512 px, frozen backbone, 40 epochs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.data_dir:
        cfg["data"]["out_dir"] = args.data_dir
    model_path = args.model or cfg["train"]["model"]
    name = args.name or Path(model_path).stem
    over = dict(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device,
                workers=args.workers, fraction=args.fraction, freeze=args.freeze, patience=args.patience)
    out_dir = Path(cfg["data"]["out_dir"])
    data_yaml = args.data or out_dir / "data.yaml"

    if args.fast:
        fc = cfg.get("fast", {})
        data_yaml = args.data or make_fast_lists(out_dir, fc)
        fast_over = dict(imgsz=fc.get("imgsz", 512), epochs=fc.get("epochs", 40), patience=fc.get("patience", 10),
                         batch=fc.get("batch", 6), freeze=fc.get("freeze", 10), deterministic=False,
                         warmup_epochs=1, close_mosaic=5, workers=fc.get("workers", 4))
        over = {**fast_over, **{k: v for k, v in over.items() if v is not None}}
        if args.name is None:
            name = f"{name}-fast"

    if not args.cv:
        best = train_one(model_path, data_yaml, cfg, name, args.project, **over)
        print(f"Best weights: {best}")
        return

    # ---- k-fold cross-validation (proposal: Validation Strategy) ----
    from ultralytics import RTDETR, YOLO  # noqa
    rows = []
    for f in range(cfg["data"]["kfolds"]):
        fy = out_dir / "folds" / f"fold{f}.yaml"
        best = train_one(model_path, fy, cfg, f"{name}_fold{f}", args.project, **over)
        m = build(str(best)).val(data=str(fy), split="val", imgsz=args.imgsz or cfg["train"]["imgsz"],
                                 device=args.device or cfg["train"]["device"], plots=False)
        rows.append(dict(fold=f, mAP50=m.box.map50, mAP50_95=m.box.map, precision=m.box.mp, recall=m.box.mr))
        print(rows[-1])
    import pandas as pd
    df = pd.DataFrame(rows)
    df.loc["mean"] = df.mean(numeric_only=True)
    df.loc["std"] = df.iloc[:-1].std(numeric_only=True)
    out = Path(args.project) / f"{name}_cv_summary.csv"
    df.to_csv(out)
    print(df.to_string())
    print(f"CV summary: {out}")


if __name__ == "__main__":
    main()
