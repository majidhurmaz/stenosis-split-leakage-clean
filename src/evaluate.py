"""Evaluate a trained detector on the held-out TEST patients (+ measurement + speed).

Metrics (proposal §Evaluation Metrics):
  Detection   : mAP@0.5, mAP@0.5:0.95 (COCO), Precision, Recall, F1, FDR at the
                operating confidence; mean IoU and Dice of matched boxes; recall by
                lesion size; image-level detection rate
  Measurement : %DS / MLD / length / area / volume for every detection; agreement
                between measurements on predicted vs ground-truth boxes
  Real-time   : detection latency, measurement latency, end-to-end FPS

  python src/evaluate.py --weights runs/rtdetr-l/weights/best.pt --name rtdetr-l
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from measure import draw, measure_stenosis
from stats_utils import agreement
from train import build


def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def match(preds, gts, thr):
    """Greedy (confidence-ordered) matching. preds: [(box, conf)], gts: [box]."""
    used, out = set(), []
    for pb, pc in sorted(preds, key=lambda t: -t[1]):
        best, bj = 0, -1
        for j, g in enumerate(gts):
            if j not in used:
                v = iou_xywh(pb, g)
                if v > best:
                    best, bj = v, j
        if best >= thr:
            used.add(bj)
            out.append((pb, pc, bj, best))
        else:
            out.append((pb, pc, -1, best))
    return out, [j for j in range(len(gts)) if j not in used]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default=None, help="prepared dataset folder (default: data.out_dir in config)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=None, help="operating confidence threshold")
    ap.add_argument("--no-measure", action="store_true")
    ap.add_argument("--tune-conf", action="store_true",
                    help="choose the confidence threshold that maximises F1 on the VALIDATION set, then apply it to test")
    ap.add_argument("--limit", type=int, default=0, help="debug: evaluate only N images")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.data_dir:
        cfg["data"]["out_dir"] = args.data_dir
    ds = Path(cfg["data"]["out_dir"])
    dev = args.device if args.device is not None else cfg["train"]["device"]
    imgsz = cfg["train"]["imgsz"]
    conf_thr = args.conf if args.conf is not None else cfg["infer"]["conf"]
    name = args.name or Path(args.weights).parent.parent.name
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    model = build(args.weights)
    imgsz = int((getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz") or imgsz)  # = training size

    gt = json.load(open(ds / f"coco_{args.split}.json"))

    # ---------- 1. standard COCO-style metrics (Ultralytics validator), overall + per source
    src_map = []
    if not args.limit:
        m = model.val(data=str(ds / "data.yaml"), split=args.split, imgsz=imgsz, device=dev,
                      project=str(out.resolve()), name="val", exist_ok=True, plots=True)
        coco = dict(mAP50=m.box.map50, mAP50_95=m.box.map, precision=m.box.mp, recall=m.box.mr)
        by_src = defaultdict(list)
        for im in gt["images"]:
            by_src[im.get("source", "all")].append(str((ds / "images" / args.split / im["file_name"]).resolve()))
        if len(by_src) > 1:
            ldir = out / "source_lists"
            ldir.mkdir(exist_ok=True)
            for s, files in by_src.items():
                (ldir / f"{s}.txt").write_text("\n".join(files))
                yaml.safe_dump(dict(path=str(ds.resolve()), train=str((ldir / f"{s}.txt").resolve()),
                                    val=str((ldir / f"{s}.txt").resolve()), names={0: "stenosis"}),
                               open(ldir / f"{s}.yaml", "w"))
                ms = model.val(data=str(ldir / f"{s}.yaml"), split="val", imgsz=imgsz, device=dev,
                               project=str(out.resolve()), name=f"val_{s}", exist_ok=True, plots=False)
                src_map.append(dict(source=s, mAP50=ms.box.map50, mAP50_95=ms.box.map))
    else:
        coco = {}

    # ---------- 2. per-image predictions + timing
    gts = {i["id"]: [] for i in gt["images"]}
    for a in gt["annotations"]:
        gts[a["image_id"]].append(a["bbox"])
    images = gt["images"][: args.limit] if args.limit else gt["images"]
    img_dir = ds / "images" / args.split

    # warm-up
    w0 = cv2.imread(str(img_dir / images[0]["file_name"]))
    for _ in range(3):
        model.predict(w0, imgsz=imgsz, conf=0.001, device=dev, verbose=False)

    if args.tune_conf:
        gv = json.load(open(ds / "coco_val.json"))
        gtv = {i["id"]: [] for i in gv["images"]}
        for a in gv["annotations"]:
            gtv[a["image_id"]].append(a["bbox"])
        vims = gv["images"][:: max(1, len(gv["images"]) // 1200)]          # up to ~1200 val images
        cache = []
        for im in vims:
            r = model.predict(str(ds / "images" / "val" / im["file_name"]), imgsz=imgsz, conf=0.01,
                              device=dev, verbose=False)[0]
            b = r.boxes
            cache.append(([((x1, y1, x2 - x1, y2 - y1), float(c)) for (x1, y1, x2, y2), c in
                           zip(b.xyxy.cpu().numpy().tolist(), b.conf.cpu().numpy().tolist())], gtv[im["id"]]))
        best = (-1, conf_thr)
        tune_rows = []
        for t in np.round(np.arange(0.05, 0.91, 0.05), 2):
            tp = fp = fn = 0
            for preds, g in cache:
                mm, ms_ = match([p for p in preds if p[1] >= t], g, 0.5)
                k = sum(1 for x in mm if x[2] >= 0)
                tp += k; fp += len(mm) - k; fn += len(ms_)
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            tune_rows.append(dict(conf=t, tp=tp, fp=fp, fn=fn, F1=f1))
            if f1 > best[0]:
                best = (f1, float(t))
        pd.DataFrame(tune_rows).to_csv(out / "conf_tuning_val.csv", index=False)
        conf_thr = best[1]
        print(f"[tune-conf] best validation F1 = {best[0]:.3f} at confidence {conf_thr:.2f} -> used for test")

    rows, det_ms, meas_ms, examples = [], [], [], []
    tp50 = fp50 = fn50 = 0
    tp30 = fp30 = fn30 = 0
    ious, hits = [], []
    for im in images:
        img = cv2.imread(str(img_dir / im["file_name"]))
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=imgsz, conf=0.001, device=dev, verbose=False)[0]
        det_ms.append((time.perf_counter() - t0) * 1000)
        b = r.boxes
        preds = [((x1, y1, x2 - x1, y2 - y1), float(c)) for (x1, y1, x2, y2), c in
                 zip(b.xyxy.cpu().numpy().tolist(), b.conf.cpu().numpy().tolist())]
        op = [p for p in preds if p[1] >= conf_thr]
        g = gts[im["id"]]
        matched, missed = match(op, g, 0.5)
        tp = sum(1 for t in matched if t[2] >= 0)
        tp50 += tp
        fp50 += len(matched) - tp
        fn50 += len(missed)
        ious += [t[3] for t in matched if t[2] >= 0]
        # lenient localisation criteria (stenosis extent is subjective; boxes are tiny)
        m30, miss30 = match(op, g, 0.3)
        t30 = sum(1 for t in m30 if t[2] >= 0)
        tp30 += t30
        fp30 += len(m30) - t30
        fn30 += len(miss30)
        top = max(op, key=lambda t: t[1]) if op else None
        if top is not None:
            cx, cy = top[0][0] + top[0][2] / 2, top[0][1] + top[0][3] / 2
            hit = any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in g)
        else:
            hit = False
        hits.append(hit)
        rec = dict(image=im["file_name"], source=im.get("source", "all"), patient=im.get("patient", ""),
                   video=im.get("video", ""), negative=bool(im.get("negative", len(g) == 0)),
                   tp30=t30, fp30=len(m30) - t30, fn30=len(miss30), center_hit=hit,
                   n_gt=len(g), n_pred=len(op), tp=tp, fp=len(matched) - tp,
                   fn=len(missed), top_conf=top[1] if top else np.nan,
                   gt_area=float(np.mean([x[2] * x[3] for x in g])) if g else np.nan,
                   best_iou=max([t[3] for t in matched], default=0.0),
                   pred_boxes=json.dumps([[round(v, 1) for v in p[0]] + [round(p[1], 3)] for p in op]),
                   gt_boxes=json.dumps(g))
        if not args.no_measure and top is not None:
            t1 = time.perf_counter()
            mp = measure_stenosis(img, top[0], cfg["measure"]["mm_per_pixel"], cfg["measure"]["box_margin"],
                                  cfg["measure"]["min_crop"], cfg["measure"]["lesion_threshold"])
            meas_ms.append((time.perf_counter() - t1) * 1000)
            rec.update({f"ai_{k}": v for k, v in mp.row().items()})
            # same measurement on the matched ground-truth box (pipeline robustness)
            mt = [t for t in matched if t[0] == top[0] and t[2] >= 0]
            if mt:
                mg = measure_stenosis(img, g[mt[0][2]], cfg["measure"]["mm_per_pixel"])
                rec["gtbox_percent_diameter_stenosis"] = mg.percent_diameter_stenosis
                rec["gtbox_lesion_length_px"] = mg.lesion_length_px
            if len(examples) < 16 and len(rows) % max(1, len(images) // 16) == 0:
                o = draw(img, top[0], top[1], mp)
                for gb in g:
                    x, y, w, h = map(int, gb)
                    cv2.rectangle(o, (x, y), (x + w, y + h), (0, 255, 0), 1)
                examples.append(cv2.resize(o, (320, 320)))
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"predictions_{args.split}.csv", index=False)

    P = tp50 / max(tp50 + fp50, 1)
    R = tp50 / max(tp50 + fn50, 1)
    summary = dict(model=name, split=args.split, images=len(df), conf_threshold=conf_thr, **coco,
                   precision_at_conf=P, recall_at_conf=R, F1_at_conf=2 * P * R / max(P + R, 1e-9),
                   FDR_at_conf=1 - P, mean_IoU_matched=float(np.mean(ious)) if ious else np.nan,
                   mean_Dice_matched=float(np.mean([2 * i / (1 + i) for i in ious])) if ious else np.nan,
                   image_detection_rate=float((df[~df.negative].tp > 0).mean()) if (~df.negative).any() else np.nan,
                   precision_iou30=tp30 / max(tp30 + fp30, 1), recall_iou30=tp30 / max(tp30 + fn30, 1),
                   F1_iou30=2 * tp30 / max(2 * tp30 + fp30 + fn30, 1),
                   top1_center_hit_rate=float(df[~df.negative].center_hit.mean()) if (~df.negative).any() else np.nan,
                   recall_small=float(df[df.gt_area < 32 ** 2].tp.sum() / max(df[df.gt_area < 32 ** 2].n_gt.sum(), 1)),
                   recall_medium=float(df[df.gt_area >= 32 ** 2].tp.sum() / max(df[df.gt_area >= 32 ** 2].n_gt.sum(), 1)),
                   detect_ms_median=float(np.median(det_ms)), detect_ms_p95=float(np.percentile(det_ms, 95)))
    if meas_ms:
        tot = np.array(det_ms[: len(meas_ms)]) + np.array(meas_ms)
        summary.update(measure_ms_median=float(np.median(meas_ms)), measure_ms_p95=float(np.percentile(meas_ms, 95)),
                       end_to_end_ms_median=float(np.median(tot)), end_to_end_fps=float(1000 / np.median(tot)),
                       realtime_at_15fps=bool(np.percentile(tot, 95) <= 1000 / 15),
                       measurement_success_rate=float(df.ai_ok.mean()) if "ai_ok" in df else np.nan)
        if "gtbox_percent_diameter_stenosis" in df:
            ag = agreement(df.gtbox_percent_diameter_stenosis, df.ai_percent_diameter_stenosis)
            summary.update({f"DS_predbox_vs_gtbox_{k}": v for k, v in ag.items()
                            if k in ("n", "MAE", "bias", "ICC21")})
    # ---------- 3. normal / negative frames: specificity (frame, video, patient level)
    neg = df[df.negative]
    if len(neg):
        summary.update(negative_frames=int(len(neg)),
                       frame_specificity=float((neg.n_pred == 0).mean()),
                       false_positives_per_negative_frame=float(neg.n_pred.mean()))
        # cine run = video: called positive when the lesion is flagged in >= half of its key frames
        vid = df.groupby("video").agg(truth=("negative", lambda s: int((~s).any())),
                                      flagged=("n_pred", lambda s: float((s > 0).mean())))
        vid["pred"] = (vid.flagged >= 0.5).astype(int)
        tp_ = int(((vid.truth == 1) & (vid.pred == 1)).sum()); tn_ = int(((vid.truth == 0) & (vid.pred == 0)).sum())
        fp_ = int(((vid.truth == 0) & (vid.pred == 1)).sum()); fn_ = int(((vid.truth == 1) & (vid.pred == 0)).sum())
        summary.update(videos=len(vid), video_sensitivity=tp_ / max(tp_ + fn_, 1),
                       video_specificity=tn_ / max(tn_ + fp_, 1), video_accuracy=(tp_ + tn_) / max(len(vid), 1))
        vid.to_csv(out / f"video_level_{args.split}.csv")

    # ---------- 4. per-source breakdown (thesis table: one row per dataset)
    per = []
    for s_, g_ in df.groupby("source"):
        pos = g_[~g_.negative]
        tp5, fp5, fn5 = g_.tp.sum(), g_.fp.sum(), g_.fn.sum()
        tp3, fp3, fn3 = g_.tp30.sum(), g_.fp30.sum(), g_.fn30.sum()
        row = dict(source=s_, images=len(g_), positive_frames=len(pos), negative_frames=int(g_.negative.sum()),
                   precision50=tp5 / max(tp5 + fp5, 1), recall50=tp5 / max(tp5 + fn5, 1),
                   F1_50=2 * tp5 / max(2 * tp5 + fp5 + fn5, 1), F1_30=2 * tp3 / max(2 * tp3 + fp3 + fn3, 1),
                   center_hit=float(pos.center_hit.mean()) if len(pos) else np.nan,
                   frame_specificity=float((g_[g_.negative].n_pred == 0).mean()) if g_.negative.any() else np.nan)
        row.update(next((d for d in src_map if d["source"] == s_), {}))
        per.append(row)
    per = pd.DataFrame(per)
    per.to_csv(out / f"per_source_{args.split}.csv", index=False)

    pd.Series(summary).to_csv(out / f"summary_{args.split}.csv", header=["value"])
    json.dump(summary, open(out / f"summary_{args.split}.json", "w"), indent=2, default=float)
    if examples:
        while len(examples) % 4:
            examples.append(np.zeros_like(examples[0]))
        grid = np.vstack([np.hstack(examples[i:i + 4]) for i in range(0, len(examples), 4)])
        cv2.imwrite(str(out / f"examples_{args.split}.jpg"), grid)
    for k, v in summary.items():
        print(f"{k:38s} {v:.4f}" if isinstance(v, float) else f"{k:38s} {v}")
    print("\nPer dataset:")
    print(per.round(3).to_string(index=False))
    print(f"\nResults in {out.resolve()}")


if __name__ == "__main__":
    main()
