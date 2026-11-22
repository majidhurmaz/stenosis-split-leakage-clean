"""Real-time stenosis/plaque detection + measurement on cine angiography.

Processes a cine run frame-by-frame exactly as it would be in the cath-lab:
  frame -> CLAHE -> transformer detector -> QCA measurement -> overlay -> display/record
and reports per-frame latency against the acquisition frame rate (7-15 fps).

Sources:
  --source run.mp4 / run.avi          video file
  --source run.dcm                    multi-frame DICOM (calibration read automatically)
  --source path/to/frames/            folder of images (sorted by name)
  --source seq:dan_14_029_7           key frames of a sequence in the prepared dataset
  --source ".../CADICA/selectedVideos/p20/v3/input" --fps 10   a full CADICA cine run
  --source 0                          live capture device (frame grabber)

  python src/infer_realtime.py --weights runs/rtdetr-l/weights/best.pt --source seq:dan_14_029_7
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from evaluate import iou_xywh
from measure import draw, measure_stenosis
from preprocess import enhance, read_dicom_cine
from train import build


def frames_from(source: str, dataset_dir: Path):
    """Yield (frame_bgr, already_enhanced, meta) and return fps/mm_per_pixel via first meta."""
    if source.startswith("seq:"):
        pref = source[4:]
        files = sorted(dataset_dir.glob(f"images/*/{pref}_*.jpg"), key=lambda p: int(p.stem.split("_")[-1]))
        if not files:
            raise SystemExit(f"No frames found for sequence {pref}")
        for f in files:
            yield cv2.imread(str(f)), True, dict(name=f.name, fps=15.0, mmpp=None)
        return
    p = Path(source)
    if p.is_dir():
        for f in sorted(x for x in p.iterdir() if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif")):
            yield cv2.imread(str(f)), False, dict(name=f.name, fps=15.0, mmpp=None)
        return
    if p.suffix.lower() in (".dcm", "") and p.exists():
        frames, mmpp, fps = read_dicom_cine(str(p))
        for i, f in enumerate(frames):
            yield cv2.cvtColor(f, cv2.COLOR_GRAY2BGR), False, dict(name=f"{p.stem}_{i:04d}", fps=fps or 15.0, mmpp=mmpp)
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f, False, dict(name=f"frame_{i:05d}", fps=fps, mmpp=None)
        i += 1


class LesionTrack:
    """Very small IoU tracker: keeps the lesion identity across frames and smooths
    the measurement over the last N frames (cardiac motion / contrast wash-in)."""

    def __init__(self, n=5):
        self.box, self.hist, self.miss = None, deque(maxlen=n), 0

    def update(self, box, ds):
        if self.box is not None and iou_xywh(self.box, box) < 0.1:
            self.hist.clear()
        self.box, self.miss = box, 0
        if np.isfinite(ds):
            self.hist.append(ds)
        return float(np.median(self.hist)) if self.hist else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default=None, help="prepared dataset folder (default: data.out_dir in config)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="results/realtime")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--mm-per-pixel", type=float, default=None)
    ap.add_argument("--fps", type=float, default=None, help="acquisition frame rate of the source (CADICA = 10)")
    ap.add_argument("--show", action="store_true", help="live display window")
    ap.add_argument("--half", action="store_true", help="FP16 inference on GPU")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.data_dir:
        cfg["data"]["out_dir"] = args.data_dir
    dev = args.device if args.device is not None else cfg["train"]["device"]
    conf = args.conf if args.conf is not None else cfg["infer"]["conf"]
    imgsz = cfg["train"]["imgsz"]
    pp, mc = cfg["preprocess"], cfg["measure"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build(args.weights)
    imgsz = int((getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz") or imgsz)  # = training size
    tag = args.source.replace("seq:", "").replace("/", "_").strip("_") or "run"

    writer, rows, track = None, [], LesionTrack()
    for k, (frame, enhanced, meta) in enumerate(frames_from(args.source, Path(cfg["data"]["out_dir"]))):
        if args.fps:
            meta["fps"] = args.fps
        t0 = time.perf_counter()
        img = frame if enhanced else enhance(frame, pp["clahe"], pp["clahe_clip"], pp["clahe_tile"], pp["denoise"])
        t1 = time.perf_counter()
        r = model.predict(img, imgsz=imgsz, conf=conf, device=dev, verbose=False, **({"half": True} if args.half else {}))[0]
        t2 = time.perf_counter()
        mmpp = args.mm_per_pixel or meta["mmpp"] or mc["mm_per_pixel"]
        vis = img.copy()
        rec = dict(frame=k, name=meta["name"], n_det=len(r.boxes))
        if len(r.boxes):
            j = int(r.boxes.conf.argmax())
            x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
            box, c = (x1, y1, x2 - x1, y2 - y1), float(r.boxes.conf[j])
            m = measure_stenosis(img, box, mmpp, mc["box_margin"], mc["min_crop"], mc["lesion_threshold"])
            smooth = track.update(box, m.percent_diameter_stenosis)
            vis = draw(img, box, c, m)
            for jj in range(len(r.boxes)):           # secondary detections: box only
                if jj != j:
                    a1, b1, a2, b2 = map(int, r.boxes.xyxy[jj].tolist())
                    cv2.rectangle(vis, (a1, b1), (a2, b2), (0, 140, 255), 1)
            rec.update(conf=c, box=[round(v, 1) for v in box], ds_smoothed=smooth, **m.row())
        t3 = time.perf_counter()
        lat = dict(pre_ms=(t1 - t0) * 1e3, det_ms=(t2 - t1) * 1e3, meas_ms=(t3 - t2) * 1e3, total_ms=(t3 - t0) * 1e3)
        rec.update(lat)
        rows.append(rec)
        hud = f"{lat['total_ms']:.0f} ms/frame ({1000 / lat['total_ms']:.1f} fps)   acq {meta['fps']:.0f} fps"
        if "ds_smoothed" in rec and np.isfinite(rec["ds_smoothed"]):
            hud += f"   DS(smoothed) {rec['ds_smoothed']:.0f}%"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(vis, hud, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(str(out / f"{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     float(meta["fps"]), (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        if args.show:
            cv2.imshow("CAD plaque AI", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if writer:
        writer.release()
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{tag}_frames.csv", index=False)
    tot = df.total_ms.iloc[1:] if len(df) > 1 else df.total_ms
    acq = meta["fps"]
    print(f"Frames: {len(df)}  detected in {(df.n_det > 0).mean() * 100:.1f}% of frames")
    print(f"Latency median {tot.median():.1f} ms (det {df.det_ms.median():.1f}, measure {df.meas_ms.median():.1f}), "
          f"p95 {tot.quantile(0.95):.1f} ms -> {1000 / tot.median():.1f} fps; "
          f"real-time at {acq:.0f} fps: {'YES' if tot.quantile(0.95) <= 1000 / acq else 'NO'}")
    print(f"Video: {out / (tag + '.mp4')}")


if __name__ == "__main__":
    main()
