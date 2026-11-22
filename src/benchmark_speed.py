"""Clean real-time benchmark: per-frame latency of each detector on the GPU (and the measurement step).

Run with NOTHING else using the GPU/CPU. Uses N test frames, batch size 1 (as in a live cine stream),
warm-up first, CUDA-synchronised timing, FP32 and FP16.

  python src/benchmark_speed.py --weights runs/rtdetr-l-v2/weights/best.pt runs/yolo11s-fast/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from measure import measure_stenosis
from train import build


def bench(model, frames, imgsz, half, device):
    for f in frames[:20]:                                          # warm-up
        model.predict(f, imgsz=imgsz, device=device, half=half, verbose=False)
    t_pre, t_inf, t_post, wall = [], [], [], []
    for f in frames:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(f, imgsz=imgsz, device=device, half=half, conf=0.25, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1000)
        t_pre.append(r.speed["preprocess"]); t_inf.append(r.speed["inference"]); t_post.append(r.speed["postprocess"])
    q = lambda a, p: float(np.percentile(a, p))
    return dict(preprocess_ms=q(t_pre, 50), inference_ms_median=q(t_inf, 50), inference_ms_p95=q(t_inf, 95),
                postprocess_ms=q(t_post, 50), end_to_end_detector_ms_median=q(wall, 50),
                end_to_end_detector_ms_p95=q(wall, 95), detector_fps=1000 / q(wall, 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="results/speed_benchmark.csv")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ds = Path(cfg["data"]["out_dir"])
    files = sorted((ds / "images" / "test").glob("*.jpg"))
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(a.n, len(files))).astype(int)]
    frames = [cv2.imread(str(f)) for f in files]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    rows = []
    for w in a.weights:
        model = build(w)
        imgsz = int((getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz") or cfg["train"]["imgsz"])
        for half in (False, True):
            if half and not torch.cuda.is_available():
                continue
            r = bench(model, frames, imgsz, half, a.device)
            rows.append(dict(model=Path(w).parent.parent.name, imgsz=imgsz, precision="FP16" if half else "FP32",
                             device=gpu, frames=len(frames), **r))
            print(rows[-1])
    # measurement step (CPU) on detected boxes of the first model
    model = build(a.weights[0])
    imgsz = rows[0]["imgsz"]
    mt = []
    for f in frames[:150]:
        r = model.predict(f, imgsz=imgsz, device=a.device, conf=0.25, verbose=False)[0]
        if len(r.boxes):
            x1, y1, x2, y2 = r.boxes.xyxy[int(r.boxes.conf.argmax())].tolist()
            t0 = time.perf_counter()
            measure_stenosis(f, (x1, y1, x2 - x1, y2 - y1))
            mt.append((time.perf_counter() - t0) * 1000)
    df = pd.DataFrame(rows)
    if mt:
        df["measurement_ms_median_cpu"] = float(np.median(mt))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.round(1).to_string(index=False))
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
