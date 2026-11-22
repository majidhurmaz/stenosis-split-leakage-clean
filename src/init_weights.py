"""Build StenoHybrid-S and initialise it from two COCO-pretrained models.

Backbone layers 0-10      <- yolo11s.pt  (same layer index and shape)
Encoder, neck, decoder    <- rtdetr-l.pt (layer index shifted by +1; only tensors whose shapes match)
Layers with no compatible source (the three input projections) stay randomly initialised.

Usage:
    python src/init_weights.py --cfg models/stenohybrid-s.yaml --out stenohybrid-s-init.pt
    python src/init_weights.py --cfg models/stenohybrid-s-noaifi.yaml --out stenohybrid-s-noaifi-init.pt
"""
import argparse, re
from pathlib import Path
import torch
from ultralytics import RTDETR


def sd_of(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck.get("ema") or ck["model"]
    return m.float().state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="models/stenohybrid-s.yaml")
    ap.add_argument("--yolo", default="yolo11s.pt")
    ap.add_argument("--rtdetr", default="rtdetr-l.pt")
    ap.add_argument("--out", default="stenohybrid-s-init.pt")
    a = ap.parse_args()

    model = RTDETR(a.cfg)
    net = model.model
    tgt = net.state_dict()
    from ultralytics.utils.downloads import attempt_download_asset
    ysd, rsd = sd_of(attempt_download_asset(a.yolo)), sd_of(attempt_download_asset(a.rtdetr))

    new, src_of = {}, {}
    for k, v in tgt.items():
        m = re.match(r"model\.(\d+)\.(.*)", k)
        if not m:
            continue
        i, rest = int(m.group(1)), m.group(2)
        if i <= 10:                                   # CNN backbone from YOLO11s
            sk = f"model.{i}.{rest}"
            if sk in ysd and ysd[sk].shape == v.shape:
                new[k] = ysd[sk]; src_of[k] = "yolo11s"
        else:                                         # encoder / neck / decoder from RT-DETR-L
            sk = f"model.{i - 1}.{rest}"
            if sk in rsd and rsd[sk].shape == v.shape:
                new[k] = rsd[sk]; src_of[k] = "rtdetr-l"
            # class-score heads do not match (COCO 80 classes, ours 1) and stay random
    tgt.update(new)
    net.load_state_dict(tgt)

    n_all = len([k for k in tgt if k.startswith("model.")])
    n_y = sum(1 for s in src_of.values() if s == "yolo11s")
    n_r = sum(1 for s in src_of.values() if s == "rtdetr-l")
    params = sum(p.numel() for p in net.parameters())
    print(f"{a.cfg}: {params/1e6:.2f} M parameters")
    print(f"tensors initialised: {n_y} from yolo11s, {n_r} from rtdetr-l, {n_all - n_y - n_r} random (of {n_all})")
    missing = sorted({int(re.match(r'model\.(\d+)', k).group(1)) for k in tgt if k.startswith('model.') and k not in new})
    print("layers with randomly initialised tensors:", missing)

    model.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
