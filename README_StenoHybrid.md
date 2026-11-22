# StenoHybrid-S: proposed hybrid CNN–transformer model

## Install (copy into the existing project)

| File in this package | Where it goes in `cad_plaque_ai` |
|---|---|
| `models/stenohybrid-s.yaml` | `models/` (new folder) |
| `models/stenohybrid-s-noaifi.yaml` | `models/` |
| `src/init_weights.py` | `src/` (new file) |
| `src/train.py` | `src/`, replacing the old one |
| `run_stenohybrid.bat` | project root |

The `train.py` changes are small:

- names containing "stenohybrid" are trained with the RT-DETR trainer;
- a `--seed` option is added. It changes only the training seed, not the data split.

## Architecture

| Block | What it does | Source of initial weights |
|---|---|---|
| CNN backbone (YOLO11s: Conv, C3k2, SPPF, C2PSA), layers 0–10 | Extracts local features at three scales: P3 (1/8), P4 (1/16), P5 (1/32) | yolo11s.pt (COCO) |
| Projection and tokenisation, layer 11 | 1×1 conv to 256 channels; the P5 map is flattened into 400 tokens at 640 px | random |
| Transformer encoder (AIFI), layer 12 | Multi-head self-attention (8 heads) and feed-forward network over all P5 tokens, with 2D sine–cosine positional encoding | rtdetr-l.pt |
| CNN cross-scale fusion (RepC3), layers 13–28 | Top-down and bottom-up fusion that returns global context to the fine P3 map, where small stenoses are found | rtdetr-l.pt, except the P3 and P4 projections |
| ROI focus (query selection) and transformer decoder, layer 29 | Scores every multi-scale token, keeps the 300 most stenosis-like as queries, refines them with 6 decoder layers, and outputs boxes and scores without NMS | rtdetr-l.pt, except the 1-class score heads |

- **Parameters:** 24.0 M, against about 32 M for RT-DETR-L and about 9.4 M for YOLO11s.
- **Initialisation:** 683 of 701 tensors start from pretrained weights.
- **Status:** built and checked; one training epoch on test data ran without error.

## Ablation and comparison plan (Paper 3)

| Run | Purpose |
|---|---|
| `yolo11s-v2` | Pure CNN baseline |
| `rtdetr-l-v2` (already trained) | Existing hybrid baseline |
| `stenohybrid-s` | Proposed model |
| `stenohybrid-s-noaifi` | Proposed model with the transformer encoder replaced by a conv block. Shows how much the transformer contributes |

All runs use the same patient-level split, 640 px, AdamW with learning rate 1e-4, up to 80 epochs, patience 20 and batch 4.
