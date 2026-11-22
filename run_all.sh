#!/usr/bin/env bash
# Full experiment pipeline on a GPU machine. Run from the project root:
#   bash run_all.sh
set -e

# 0. data (put the Roboflow export in ./data with train/ valid/ test/)
python src/prepare_data.py

# 1. Objective 2 technical validation (no GPU needed)
python src/phantom_validation.py --n 300 --out results/phantom

# 2. Objective 1 — train proposed model and baseline (same patient-level split)
python src/train.py --model rtdetr-l.pt --name rtdetr-l
python src/train.py --model yolo11m.pt  --name yolo11m

# 3. test-set evaluation: detection + measurement + speed
python src/evaluate.py --weights runs/rtdetr-l/weights/best.pt --name rtdetr-l
python src/evaluate.py --weights runs/yolo11m/weights/best.pt  --name yolo11m

# 4. 5-fold patient-grouped cross-validation of the proposed model
python src/train.py --model rtdetr-l.pt --name rtdetr-l --cv

# 5. real-time demo on every held-out test cine sequence
for s in $(python -c "import pandas as pd; d=pd.read_csv('dataset/split_manifest.csv'); d=d[d.split=='test']; print(' '.join(sorted(set(d.stem.str.rsplit('_',n=1).str[0]))))"); do
  python src/infer_realtime.py --weights runs/rtdetr-l/weights/best.pt --source seq:$s --out results/realtime
done

# 6. expert review pack for Dr. Samiullah
python src/expert_review.py make --pred results/rtdetr-l/predictions_test.csv --n 200
echo "Send results/review_pack to the expert; afterwards run:"
echo "  python src/expert_review.py analyze --pack results/review_pack"
