@echo off
REM ============================================================================
REM  StenoHybrid-S experiments (proposed hybrid CNN-transformer model + ablation)
REM  Run from the project root:  D:\PhD Project\ClaudProject\cad_plaque_ai
REM  Uses the SAME patient-level split and fast.yaml lists as the earlier runs.
REM ============================================================================

REM ---- 0. environment check: stops here if torch / ultralytics / GPU are missing
python -c "import torch, ultralytics, yaml, cv2; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__, '| CUDA GPU:', torch.cuda.is_available())" || (echo. & echo ERROR: packages missing. Activate the environment used for earlier training, e.g.  conda activate ENVNAME & exit /b 1)
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || (echo. & echo ERROR: no CUDA GPU visible to torch. Install the CUDA build of torch. & exit /b 1)

REM ---- 1. build the models and load pretrained weights (a few minutes, CPU is fine)
python src\init_weights.py --cfg models\stenohybrid-s.yaml        --out stenohybrid-s-init.pt || exit /b 1
python src\init_weights.py --cfg models\stenohybrid-s-noaifi.yaml --out stenohybrid-s-noaifi-init.pt || exit /b 1

REM ---- 2. MAIN RUN: proposed model, settings identical to rtdetr-l-v2 (about 18-24 h)
python src\train.py --model stenohybrid-s-init.pt --fast --imgsz 640 --freeze 0 --epochs 80 --patience 20 --batch 4 --name stenohybrid-s || exit /b 1

REM ---- 3. ABLATION: same model without the transformer encoder (about 18-24 h)
python src\train.py --model stenohybrid-s-noaifi-init.pt --fast --imgsz 640 --freeze 0 --epochs 80 --patience 20 --batch 4 --name stenohybrid-s-noaifi || exit /b 1

REM ---- 4. CNN BASELINE under the same settings (about 3-5 h)
python src\train.py --model yolo11s.pt --fast --imgsz 640 --freeze 0 --epochs 80 --patience 20 --batch 4 --name yolo11s-v2 || exit /b 1

REM ---- 5. test-set evaluation (rtdetr-l-v2 already exists)
python src\evaluate.py --weights runs\stenohybrid-s\weights\best.pt        --name stenohybrid-s || exit /b 1
python src\evaluate.py --weights runs\stenohybrid-s-noaifi\weights\best.pt --name stenohybrid-s-noaifi || exit /b 1
python src\evaluate.py --weights runs\yolo11s-v2\weights\best.pt           --name yolo11s-v2 || exit /b 1

REM ---- 6. latency on the laptop GPU
python src\benchmark_speed.py --weights runs\stenohybrid-s\weights\best.pt runs\rtdetr-l-v2\weights\best.pt runs\yolo11s-v2\weights\best.pt || exit /b 1

REM ---- 7. OPTIONAL if time allows: two more seeds of the proposed model
REM python src\train.py --model stenohybrid-s-init.pt --fast --imgsz 640 --freeze 0 --epochs 80 --patience 20 --batch 4 --seed 7  --name stenohybrid-s-seed7
REM python src\train.py --model stenohybrid-s-init.pt --fast --imgsz 640 --freeze 0 --epochs 80 --patience 20 --batch 4 --seed 13 --name stenohybrid-s-seed13

echo Done. Zip the folders runs\stenohybrid-s, runs\stenohybrid-s-noaifi, runs\yolo11s-v2 (without weights) and results\ and send them.
