@echo off
REM ============================================================================
REM  Paper 2 (data leakage): YOLO11s under the FRAME-LEVEL split
REM  Same settings as runs\yolo11s-fast (patient-level) and runs\rtdetr-l-fast-frame
REM  Run from: D:\PhD Project\ClaudProject\cad_plaque_ai   (conda activate plaque)
REM  Time: about 1.5 hours training + a few minutes evaluation
REM ============================================================================
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || (echo ERROR: no CUDA GPU. Run: conda activate plaque & exit /b 1)

python src\train.py --data-dir dataset_frame --model yolo11s.pt --fast --name yolo11s-fast-frame || exit /b 1
python src\evaluate.py --data-dir dataset_frame --weights runs\yolo11s-fast-frame\weights\best.pt --name yolo11s-fast-frame --tune-conf || exit /b 1
echo.
echo DONE. Results in results\yolo11s-fast-frame
