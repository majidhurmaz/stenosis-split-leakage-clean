@echo off
REM Re-evaluate the three new models with the confidence threshold tuned on the validation set
REM (same procedure as rtdetr-l-v2), so that F1 and specificity are compared fairly.
REM Run AFTER run_stenohybrid.bat has finished.
python src\evaluate.py --weights runs\stenohybrid-s\weights\best.pt --name stenohybrid-s --tune-conf || exit /b 1
python src\evaluate.py --weights runs\stenohybrid-s-noaifi\weights\best.pt --name stenohybrid-s-noaifi --tune-conf || exit /b 1
python src\evaluate.py --weights runs\yolo11s-v2\weights\best.pt --name yolo11s-v2 --tune-conf || exit /b 1
echo Done.
