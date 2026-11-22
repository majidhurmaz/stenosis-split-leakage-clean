# CAD Plaque AI — real-time coronary stenosis/plaque detection and measurement

Implementation of the PhD proposal *"Automated Detection and Measurement of Coronary
Artery Plaque Using AI Transformer"*.

| Proposal objective | Implementation | Script |
|---|---|---|
| **Objective 1, plaque detection**: transformer model that locates plaque on angiograms | **RT-DETR** (Real-Time DEtection TRansformer): a CNN backbone with a transformer encoder and decoder, i.e. the hybrid CNN + transformer from the proposal. **YOLO11** (CNN only) is the baseline. | `src/train.py` |
| **Objective 2, plaque measurement**: area and volume without eyeballing | QCA-style measurement inside each detection: centreline, edge diameter plus videodensitometry, interpolated reference, then %DS, %AS, MLD, lesion length, projected plaque area and estimated plaque volume | `src/measure.py` |
| Pre-processing (normalisation, CLAHE, DICOM → image) | | `src/preprocess.py`, `src/prepare_data.py` |
| Evaluation (accuracy, precision, recall, F1, Dice, IoU, processing time) | | `src/evaluate.py` |
| k-fold cross-validation | 5-fold, grouped by patient | `src/train.py --cv` |
| Real-time cine angiography integration | Frame-by-frame pipeline, latency compared with 7–15 fps acquisition, video output | `src/infer_realtime.py` |
| Comparison with expert (Kappa, ICC, paired t-test, MAE, RMSE) | Two-stage blind review pack for Dr. Samiullah, plus statistics | `src/expert_review.py` |
| Technical validation of measurement | Digital phantoms with known ground truth | `src/phantom_validation.py` |

---

## 1. Datasets (version 2: three public sources combined)

`prepare_data.py` merges three public datasets into one leakage-free dataset. Paths are set in `configs/default.yaml`.

| Source | Content | Split | Labelling |
|---|---|---|---|
| Danilov et al. 2021 | 64 patients, 8,325 frames, boxes | patient level 70/15/15 | every box = stenosis |
| ARCADE 2023 (stenosis task) | 1,500 images, polygon **masks** | official train/val/test (no patient IDs) | every mask = stenosis; masks saved for Dice/IoU |
| CADICA 2024 | 42 patients, cine key frames, graded boxes, lesion-free videos | patient level, stratified by patient category | box = stenosis if grade **≥ 70%**; <20% = background; frames with 20–69% lesions dropped; lesion-free videos = **negative (normal) frames** |

CADICA patient categories (metadata.xlsx): **5 normal** (no vessel affected, max stenosis <20%), **5 non-obstructive** (no vessel affected, 20–50%), 32 obstructive CAD. Each category is spread over train, val and test. `dataset/eda/` holds the counts per source and split (`split_summary.csv`) and the patient table (`cadica_patients.csv`, `cadica_patient_categories.csv`).

`evaluate.py` reports every metric overall and **per source** (`per_source_test.csv`), plus **specificity** on the normal frames at frame and video level (`video_level_test.csv`).

## 1b. Original Danilov dataset: findings

**Source.** The supplied data is the public *Angiographic dataset for stenosis detection*
(Danilov, Klyshnikov, Kutikhin, Gerget, Frangi, Ovcharenko), Mendeley Data V2, 2021,
doi:10.17632/ydrm75xywg.2, licence CC BY 4.0. It is the dataset of Danilov et al.,
*"Real-time coronary artery stenosis detection based on modern neural networks"*,
Scientific Reports 11:7582 (2021). This copy was exported through Roboflow at 576×576 px.
Cite both references in the thesis.

| | |
|---|---|
| Images | 8,325 (all single-frame JPG, 576×576, grayscale) |
| Patients (IDs in file names `14_<patient>_<series>_<frame>`) | 64 |
| Annotation | 1 class "Stenosis", **bounding boxes only (no segmentation masks)** |
| Boxes per image | 1 (one image has 2) |
| Normal (stenosis-free) frames | **none** |

**1. Data leakage in the supplied split.** All 64 patients appear in train, valid *and*
test, because Roboflow split individual frames at random. Neighbouring cine frames are
almost identical, so this split inflates every metric. `prepare_data.py` therefore
**re-splits by patient** (70/15/15 by image count, seed 42):

| split | images | patients | cine sequences |
|---|---|---|---|
| train | 5,813 | 46 | 158 |
| val | 1,254 | 9 | 25 |
| test | 1,258 | 9 | 31 |

Test patients are never seen during training. Results will be *lower* than in papers
that used the frame-level split. This is expected and is the scientifically correct result.

**2. Boxes, not masks.** Dice and IoU are therefore computed on boxes (matched
predicted vs ground-truth box). Pixel-level plaque area comes from the measurement
module, not from a segmentation mask.

**3. No normal frames.** Image-level specificity cannot be estimated on this dataset.
When the MTI HMC Peshawar data is added, include stenosis-free frames.

**4. Size shift.** The test patients have more small lesions (80% small boxes vs 52% in
train). Report recall by size: `evaluate.py` gives `recall_small` and `recall_medium`.

## 2. Install (GPU machine)

```bash
# 1) PyTorch with CUDA matching your driver (see pytorch.org), e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 2) the rest
pip install -r requirements.txt
```

Put the Roboflow export (folders `train/ valid/ test/`) in `./data`.

## 3. Run

The whole pipeline is in `bash run_all.sh`. Step by step:

```bash
python src/prepare_data.py                                   # patient-level split + CLAHE + YOLO labels + folds + EDA
python src/phantom_validation.py --n 300                      # Objective 2 technical validation (CPU)

python src/train.py --model rtdetr-l.pt --name rtdetr-l       # PROPOSED (transformer)
python src/train.py --model rtdetr-x.pt --name rtdetr-x       # optional larger transformer
python src/train.py --model yolo11m.pt  --name yolo11m        # CNN baseline
python src/train.py --model rtdetr-l.pt --name rtdetr-l --cv  # 5-fold patient-grouped CV

python src/evaluate.py --weights runs/rtdetr-l/weights/best.pt --name rtdetr-l
python src/infer_realtime.py --weights runs/rtdetr-l/weights/best.pt --source "D:/PhD Project/Dataset/CADICA/CADICA/selectedVideos/p20/v3/input" --fps 10
python src/expert_review.py make --pred results/rtdetr-l/predictions_test.csv --n 200
# ... expert fills results/review_pack/expert_review.xlsx ...
python src/expert_review.py analyze --pack results/review_pack
```

Settings live in `configs/default.yaml`. On a 24 GB GPU, batch 16 at 640 px works for
RT-DETR-l. Lower `batch` if you run out of memory. Training time is roughly 2–5 h for
150 epochs, depending on the GPU.

**Real-time inputs.** `--source` accepts a video (`.mp4/.avi`), a multi-frame DICOM (`.dcm`,
from which pixel spacing and frame rate are read automatically), a folder of frames, `seq:<id>`
for a dataset sequence, or `0` for a live frame-grabber. Add `--show` for a live window
and `--half` for FP16 on the GPU.

## 4. Outputs (for the thesis)

| Folder / file | Content |
|---|---|
| `results/eda/` | split summary, box statistics, size classes, `eda.png`, `split_manifest.csv` (exact patient split; put it in the appendix) |
| `results/phantom/` | measurement validation: `phantom_summary.csv`, Bland-Altman `phantom_agreement.png` |
| `runs/<model>/` | training curves, PR/F1 curves, confusion matrix (Ultralytics) |
| `results/<model>/summary_test.json` | mAP50, mAP50-95, P, R, F1, FDR, mean IoU, mean Dice, recall by size, latency, FPS, real-time yes/no |
| `results/<model>/predictions_test.csv` | per-image predictions and all measurements |
| `results/<model>/examples_test.jpg` | overlay examples (red = AI, green = ground truth) |
| `runs/<model>_cv_summary.csv` | 5-fold mean ± SD |
| `results/realtime/*.mp4, *_frames.csv` | real-time demo videos and per-frame latency |
| `results/review_pack/analysis/` | expert agreement report (kappa, weighted kappa, ICC, Bland-Altman, t-test, Wilcoxon, MAE, RMSE) |

## 5. Measurement module (Objective 2)

Angiography shows the contrast-filled **lumen**, not the plaque itself. Plaque is
quantified the way QCA does it, as the gap between the reconstructed healthy vessel
and the measured lumen:

1. Black top-hat vesselness, then a **minimum-cost centreline** through the lesion. The path follows the vessel axis estimated from the Hessian, so it bridges the faint lumen at tight stenoses.
2. Intensity profiles perpendicular to the centreline give the **edge diameter** D(s), taken from max-gradient edges smoothed along the vessel.
3. **Videodensitometry**: by Beer-Lambert, the integral of −log(I/I_bg) across a profile is proportional to the cross-sectional lumen area A(s). This holds for non-circular lumens and stays reliable where edges blur.
4. The **interpolated reference** is a robust line through the proximal and distal healthy segments.
5. Outputs:
   - %DS. The default is *hybrid*, the mean of the edge and densitometric ratios, which is robust to vessel overlap.
   - %AS, MLD, reference diameter, and lesion length (where D < 0.8 × reference).
   - **Projected plaque area** = ∫(D_ref − D) ds.
   - **Estimated plaque volume** = ∫ π/4 · D_ref² · (1 − A/A_ref) ds.
   - Severity class, and a quality score (0–1).

Units are pixels by default. Set `measure.mm_per_pixel` (catheter calibration, e.g. a 6F
catheter is 2.0 mm), or use DICOM input where PixelSpacing is read automatically, to get
mm, mm² and mm³.

**Phantom validation** (n = 300 synthetic curved vessels, X-ray-like attenuation, noise
and blur; true %DS 20–85%):

| Metric | MAE | Bias | ICC(2,1) | r |
|---|---|---|---|---|
| %DS hybrid (primary) | 2.97 | −0.36 | 0.972 | 0.978 |
| %DS densitometric | 2.17 | +0.97 | 0.990 | 0.992 |
| %DS edge | 4.77 | −1.27 | 0.924 | 0.936 |
| MLD (px) | 0.63 | −0.48 | 0.958 | 0.987 |
| Reference diameter (px) | 1.07 | −1.07 | 0.943 | 0.999 |
| Lesion length (px) | 1.82 | +1.77 | 0.946 | 0.987 |

All 300 phantoms were measured successfully. The edge method under-estimates tight
(>70%) stenoses because of the resolution limit; densitometry corrects most of this.
On real test angiograms with ground-truth boxes, 96% of lesions were measured, with a
median of 37 ms per lesion on a CPU (`results/measurement_examples_gtboxes.png`).
Phantoms are idealised. Clinical accuracy must come from the expert comparison (§6).

## 6. Expert validation (Dr. Samiullah)

No expert measurement file exists, so validation is prospective, in **two stages** that
avoid anchoring bias:

1. **Stage 1, blind.** Raw images only. The expert records whether a stenosis is present, a visual %DS, the severity and the vessel.
2. **Stage 2, AI review.** Raw image next to the AI overlay. The expert records whether the detection is correct, partially correct, a false positive or a miss, whether the location is right, and whether the measurement is acceptable.

`expert_review.py make` creates the images, an Excel file with drop-downs and an
instruction sheet. AI values are kept in a separate key file that the expert does not
see. `analyze` computes:

- Cohen's kappa for detection, and quadratic-weighted kappa for severity class.
- ICC(2,1) with 95% CI for AI %DS vs visual %DS.
- Bland-Altman, paired t-test, Wilcoxon, MAE and RMSE.

Visual estimation is known to over-estimate stenosis compared with QCA, so a negative
AI − expert bias is expected and should be discussed in the thesis.

## 7. Differences from the proposal text (justify these in the viva)

| Proposal says | Implemented | Reason |
|---|---|---|
| Resize to 256×256 | 640 (native 576) | Median lesion box is about 30 px at 576 and would shrink to about 13 px at 256, which loses small stenoses |
| Pure ViT and categorical cross-entropy | RT-DETR (CNN + transformer), with GIoU + L1 + varifocal classification loss | The data has boxes, so this is a detection task. RT-DETR is a transformer detector designed for real time. Plain ViT segmentation would need masks |
| Dice/IoU segmentation | Box-level IoU/Dice, plus mAP | No masks in the data |
| Plaque type (calcified/soft/mixed) | Not attempted | Plaque composition cannot be seen on 2-D X-ray angiography; it needs IVUS/OCT/CCTA. It is also not among the objectives |
| Plaque volume | Estimated volume, assuming a circular cross-section, with densitometric correction | True volume needs 3-D imaging. Call it "estimated" |
| 5000 MTI HMC images | Public Danilov dataset (8,325 images, 64 patients) | MTI data can be added later: DICOM input is supported, and `prepare_data.py` can be adapted to a combined split, or the MTI data used as **external validation** (proposal §Validation strategy) |
| Timeline: 1 year vs 1.5 years | — | The proposal is inconsistent: phases add up to 11 months |

## 8. References

- Danilov VV, Klyshnikov KY, Gerget OM, et al. Real-time coronary artery stenosis detection based on modern neural networks. *Sci Rep.* 2021;11:7582.
- Ovcharenko E, et al. Angiographic dataset for stenosis detection. Mendeley Data, V2, 2021. doi:10.17632/ydrm75xywg.2
- Zhao Y, Lv W, Xu S, et al. DETRs Beat YOLOs on Real-time Object Detection. *CVPR* 2024.
- Jocher G, Qiu J. Ultralytics YOLO11 / RT-DETR implementation. github.com/ultralytics/ultralytics (AGPL-3.0).
- Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability. *Psychol Bull.* 1979;86:420–428.
- McGraw KO, Wong SP. Forming inferences about some intraclass correlation coefficients. *Psychol Methods.* 1996;1:30–46.
- Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. *Lancet.* 1986;1:307–310.

## 9. One-month publication experiments

| Experiment | Command | Output |
|---|---|---|
| Proposed model, correct split | `python src/train.py --model rtdetr-l.pt --fast` then `python src/evaluate.py --weights runs/rtdetr-l-fast/weights/best.pt --name rtdetr-l-fast` | `results/rtdetr-l-fast/` |
| Data-leakage experiment (leaky frame split, same settings) | `python src/prepare_data.py --split-mode frame --out dataset_frame`, then `python src/train.py --model rtdetr-l.pt --fast --data-dir dataset_frame --name rtdetr-l-fast-frame`, then `python src/evaluate.py --weights runs/rtdetr-l-fast-frame/weights/best.pt --data-dir dataset_frame --name rtdetr-l-fast-frame` | `results/rtdetr-l-fast-frame/` |
| CNN baseline | `python src/train.py --model yolo11s.pt --fast` then evaluate | `results/yolo11s-fast/` |
| Measurement vs cardiologist grades (no GPU needed) | `python src/validate_measurement_cadica.py` | `results/measurement_cadica/` |

The leakage effect = metrics of the frame-split model on its own (leaky) test set minus metrics of the patient-split model on the held-out patients.
