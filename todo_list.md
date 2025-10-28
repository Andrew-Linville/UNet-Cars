# Test Plan

## 🧭 Goal
Train multiple UNet-SNGP models to:
- Compare performance to the base UNet.
- Evaluate how SNGP hyperparameters affect training and inference.
- Compare against traditional uncertainty quantification (UQ) methods such as MC Dropout.

---

## ✅ Phases Overview
1. **Plumbing & Reproducibility**
2. **Metrics & Logging**
3. **Train / Eval Scripts**
4. **Baselines (MC Dropout)**
5. **Study Launcher (ARC)**
6. **Aggregation & Plots**

---

## 🧩 0. Prerequisites
- [ ] Freeze exact data splits (save indices + hash in `config.yaml`).
- [ ] Add `meta.json` writer (git SHA, SLURM job ID, torch/cuda versions, start/end time).
- [ ] Directory helper to create run folder and return canonical paths.

---

## 📊 1. Unified Metrics Module — `src/metrics/metrics.py`
**Purpose:** single import point used by train/test/infer.

### Segmentation
- [ ] `miou(pred, target, num_classes, ignore_index=None)`
- [ ] `dice(pred, target, num_classes, average='macro')`
- [ ] `pixel_acc(pred, target, ignore_index=None)`

### Calibration
- [ ] `ece(probs, labels, n_bins=15)`
- [ ] `mce(probs, labels, n_bins=15)`
- [ ] `brier_score(probs, labels)` — mean squared error on one-hot targets
- [ ] `nll(logits, labels)` — cross-entropy (negative log-likelihood)
- [ ] `reliability_bins(probs, labels, n_bins=15)` — returns bin stats for plotting

### OOD Detection
- [ ] `auroc(id_scores, ood_scores)`
- [ ] `aupr(id_scores, ood_scores, positive='ood')`
- [ ] `fpr_at_tpr(id_scores, ood_scores, tpr=0.95)`

---

## 🧾 2. Logging Utilities — `src/utils/loggers.py`
**Purpose:** consistent CSV/Parquet and TensorBoard logging.

### TrainingLogger
- `log_step(step, loss, lr, gpu_mem, step_time, throughput)`
- `log_epoch(epoch, train_loss, val_loss, val_miou, val_ece, …)`
- `flush_csv(file='training_metrics.csv')`

### TestLogger
- `log(summary_dict)` → writes `test_metrics.csv`

### TBLogger
- writes scalars/images → single event file per run.

---

## 📁 3. Run Scaffolding & Manifest — `src/utils/run_io.py`
- [ ] `create_run_dir(root, group, subcase) → Path`
- [ ] `write_config_yaml(run_dir, cfg_dict)`
- [ ] `write_meta_json(run_dir, meta_dict)`
- [ ] `append_manifest(manifest_path, row_dict)` → updates `runs/runs_manifest.csv`

---

## 🧠 4. Training / Evaluation / Inference

### `src/train.py`
- Load config, seed, data, model, optimizer.
- Train per epoch:
  - record loss, step time, throughput, GPU memory
  - compute val mIoU/Dice/ECE/Brier/NLL
  - save best checkpoint (by val mIoU)
  - log CSV + TensorBoard
- On finish: append manifest entry with summary metrics.

### `src/test.py`
- Load best checkpoint for run.
- Evaluate on:
  - IID test set → segmentation + calibration metrics.
  - OOD set → AUROC, AUPR, FPR@95 using uncertainty (variance or entropy).
- Write `test_metrics.csv`.
- Save a small IID/OOD image sample set.

### `src/infer_images.py`
- Given checkpoint + image folder, generate visualization panels:
  - original RGB  
  - mask  
  - uncertainty map  
- Save to `run_dir/test_images/IID` or `OOD`.

---

## 💧 5. MC Dropout Baseline

### `src/model_factory.py`
- `build_unet(cfg)`
- `build_unet_sngp(cfg)`
- `build_unet_mcdropout(cfg)` — inserts dropout; enable at test time.

### `src/test.py`
- If `mc_dropout_samples > 1`, perform T stochastic forward passes.
- Aggregate mean prediction and variance for uncertainty metrics.

---

## ⚙️ 6. Study Definitions & ARC Launcher

### `studies/definitions.py`
Define experiment cases:
```python
cases = {
  "UNet_Control": [{}],
  "UNet_SNGP/Base_hypers": [{"rff_dim":512, "proj_dim":128, "ridge":1e-6}],
  "UNet_SNGP/RFF_dim": [{"rff_dim":64}, {"rff_dim":128}, {"rff_dim":256}],
  "UNet_SNGP/Proj_dim": [{"proj_dim":64}, {"proj_dim":128}, {"proj_dim":256}],
  "UNet_SNGP/Ridge_penalty": [{"ridge":1e-8}, {"ridge":1e-6}, {"ridge":1e-4}],
  "UNet_MC_Dropout": [{"dropout_p":0.2}, {"dropout_p":0.4}]
}
