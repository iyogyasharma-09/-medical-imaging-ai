# Evaluation Pipeline Comparison

## Audit Date: 2026-08-17
## Auditor: Protocol Consistency Audit (Read-Only Forensic)

---

## 1. Scripts Compared

| Script | Purpose | Checkpoint Source | Status |
|---|---|---|---|
| `evaluate_models.py` | Earlier evaluation | `models/` directory | Used whatever checkpoint was in models/ at runtime |
| `evaluate_locked_test.py` | Locked test evaluation | `protocol_archives/protocol_A/B/C/` | Uses canonical archived checkpoints |
| `run_day5_phase1_locked_test.py` | Final locked test + integrity | `locked_test_evaluation/predictions/` (reuse) | Reprocesses predictions from evaluate_locked_test.py |
| `train_brats2023.py` `run_epoch()` | Training-time val metrics | N/A (inline) | Uses `recall_score(average=None)` for LGG recall |

---

## 2. Checkpoint Loading

### evaluate_models.py (Earlier)
```python
CHECKPOINTS = {
    "resnet18_medicalnet": MODELS_DIR / "resnet18_medicalnet_lgg_hgg_3d.pt",
    "densenet121_3d":      MODELS_DIR / "densenet121_3d_lgg_hgg_3d.pt",
    "swin_transformer_3d": MODELS_DIR / "swin_transformer_3d_lgg_hgg_3d.pt",
}
```
**Critical**: This loads from `models/` directory, which is the ACTIVE training output path. The checkpoint in this directory changes every time training completes. It does NOT distinguish between protocols.

### evaluate_locked_test.py (Locked Test)
```python
ARCHIVE_DIR = RESULTS_DIR / "protocol_archives"
# Loads from: protocol_archives/protocol_A/resnet18_medicalnet_lgg_hgg_3d.pt
ckpt_path = proto_dir / ckpt_filename
```
**Correct**: This loads from the archived protocol-specific directories. Each protocol has its own frozen checkpoint.

---

## 3. Threshold Comparison

| Script | Threshold Method | Value |
|---|---|---|
| `evaluate_models.py` | `y_pred = y_prob.argmax(axis=1)` | Equivalent to 0.5 (argmax of softmax) |
| `evaluate_locked_test.py` | `y_pred = (y_prob_1 >= 0.5).astype(int)` | Fixed 0.5 |
| `run_day5_phase1_locked_test.py` | Reuses `predicted_label` from CSV | Fixed 0.5 (inherited) |
| `train_brats2023.py` | `pred = prob.argmax(axis=1)` | Equivalent to 0.5 |

**Verdict**: All pipelines use an effective threshold of 0.5. No threshold discrepancy.

---

## 4. LGG Recall Calculation

### evaluate_locked_test.py
```python
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
lgg_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
```

### evaluate_models.py
```python
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
lgg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
```

### train_brats2023.py
```python
recalls = recall_score(true, pred, average=None, zero_division=0)
lgg_rec = float(recalls[0]) if len(recalls) > 0 else 0.0
```

**Note on confusion matrix convention**: Both evaluation scripts use `confusion_matrix(y_true, y_pred, labels=[0, 1])` or equivalent. With `labels=[0,1]`:
- Row 0 = true class 0 (LGG), Col 0 = predicted 0 (LGG), Col 1 = predicted 1 (HGG)
- `cm[0,0]` = TN = correctly predicted LGG
- `cm[0,1]` = FP = LGG misclassified as HGG

The `tn/(tn+fp)` formula = `cm[0,0] / (cm[0,0] + cm[0,1])` = class 0 recall = LGG recall. **This is mathematically equivalent to `recall_score(average=None)[0]`.**

**Verified**: The LGG Recall calculation is consistent across all pipelines.

---

## 5. Preprocessing Pipeline

| Component | evaluate_models.py | evaluate_locked_test.py | train_brats2023.py (val) |
|---|---|---|---|
| Transform | `get_val_transforms()` | `get_val_transforms()` | `get_val_transforms()` |
| Modalities | [t1n, t1c, t2w, t2f] | [t1n, t1c, t2w, t2f] | [t1n, t1c, t2w, t2f] |
| Spatial Size | 96×96×96 | 96×96×96 | 96×96×96 |
| Normalization | z-score, channel-wise, nonzero | z-score, channel-wise, nonzero | z-score, channel-wise, nonzero |
| Orientation | RAS | RAS | RAS |
| Spacing | 1mm isotropic | 1mm isotropic | 1mm isotropic |
| Crop | Foreground (nonzero) | Foreground (nonzero) | Foreground (nonzero) |
| Batch Size | 4 | 4 | 2 (train) / 2 (val) |
| AMP | Yes | Yes | Yes |
| shuffle | False | False | False (val/test) |

**Verdict**: Preprocessing is identical across all evaluation pipelines. No discrepancy.

---

## 6. Label Mapping

| Source | LGG | HGG |
|---|---|---|
| `config.py` | 0 | 1 |
| `TARGET_TO_CLASS` | `{0: "lgg", 1: "hgg"}` | |
| Test manifest `label_grade` | 0 | 1 |
| Prediction CSVs `true_label` | 0 | 1 |
| Confusion matrix `labels=[0,1]` | Row 0 = LGG | Row 1 = HGG |

**Verdict**: Label mapping is consistent everywhere. No inversion detected.

---

## 7. Model Construction

Both `evaluate_models.py` and `evaluate_locked_test.py` call:
```python
model = get_model(model_key, num_classes=2)
model.load_state_dict(ckpt["model_state_dict"])
```

Same `get_model()` function from `train_brats2023.py`. No discrepancy in model architecture construction.

---

## 8. Critical Difference Summary

The ONLY meaningful difference between the two pipelines is:

| Factor | evaluate_models.py | evaluate_locked_test.py |
|---|---|---|
| **Checkpoint source** | `models/` directory (mutable) | `protocol_archives/protocol_X/` (frozen) |

This is the root cause of the discrepancy.
