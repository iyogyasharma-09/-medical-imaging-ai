import json
d = json.load(open("results/advanced_analysis.json"))

print("=== THRESHOLD TUNING ===")
for m, r in d["threshold_tuning"].items():
    print(f"  {m[:22]:22s}  thr={r['best_threshold']:.2f}  acc={r['test_acc_at_best_thr']:.4f}  sens={r['test_sensitivity']:.3f}  spec={r['test_specificity']:.3f}")

print("\n=== ENSEMBLE ===")
for name, r in d["ensemble"].items():
    print(f"  {name:22s}  acc={r['accuracy']:.4f}  auc={r['auc']:.4f}  sens={r['sensitivity']:.3f}  spec={r['specificity']:.3f}")

print("\n=== CALIBRATION ===")
for m, r in d["calibration"].items():
    print(f"  {m[:22]:22s}  ECE={r['ece']:.4f}  Brier={r['brier_score']:.4f}")

print("\n=== 5-FOLD CV ===")
for m, r in d["cross_validation"].items():
    print(f"  {m[:22]:22s}  acc={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}  auc={r['auc_mean']:.4f}+/-{r['auc_std']:.4f}")
