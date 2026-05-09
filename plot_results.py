"""
plot_results.py
---------------
Generates all training and evaluation plots using matplotlib.
Saves to results/plots/ directory.

Plots:
  1. Training curves (loss + accuracy) per model
  2. Val accuracy comparison across models
  3. Val AUC comparison across models  
  4. Evaluation metrics radar chart (val + test)
  5. Confusion matrices (test set) per model
  6. LGG vs HGG recall comparison (class imbalance view)
  7. Training loss convergence all models (overlay)
  8. AUC progression during training
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BASE_DIR  = Path(__file__).parent.resolve()
RESULTS   = BASE_DIR / "results"
PLOTS_DIR = RESULTS / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "resnet18_medicalnet":  "#4C72B0",
    "densenet121_3d":       "#DD8452",
    "swin_transformer_3d":  "#55A868",
}
LABELS = {
    "resnet18_medicalnet":  "ResNet18-MedNet",
    "densenet121_3d":       "DenseNet121-3D",
    "swin_transformer_3d":  "SwinTransformer-3D",
}
STYLE = {
    "figure.facecolor":  "#0F1117",
    "axes.facecolor":    "#1A1D27",
    "axes.edgecolor":    "#3A3D4D",
    "axes.labelcolor":   "#E0E0E0",
    "xtick.color":       "#B0B0B0",
    "ytick.color":       "#B0B0B0",
    "text.color":        "#E0E0E0",
    "grid.color":        "#2A2D3D",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#1A1D27",
    "legend.edgecolor":  "#3A3D4D",
}
plt.rcParams.update(STYLE)
plt.rcParams["font.family"] = "DejaVu Sans"


# ── Load data ─────────────────────────────────────────────────────────────────
def load_history(name):
    p = RESULTS / f"{name}_history.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    # Each entry: {epoch, train:{loss,accuracy,f1,auc}, val:{...}, lr}
    return data.get("history", [])

def load_eval():
    p = RESULTS / "evaluation_report.json"
    return json.loads(p.read_text()) if p.exists() else {}

MODELS   = list(COLORS.keys())
HISTORIES = {m: load_history(m) for m in MODELS}
EVAL      = load_eval()


# ── Helper ────────────────────────────────────────────────────────────────────
def savefig(name):
    path = PLOTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=plt.rcParams["figure.facecolor"])
    plt.close()
    print(f"  Saved: {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Training curves per model (loss + accuracy side by side)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_training_curves():
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Training Curves — All Models (GLI LGG/HGG)", fontsize=15, fontweight="bold", y=1.01)

    for col, mname in enumerate(MODELS):
        hist = HISTORIES[mname]
        if hist is None:
            continue
        epochs     = [h["epoch"]                for h in hist]
        tr_loss    = [h["train"]["loss"]         for h in hist]
        val_loss   = [h["val"]["loss"]           for h in hist]
        tr_acc     = [h["train"]["accuracy"]*100 for h in hist]
        val_acc    = [h["val"]["accuracy"]*100   for h in hist]
        color      = COLORS[mname]

        # Loss
        ax = axes[0, col]
        ax.plot(epochs, tr_loss,  color=color,       lw=2, label="Train Loss")
        ax.plot(epochs, val_loss, color=color, lw=2, ls="--", alpha=0.7, label="Val Loss")
        ax.set_title(LABELS[mname], color=color, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss" if col == 0 else "")
        ax.legend(fontsize=8); ax.grid(True)

        # Accuracy
        ax = axes[1, col]
        ax.plot(epochs, tr_acc,  color=color,       lw=2, label="Train Acc")
        ax.plot(epochs, val_acc, color=color, lw=2, ls="--", alpha=0.7, label="Val Acc")
        ax.axhline(max(val_acc), color=color, lw=1, ls=":", alpha=0.5,
                   label=f"Best Val {max(val_acc):.1f}%")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)" if col == 0 else "")
        ax.set_ylim(40, 105); ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout()
    savefig("01_training_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Val loss + AUC convergence overlay (all 3 on same axes)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_convergence_overlay():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Convergence Comparison — All Models", fontsize=14, fontweight="bold")

    for mname in MODELS:
        hist  = HISTORIES[mname]
        if hist is None: continue
        epochs   = [h["epoch"]                for h in hist]
        val_loss = [h["val"]["loss"]           for h in hist]
        val_auc  = [h["val"].get("auc", 0)    for h in hist]
        c = COLORS[mname]
        ax1.plot(epochs, val_loss, color=c, lw=2.5, marker="o", ms=5, label=LABELS[mname])
        ax2.plot(epochs, val_auc,  color=c, lw=2.5, marker="s", ms=5, label=LABELS[mname])

    ax1.set_title("Validation Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True)
    ax2.set_title("Validation AUC");  ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC")
    ax2.set_ylim(0.5, 1.0); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    savefig("02_convergence_overlay.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Grouped bar: Acc / AUC / F1 / Sensitivity / Specificity (Val + Test)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_metrics_bar():
    metrics   = ["accuracy", "auc", "f1_weighted", "sensitivity", "specificity"]
    mlabels   = ["Accuracy", "AUC", "F1", "Sensitivity\n(HGG Recall)", "Specificity\n(LGG Recall)"]
    n_metrics = len(metrics)
    n_models  = len(MODELS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Evaluation Metrics — Val & Test Sets", fontsize=14, fontweight="bold")

    for ax_idx, split in enumerate(["val", "test"]):
        ax  = axes[ax_idx]
        x   = np.arange(n_metrics)
        w   = 0.25
        for i, mname in enumerate(MODELS):
            if mname not in EVAL: continue
            vals = [EVAL[mname][split].get(m, 0) for m in metrics]
            bars = ax.bar(x + i*w, vals, w, label=LABELS[mname],
                          color=COLORS[mname], alpha=0.85, edgecolor="white", lw=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="#E0E0E0")

        ax.set_title(split.upper() + " SET", fontsize=12, fontweight="bold")
        ax.set_xticks(x + w); ax.set_xticklabels(mlabels, fontsize=9)
        ax.set_ylim(0, 1.12); ax.set_ylabel("Score"); ax.grid(True, axis="y")
        ax.legend(fontsize=9)

    plt.tight_layout()
    savefig("03_metrics_bar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Confusion matrices (test set)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Confusion Matrices — Test Set (GLI)", fontsize=14, fontweight="bold")

    for ax, mname in zip(axes, MODELS):
        if mname not in EVAL: continue
        cm = np.array(EVAL[mname]["test"]["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["LGG", "HGG"]); ax.set_yticklabels(["LGG", "HGG"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(LABELS[mname], color=COLORS[mname], fontsize=11, fontweight="bold")

        for i in range(2):
            for j in range(2):
                txt_color = "white" if cm_norm[i, j] > 0.5 else "#E0E0E0"
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                        ha="center", va="center", fontsize=11,
                        color=txt_color, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    savefig("04_confusion_matrices.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5 — LGG vs HGG recall (class imbalance analysis)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_class_recall():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Per-Class Recall — Class Imbalance Analysis", fontsize=14, fontweight="bold")

    for ax_idx, split in enumerate(["val", "test"]):
        ax = axes[ax_idx]
        mnames  = [LABELS[m] for m in MODELS if m in EVAL]
        lgg_rec = [EVAL[m][split]["per_class"]["LGG"]["recall"] for m in MODELS if m in EVAL]
        hgg_rec = [EVAL[m][split]["per_class"]["HGG"]["recall"] for m in MODELS if m in EVAL]
        lgg_n   = EVAL[MODELS[0]][split]["per_class"]["LGG"]["n"]
        hgg_n   = EVAL[MODELS[0]][split]["per_class"]["HGG"]["n"]

        x = np.arange(len(mnames))
        w = 0.35
        b1 = ax.bar(x - w/2, lgg_rec, w, label=f"LGG (n={lgg_n})", color="#E74C3C", alpha=0.85)
        b2 = ax.bar(x + w/2, hgg_rec, w, label=f"HGG (n={hgg_n})", color="#2ECC71", alpha=0.85)

        for bar, v in zip(list(b1)+list(b2), lgg_rec+hgg_rec):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, color="#E0E0E0")

        ax.set_title(f"{split.upper()} SET", fontsize=12, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(mnames, fontsize=9)
        ax.set_ylim(0, 1.2); ax.set_ylabel("Recall"); ax.grid(True, axis="y")
        ax.legend(fontsize=9)
        ax.axhline(1.0, color="white", lw=0.5, ls="--", alpha=0.3)

    plt.tight_layout()
    savefig("05_class_recall.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Radar chart: all metrics per model (test set)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_radar():
    categories  = ["Accuracy", "AUC", "F1", "Sensitivity", "Specificity"]
    metric_keys = ["accuracy", "auc", "f1_weighted", "sensitivity", "specificity"]
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(STYLE["figure.facecolor"])
    ax.set_facecolor(STYLE["axes.facecolor"])
    ax.set_title("Model Comparison — Test Set\n(Radar)", fontsize=13, fontweight="bold", pad=20)

    for mname in MODELS:
        if mname not in EVAL: continue
        vals = [EVAL[mname]["test"].get(k, 0) for k in metric_keys]
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[mname], lw=2.5, label=LABELS[mname])
        ax.fill(angles, vals, color=COLORS[mname], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, color="#E0E0E0")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"], fontsize=7, color="#808080")
    ax.grid(color="#2A2D3D", linestyle="--", alpha=0.6)
    ax.spines["polar"].set_color("#3A3D4D")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()
    savefig("06_radar_chart.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 7 — Training accuracy vs epoch (with best epoch marker)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_val_acc_overlay():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Validation Accuracy — All Models", fontsize=14, fontweight="bold")

    for mname in MODELS:
        hist = HISTORIES[mname]
        if hist is None: continue
        epochs  = [h["epoch"] for h in hist]
        val_acc = [h["val"]["accuracy"] * 100 for h in hist]
        best_ep = epochs[np.argmax(val_acc)]
        best_v  = max(val_acc)
        c = COLORS[mname]
        ax.plot(epochs, val_acc, color=c, lw=2.5, label=f"{LABELS[mname]} (best {best_v:.1f}%)")
        ax.scatter([best_ep], [best_v], color=c, s=120, zorder=5, marker="*")

    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy (%)")
    ax.set_ylim(40, 105); ax.grid(True); ax.legend(fontsize=10)
    plt.tight_layout()
    savefig("07_val_acc_overlay.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 8 — Summary dashboard (combined 4-panel)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_dashboard():
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("BraTS 2023 GLI — Full Training & Evaluation Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A: Val acc overlay
    ax_a = fig.add_subplot(gs[0, :2])
    for mname in MODELS:
        hist = HISTORIES[mname]
        if hist is None: continue
        epochs  = [h["epoch"] for h in hist]
        val_acc = [h["val"]["accuracy"] * 100 for h in hist]
        best_v  = max(val_acc)
        ax_a.plot(epochs, val_acc, color=COLORS[mname], lw=2.5,
                  label=f"{LABELS[mname]} ({best_v:.1f}%)")
        ax_a.scatter([epochs[np.argmax(val_acc)]], [best_v],
                     color=COLORS[mname], s=100, zorder=5, marker="*")
    ax_a.set_title("A  Validation Accuracy", fontweight="bold")
    ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Acc (%)"); ax_a.set_ylim(40, 105)
    ax_a.grid(True); ax_a.legend(fontsize=9)

    # Panel B: AUC bar (test)
    ax_b = fig.add_subplot(gs[0, 2])
    aucs  = [EVAL[m]["test"]["auc"] if m in EVAL else 0 for m in MODELS]
    bars  = ax_b.bar([LABELS[m] for m in MODELS], aucs,
                     color=[COLORS[m] for m in MODELS], alpha=0.85, edgecolor="white", lw=0.5)
    for bar, v in zip(bars, aucs):
        ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                  f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#E0E0E0")
    ax_b.set_title("B  Test AUC", fontweight="bold")
    ax_b.set_ylim(0, 1.05); ax_b.set_ylabel("AUC"); ax_b.grid(True, axis="y")
    ax_b.tick_params(axis="x", labelsize=8, rotation=15)

    # Panel C + D: confusion matrices (densenet only — best model)
    for i, mname in enumerate(MODELS):
        if mname not in EVAL: continue
        ax = fig.add_subplot(gs[1, i])
        cm = np.array(EVAL[mname]["test"]["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["LGG","HGG"]); ax.set_yticklabels(["LGG","HGG"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{'ABCDE'[i+2]}  {LABELS[mname]}", color=COLORS[mname], fontweight="bold")
        for ii in range(2):
            for jj in range(2):
                tc = "white" if cm_norm[ii,jj] > 0.5 else "#E0E0E0"
                ax.text(jj, ii, f"{cm[ii,jj]}\n({cm_norm[ii,jj]*100:.0f}%)",
                        ha="center", va="center", fontsize=10, color=tc, fontweight="bold")

    plt.tight_layout()
    savefig("08_dashboard.png")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nGenerating plots -> {PLOTS_DIR}\n")
    plot_training_curves()
    plot_convergence_overlay()
    plot_metrics_bar()
    plot_confusion_matrices()
    plot_class_recall()
    plot_radar()
    plot_val_acc_overlay()
    plot_dashboard()
    print(f"\nAll plots saved to: {PLOTS_DIR}")
