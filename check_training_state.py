"""Check current training state: registry, checkpoints, remaining models."""
import json
from pathlib import Path

reg_path = Path("results/model_registry.json")
reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}

print("=== Model Registry ===")
if reg:
    for name, info in reg.items():
        ckpt_exists = Path(info["checkpoint_path"]).exists()
        print(f"  {name}")
        print(f"    val_accuracy   : {info.get('val_accuracy', 'N/A'):.4f}")
        print(f"    uncertainty    : {info.get('mean_uncertainty', 'N/A'):.4f}")
        print(f"    checkpoint     : {'EXISTS' if ckpt_exists else 'MISSING'}")
        print(f"    epochs_trained : {len(info.get('history', [info])) if 'history' not in info else '(in registry)'}")
        print(f"    device         : {info.get('trained_on_device', 'N/A')}")
else:
    print("  (empty)")

print()
print("=== Saved Checkpoints ===")
models_dir = Path("models")
for pt in sorted(models_dir.glob("*.pt")):
    size_mb = pt.stat().st_size / 1024**2
    # Peek inside checkpoint
    try:
        import torch
        ckpt = torch.load(str(pt), map_location="cpu", weights_only=False)
        epochs = len(ckpt.get("history", []))
        best_acc = ckpt.get("best_val_accuracy", "?")
        model_name = ckpt.get("model_name", pt.stem)
        print(f"  {pt.name}  ({size_mb:.1f} MB)  epochs={epochs}  best_val_acc={best_acc:.4f}")
    except Exception as e:
        print(f"  {pt.name}  ({size_mb:.1f} MB)  [could not read: {e}]")

print()
from config import MODELS
trained = set(reg.keys())
all_models = list(MODELS.keys())
remaining = [m for m in all_models if m not in trained]
print("=== Status per model ===")
for m in all_models:
    status = "DONE" if m in trained else "PENDING"
    print(f"  [{status}] {m}")
print()
print(f"Remaining: {remaining}")
