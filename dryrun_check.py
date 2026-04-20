"""Quick 1-batch dry-run to validate the full GPU training loop before full training."""
import sys, torch
sys.path.insert(0, ".")
from train_brats2023 import build_dataloaders, get_model, get_training_device
import torch.nn as nn

device = get_training_device()

train_loader, val_loader, _, class_weights = build_dataloaders(
    task="GLI", label_col="label_grade", batch_size=1
)
print(f"Train batches : {len(train_loader)}")
print(f"Val batches   : {len(val_loader)}")
print(f"Class weights : {class_weights}")

print("\nLoading 1 real NIfTI batch from disk...")
batch = next(iter(train_loader))
img_shape = tuple(batch["image"].shape)
label_val = batch["label"]
print(f"  image shape : {img_shape}")
print(f"  label       : {label_val}")

print("\nRunning forward + backward pass on GPU (fp16 AMP)...")
model     = get_model("resnet18_medicalnet", num_classes=2).to(device)
cw        = torch.tensor(class_weights, dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler    = torch.amp.GradScaler(device="cuda")

imgs   = batch["image"].to(device)
labels = batch["label"].to(device)
with torch.amp.autocast(device_type="cuda", enabled=True):
    logits = model(imgs)
    loss   = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print(f"  logits : {logits.detach().cpu().tolist()}")
print(f"  loss   : {loss.item():.4f}")

vram = torch.cuda.memory_allocated() / 1024**3
vram_peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"  VRAM   : {vram:.2f} GB current, {vram_peak:.2f} GB peak")
print()
print("[PASS] Dry-run complete. Safe to start full training.")
