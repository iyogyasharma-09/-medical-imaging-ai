"""VRAM test specifically for SwinUNETR 3D at different batch sizes."""
import torch, sys
sys.path.insert(0, ".")
from config import INPUT_CHANNELS, SPATIAL_SIZE
from train_brats2023 import get_model

props = torch.cuda.get_device_properties(0)
total_gb = props.total_memory / 1024**3
print(f"GPU: {props.name}  Total VRAM: {total_gb:.1f} GB\n")

for bs in [1, 2, 4, 8]:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = get_model("swin_transformer_3d", num_classes=2).cuda()
        dummy = torch.zeros(bs, INPUT_CHANNELS, *SPATIAL_SIZE, device="cuda")
        with torch.amp.autocast(device_type="cuda", enabled=True):
            out = m(dummy)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        pct  = peak / total_gb * 100
        status = "OK" if pct < 90 else "TIGHT"
        t_start = torch.cuda.Event(enable_timing=True)
        t_end   = torch.cuda.Event(enable_timing=True)
        t_start.record()
        with torch.amp.autocast(device_type="cuda", enabled=True):
            _ = m(dummy)
        t_end.record()
        torch.cuda.synchronize()
        ms = t_start.elapsed_time(t_end)
        print(f"  batch={bs}: peak={peak:.2f}GB ({pct:.0f}%)  fwd={ms:.0f}ms  [{status}]")
        del m, dummy, out
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  batch={bs}: OOM — {str(e)[:60]}")
        torch.cuda.empty_cache()
