"""Hardware detection script — checks GPU, NPU, DirectML, OpenVINO."""
import torch

print("=== GPU / CUDA ===")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  ({p.total_memory/1024**3:.1f} GB VRAM, cc={p.major}.{p.minor})")

print("\n=== DirectML (NPU/GPU via Windows ML) ===")
try:
    import torch_directml
    count = torch_directml.device_count()
    print(f"DirectML available : True  ({count} device(s))")
    for i in range(count):
        print(f"  Device {i}: {torch_directml.device_name(i)}")
except ImportError:
    print("torch-directml     : NOT installed")

print("\n=== Intel OpenVINO ===")
try:
    from openvino.runtime import Core
    ie = Core()
    devices = ie.available_devices
    print(f"OpenVINO devices: {devices}")
    for d in devices:
        try:
            name = ie.get_property(d, "FULL_DEVICE_NAME")
            print(f"  {d}: {name}")
        except Exception:
            print(f"  {d}: (name unavailable)")
except ImportError:
    print("openvino           : NOT installed")
except Exception as e:
    print(f"OpenVINO error     : {e}")

print("\n=== ONNX Runtime ===")
try:
    import onnxruntime as ort
    print(f"Available EPs: {ort.get_available_providers()}")
except ImportError:
    print("onnxruntime        : NOT installed")

print("\n=== CPU info ===")
import platform, os
print(f"  OS      : {platform.system()} {platform.release()}")
print(f"  CPU     : {platform.processor()}")
print(f"  Cores   : {os.cpu_count()}")
