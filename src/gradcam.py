"""
Volume-level Grad-CAM for 3D backbones.
Produces a central-slice heatmap overlay from a 3D activation map.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INPUT_HEIGHT, INPUT_WIDTH


class GradCAM3D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inp, out):
            self.activations = out.detach()

        def backward_hook(_module, _gin, gout):
            self.gradients = gout[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)
        target = output.argmax(dim=1)
        one_hot = torch.zeros_like(output)
        one_hot[0, target.item()] = 1.0
        self.model.zero_grad()
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="trilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()


def get_target_layer(model_name: str, model: torch.nn.Module):
    """
    Return the target conv/attention layer for Grad-CAM.

    ResNet50     → last residual block (layer4[-1])
    DenseNet121  → last dense block (features.denseblock4)
    SwinTransformer → last Swin stage norm (swin_encoder.swinViT.layers4[0].blocks[-1].norm1)
    """
    if model_name == "resnet18_medicalnet":
        return model.backbone.layer4[-1]
    if model_name == "densenet121_3d":
        return model.backbone.features.denseblock4
    if model_name == "swin_transformer_3d":
        # Last Swin Transformer stage — norm1 of the last block
        # provides the richest localized attention signal
        return model.swin_encoder.swinViT.layers4[0].blocks[-1].norm1
    raise ValueError(f"Unsupported model for 3D Grad-CAM: {model_name}")



def generate_gradcam(model_name: str, model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    gradcam = GradCAM3D(model, get_target_layer(model_name, model))
    heatmap_3d = gradcam(input_tensor)
    depth_scores = heatmap_3d.mean(axis=(1, 2))
    depth_idx = int(np.argmax(depth_scores))
    heatmap = heatmap_3d[depth_idx]
    heatmap = cv2.resize(heatmap, (INPUT_WIDTH, INPUT_HEIGHT))

    base = input_tensor[0, 0, depth_idx].detach().cpu().numpy()
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    base_rgb = np.stack([base, base, base], axis=-1)

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = 0.55 * colored + 0.45 * base_rgb
    return np.uint8(np.clip(overlay, 0, 1) * 255)
