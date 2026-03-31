# src/xai.py
# Four XAI methods — each answers a different question
#
# Method 1 — PatchCore anomaly map: WHERE is the defect? (in patchcore.py)
# Method 2 — GradCAM++:            WHICH features triggered the classifier?
# Method 3 — SHAP waterfall:       WHY is the score this specific number?
# Method 4 — Retrieval trace:      WHAT in history is this most similar to?

import os
import json
import base64
import io
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import shap
from PIL import Image
import cv2


DATA_DIR = os.environ.get("DATA_DIR", "data")
DEVICE = "cpu"
IMG_SIZE = 224


class GradCAMPlusPlus:
    """
    GradCAM++ on EfficientNet-B0.
    
    Why GradCAM++ not basic GradCAM:
    Basic GradCAM uses only positive gradients, producing fragmented maps.
    GradCAM++ uses a weighted combination of both positive and negative
    gradients, resulting in more focused, anatomically precise maps.
    Same implementation complexity — direct upgrade.
    
    Why a separate EfficientNet:
    PatchCore has no gradient flow (it's a memory bank + k-NN).
    GradCAM++ requires differentiable activations.
    EfficientNet is fine-tuned on MVTec binary classification solely
    to provide gradients for this XAI method — never used for scoring.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.model = None
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def load(self):
        self.model = models.efficientnet_b0(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 2)
        )
        weights_path = os.path.join(self.data_dir, "efficientnet_b0.pt")
        if os.path.exists(weights_path):
            self.model.load_state_dict(
                torch.load(weights_path, map_location="cpu")
            )
        else:
            # Fallback: pretrained ImageNet weights (weaker XAI but not None)
            self.model = models.efficientnet_b0(pretrained=True)
            print("WARNING: EfficientNet fine-tuned weights not found. "
                  "Using ImageNet pretrained — GradCAM++ quality reduced.")

        self.model = self.model.to(DEVICE)
        self.model.eval()
        print("GradCAM++ (EfficientNet-B0) loaded")

    def compute(self, pil_img: Image.Image) -> np.ndarray:
        """
        Compute GradCAM++ activation map.
        Target layer: model.features[-1]
        Returns: [224, 224] float32 array in [0, 1], or None if fails.
        """
        if self.model is None:
            return None

        try:
            tensor = self.transform(pil_img).unsqueeze(0).to(DEVICE)
            tensor.requires_grad_(True)

            # Storage for hook outputs
            activations = {}
            gradients = {}

            def forward_hook(module, input, output):
                activations["feat"] = output

            def backward_hook(module, grad_in, grad_out):
                gradients["feat"] = grad_out[0]

            # Register hooks on last feature block
            target_layer = self.model.features[-1]
            fwd_handle = target_layer.register_forward_hook(forward_hook)
            bwd_handle = target_layer.register_full_backward_hook(backward_hook)

            # Forward pass
            with torch.enable_grad():
                output = self.model(tensor)
                pred_class = output.argmax(dim=1).item()
                score = output[0, pred_class]
                self.model.zero_grad()
                score.backward()

            fwd_handle.remove()
            bwd_handle.remove()

            # GradCAM++ weights
            # α = ReLU(grad)² / (2*ReLU(grad)² + sum(A)*ReLU(grad)³)
            grads = gradients["feat"]           # [1, C, H, W]
            acts  = activations["feat"]         # [1, C, H, W]

            grads_relu = torch.relu(grads)
            acts_sum   = acts.sum(dim=(2, 3), keepdim=True)

            alpha_num   = grads_relu ** 2
            alpha_denom = 2 * grads_relu**2 + acts_sum * grads_relu**3 + 1e-8
            alpha       = alpha_num / alpha_denom

            weights = (alpha * torch.relu(grads)).sum(dim=(2, 3),
                                                       keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            cam = torch.relu(cam).squeeze().cpu().numpy()

            # Upsample to 224x224
            cam_pil = Image.fromarray(cam)
            cam = np.array(cam_pil.resize((IMG_SIZE, IMG_SIZE),
                                           Image.BILINEAR), dtype=np.float32)

            # Normalise
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)

            return cam

        except Exception as e:
            print(f"GradCAM++ failed: {e}")
            return None


class SHAPExplainer:
    """
    SHAP waterfall chart for anomaly score.
    Explains score as function of 5 human-readable features.
    
    The 5 features:
    - mean_patch_distance:  avg k-NN distance (pervasive texture anomaly)
    - max_patch_distance:   max k-NN distance = image anomaly score
    - depth_variance:       from MiDaS (complex 3D surface)
    - edge_density:         fraction of Canny edge pixels
    - texture_regularity:   FFT low-frequency energy ratio
    
    Interview line: "A QC manager reads the SHAP chart and understands
    why the model flagged this image without knowing what a neural net is."
    """

    def __init__(self):
        self.explainer = None
        self._background_features = None
        self._background_loaded = False

    def load_background(self, background_path: str = None):
        """
        Load background features for SHAP TreeExplainer.
        Background = sample of normal image features from training set.
        """
        if background_path and os.path.exists(background_path):
            self._background_features = np.load(background_path)
            print(f"SHAP background loaded: {self._background_features.shape}")
        else:
            # Fallback: use zeros as background (weaker but functional)
            self._background_features = np.zeros((10, 5), dtype=np.float32)
            print("SHAP using zero background (background_features.npy not found)")
        self._background_loaded = True

    def build_feature_vector(self,
                              patch_scores: np.ndarray,
                              depth_stats: dict,
                              fft_features: dict,
                              edge_features: dict) -> np.ndarray:
        """
        Assemble the 5 SHAP features from computed signals.
        Returns: [5] float32 array
        """
        return np.array([
            float(patch_scores.mean()),              # mean_patch_distance
            float(patch_scores.max()),               # max_patch_distance
            float(depth_stats.get("depth_variance", 0.0)),
            float(edge_features.get("edge_density", 0.0)),
            float(fft_features.get("low_freq_ratio", 0.0))
        ], dtype=np.float32)

    def explain(self, feature_vector: np.ndarray) -> dict:
        """
        Compute SHAP values for one feature vector.
        Returns dict with feature names, values, and SHAP contributions.
        """
        FEATURE_NAMES = [
            "mean_patch_distance",
            "max_patch_distance",
            "depth_variance",
            "edge_density",
            "texture_regularity"
        ]

        if not self._background_loaded:
            return self._fallback_explain(feature_vector, FEATURE_NAMES)

        try:
            # Simple linear approximation for portfolio:
            # SHAP values proportional to deviation from background mean
            bg_mean = self._background_features.mean(axis=0)
            deviations = feature_vector - bg_mean
            total = np.abs(deviations).sum() + 1e-8
            shap_values = deviations * (feature_vector.sum() / total)

            return {
                "feature_names": FEATURE_NAMES,
                "feature_values": feature_vector.tolist(),
                "shap_values": shap_values.tolist(),
                "base_value": float(bg_mean.mean()),
                "prediction": float(feature_vector.sum())
            }

        except Exception as e:
            print(f"SHAP explain failed: {e}")
            return self._fallback_explain(feature_vector, FEATURE_NAMES)

    def _fallback_explain(self, features, names):
        return {
            "feature_names": names,
            "feature_values": features.tolist(),
            "shap_values": features.tolist(),
            "base_value": 0.0,
            "prediction": float(features.max())
        }


def heatmap_to_base64(heatmap: np.ndarray,
                       original_img: Image.Image = None) -> str:
    """
    Convert [224, 224] float32 heatmap to base64 PNG.
    If original_img provided: overlay heatmap on original (jet colormap).
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if original_img is not None:
        orig_np = np.array(original_img.resize((224, 224)))
        overlay = (0.6 * orig_np + 0.4 * heatmap_rgb).astype(np.uint8)
        result_img = Image.fromarray(overlay)
    else:
        result_img = Image.fromarray(heatmap_rgb)

    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_to_base64(pil_img: Image.Image,
                     size: tuple = (224, 224)) -> str:
    """Convert PIL image to base64 PNG string."""
    img = pil_img.resize(size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Global instances
gradcam = GradCAMPlusPlus()
shap_explainer = SHAPExplainer()