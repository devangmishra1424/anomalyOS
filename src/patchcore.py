# src/patchcore.py
# PatchCore feature extraction and anomaly scoring
# WideResNet-50 frozen backbone, layer2 + layer3 hooks
# This is the core ML component — built from scratch, no Anomalib

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import joblib
import os
import scipy.ndimage


DATA_DIR = os.environ.get("DATA_DIR", "data")
DEVICE = "cpu"   # HF Spaces has no GPU — always CPU at inference
IMG_SIZE = 224


class PatchCoreExtractor:
    """
    WideResNet-50 feature extractor with forward hooks.
    
    Why two layers:
    - layer2 (28x28): captures fine-grained texture anomalies
    - layer3 (14x14): captures structural/shape anomalies
    Single layer misses one or the other. Multi-scale = better AUROC.
    
    Why frozen:
    We never update any weights. PatchCore does not train on defects.
    It memorises normal patches, then measures deviation at inference.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.model = None
        self.pca = None
        self._layer2_feat = {}
        self._layer3_feat = {}

        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def load(self):
        # ── Load WideResNet-50 ────────────────────────────────
        self.model = models.wide_resnet50_2(pretrained=False)

        weights_path = os.path.join(self.data_dir, "wide_resnet50_2.pth")
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path,
                                                   map_location="cpu"))
        else:
            # Download pretrained weights
            self.model = models.wide_resnet50_2(pretrained=True)

        self.model = self.model.to(DEVICE)
        self.model.eval()

        # Freeze all weights — never updated
        for param in self.model.parameters():
            param.requires_grad = False

        # Register hooks
        self.model.layer2.register_forward_hook(self._hook_layer2)
        self.model.layer3.register_forward_hook(self._hook_layer3)

        # ── Load PCA model ────────────────────────────────────
        pca_path = os.path.join(self.data_dir, "pca_256.pkl")
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA model not found: {pca_path}")
        self.pca = joblib.load(pca_path)
        print(f"PatchCore extractor loaded | "
              f"PCA: {self.pca.n_components_} components")

    def _hook_layer2(self, module, input, output):
        self._layer2_feat["feat"] = output

    def _hook_layer3(self, module, input, output):
        self._layer3_feat["feat"] = output

    @torch.no_grad()
    def extract_patches(self, pil_img: Image.Image) -> np.ndarray:
        """
        Extract 784 patch descriptors from one image.
        
        Pipeline:
        1. Forward pass through WideResNet (hooks capture layer2, layer3)
        2. Upsample layer3 to match layer2 spatial size (14→28)
        3. Concatenate: [1, C2+C3, 28, 28]
        4. 3x3 neighbourhood aggregation (makes each patch context-aware)
        5. Reshape to [784, C2+C3]
        6. PCA reduce to [784, 256]
        
        Returns: [784, 256] float32 numpy array
        """
        tensor = self.transform(pil_img).unsqueeze(0).to(DEVICE)
        _ = self.model(tensor)   # triggers hooks

        l2 = self._layer2_feat["feat"]   # [1, C2, 28, 28]
        l3 = self._layer3_feat["feat"]   # [1, C3, 14, 14]

        # Upsample layer3 to 28x28
        l3_up = nn.functional.interpolate(
            l3, size=(28, 28), mode="bilinear", align_corners=False
        )
        combined = torch.cat([l2, l3_up], dim=1)   # [1, C2+C3, 28, 28]

        # 3x3 neighbourhood aggregation
        combined = nn.functional.avg_pool2d(
            combined, kernel_size=3, stride=1, padding=1
        )

        # Reshape: [1, C, 28, 28] → [784, C]
        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(-1, C)
        patches_np = patches.cpu().numpy().astype(np.float32)

        # PCA reduce: [784, C] → [784, 256]
        patches_reduced = self.pca.transform(patches_np).astype(np.float32)

        return patches_reduced   # [784, 256]

    def build_anomaly_map(self,
                           patch_scores: np.ndarray,
                           smooth: bool = True) -> np.ndarray:
        """
        Convert [28, 28] patch distance grid to [224, 224] anomaly heatmap.
        
        Steps:
        1. Upsample 28x28 → 224x224 (bilinear)
        2. Gaussian smoothing (sigma=4) — removes patch-boundary artifacts
        3. Normalise to [0, 1]
        
        Returns: [224, 224] float32 heatmap
        """
        # Upsample via PIL for bilinear interpolation
        from PIL import Image as PILImage
        heatmap_pil = PILImage.fromarray(patch_scores.astype(np.float32))
        heatmap = np.array(
            heatmap_pil.resize((224, 224), PILImage.BILINEAR),
            dtype=np.float32
        )

        # Gaussian smoothing
        if smooth:
            heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=2)

        # Normalise to [0, 1]
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max - h_min > 1e-8:
            heatmap = (heatmap - h_min) / (h_max - h_min)

        return heatmap

    def get_anomaly_centroid(self, heatmap: np.ndarray) -> tuple:
        """
        Find the peak (highest activation) location of the anomaly.
        Used to locate defect crop for Index 2 retrieval.
        Returns: (cx, cy) pixel coordinates of maximum activation
        """
        if heatmap.size == 0:
            return (112, 112)   # centre fallback
        
        # Use peak location, not mean of thresholded region
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return (int(max_idx[1]), int(max_idx[0]))  # cx, cy

    def calibrate_score(self,
                         raw_score: float,
                         category: str,
                         thresholds: dict) -> float:
        """
        Calibrated score: sigmoid((score - mean) / std)
        Raw k-NN distance is NOT a probability.
        Calibrated score IS interpretable as anomaly confidence.
        
        Interview line: "My scores are calibrated against the distribution
        of normal patch distances in the training set, not raw distances."
        """
        if category not in thresholds:
            return float(1 / (1 + np.exp(-raw_score)))

        cal_mean = thresholds[category]["cal_mean"]
        cal_std  = thresholds[category]["cal_std"]
        z = (raw_score - cal_mean) / (cal_std + 1e-8)
        return float(1 / (1 + np.exp(-z)))


# Global instance
patchcore = PatchCoreExtractor()