# src/depth.py
# MiDaS-small ONNX wrapper for monocular depth estimation
# Runs at inference on CPU in ~80ms
# NOT used for anomaly scoring — provides 5 depth stats that feed SHAP

import os
import numpy as np
import onnxruntime as ort
from PIL import Image


DATA_DIR = os.environ.get("DATA_DIR", "data")
MIDAS_INPUT_SIZE = 256   # MiDaS-small expects 256x256


class DepthEstimator:
    """
    Wraps MiDaS-small ONNX model.
    Loaded once at startup, runs on every Inspector Mode submission.
    
    Why MiDaS-small not MiDaS-large:
    Small runs in ~80ms CPU. Large runs in ~800ms CPU.
    We need 5 statistical summaries, not a precise depth map.
    Small is the correct tradeoff.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.session = None

    def load(self):
        model_path = os.path.join(self.data_dir, "midas_small.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MiDaS ONNX model not found: {model_path}\n"
                f"Download from: https://github.com/isl-org/MiDaS/releases"
            )
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        print(f"MiDaS-small ONNX loaded")

    def _preprocess(self, pil_img: Image.Image) -> np.ndarray:
        """
        Resize to 256x256, normalise to ImageNet mean/std.
        Returns [1, 3, 256, 256] float32 array.
        """
        img = pil_img.resize((MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE),
                              Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        # HWC → CHW → NCHW
        img_np = img_np.transpose(2, 0, 1)[np.newaxis, :]
        return img_np

    def _postprocess(self, depth_raw: np.ndarray) -> np.ndarray:
        """
        Squeeze output, resize to 224x224, normalise to [0, 1].
        Returns [224, 224] float32 array.
        """
        depth = depth_raw.squeeze()

        # Resize to match image size used everywhere else
        from PIL import Image as PILImage
        depth_pil = PILImage.fromarray(depth).resize((224, 224),
                                                      PILImage.BILINEAR)
        depth = np.array(depth_pil, dtype=np.float32)

        # Normalise to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-8:
            depth = (depth - d_min) / (d_max - d_min)
        return depth

    def get_depth_stats(self, pil_img: Image.Image) -> dict:
        """
        Run MiDaS, return 5 depth statistics.
        These are the SHAP features for depth signal.
        
        If model fails for any reason: return zeros.
        Inference continues without depth — heatmap and score unaffected.
        """
        if self.session is None:
            return self._zero_stats()

        try:
            input_tensor = self._preprocess(pil_img)
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: input_tensor})[0]
            depth = self._postprocess(output)
            return self._compute_stats(depth)

        except Exception as e:
            print(f"MiDaS inference failed: {e} — returning zeros")
            return self._zero_stats()

    def _compute_stats(self, depth: np.ndarray) -> dict:
        """
        Compute 5 statistics from [224, 224] depth map.
        
        mean_depth:          average depth across image
        depth_variance:      how much depth varies — high = complex surface
        gradient_magnitude:  average depth edge strength
        spatial_entropy:     how uniformly depth is distributed
        depth_range:         max - min depth — measures 3D relief
        """
        gx = np.gradient(depth, axis=1)
        gy = np.gradient(depth, axis=0)
        grad_mag = float(np.sqrt(gx**2 + gy**2).mean())

        hist, _ = np.histogram(depth.flatten(), bins=50, density=True)
        hist = hist + 1e-10
        from scipy.stats import entropy as scipy_entropy
        sp_entropy = float(scipy_entropy(hist))

        return {
            "mean_depth":          float(depth.mean()),
            "depth_variance":      float(depth.var()),
            "gradient_magnitude":  grad_mag,
            "spatial_entropy":     sp_entropy,
            "depth_range":         float(depth.max() - depth.min())
        }

    def _zero_stats(self) -> dict:
        return {
            "mean_depth": 0.0,
            "depth_variance": 0.0,
            "gradient_magnitude": 0.0,
            "spatial_entropy": 0.0,
            "depth_range": 0.0
        }

    def get_depth_map(self, pil_img: Image.Image) -> np.ndarray:
        """
        Returns raw [224, 224] depth map for visualisation in Inspector.
        Returns zeros array if model fails.
        """
        if self.session is None:
            return np.zeros((224, 224), dtype=np.float32)
        try:
            input_tensor = self._preprocess(pil_img)
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: input_tensor})[0]
            return self._postprocess(output)
        except Exception:
            return np.zeros((224, 224), dtype=np.float32)


# Global instance
depth_estimator = DepthEstimator()