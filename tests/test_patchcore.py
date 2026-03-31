# tests/test_patchcore.py
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch
import torch


def make_dummy_image(size=(224, 224)):
    return Image.fromarray(
        np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    )


class TestPatchCoreExtractor:

    def test_extract_patches_shape(self):
        """Patch extraction returns [784, 256] after PCA."""
        from src.patchcore import PatchCoreExtractor
        extractor = PatchCoreExtractor()
        extractor.load()

        img     = make_dummy_image()
        patches = extractor.extract_patches(img)

        assert patches.shape == (784, 256), (
            f"Expected (784, 256), got {patches.shape}"
        )
        assert patches.dtype == np.float32

    def test_extract_patches_all_normal_image(self):
        """All-white image should produce valid patches (not NaN/Inf)."""
        from src.patchcore import PatchCoreExtractor
        extractor = PatchCoreExtractor()
        extractor.load()

        white_img = Image.fromarray(
            np.full((224, 224, 3), 255, dtype=np.uint8)
        )
        patches = extractor.extract_patches(white_img)

        assert not np.isnan(patches).any(), "NaN in patch features"
        assert not np.isinf(patches).any(), "Inf in patch features"

    def test_build_anomaly_map_shape(self):
        """Anomaly map upsamples from 28x28 to 224x224."""
        from src.patchcore import PatchCoreExtractor
        extractor = PatchCoreExtractor()
        extractor.load()

        patch_scores = np.random.rand(28, 28).astype(np.float32)
        heatmap      = extractor.build_anomaly_map(patch_scores)

        assert heatmap.shape == (224, 224)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_anomaly_map_values_in_range(self):
        """Normalised heatmap must be in [0, 1]."""
        from src.patchcore import PatchCoreExtractor
        extractor = PatchCoreExtractor()
        extractor.load()

        # Extreme values
        patch_scores = np.ones((28, 28), dtype=np.float32) * 999.0
        heatmap      = extractor.build_anomaly_map(patch_scores)

        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-6

    def test_get_anomaly_centroid_returns_valid_coords(self):
        """Centroid must be within image bounds."""
        from src.patchcore import PatchCoreExtractor
        extractor = PatchCoreExtractor()
        extractor.load()

        # Single hot spot at top-left
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[10:30, 10:30] = 1.0
        cx, cy = extractor.get_anomaly_centroid(heatmap)

        assert 0 <= cx < 224
        assert 0 <= cy < 224

    def test_calibrate_score_output_range(self):
        """Calibrated score must be in [0, 1]."""
        from src.patchcore import PatchCoreExtractor
        extractor = PatchCoreExtractor()
        extractor.load()

        thresholds = {
            "bottle": {"cal_mean": 0.5, "cal_std": 0.1, "threshold": 0.8}
        }
        for raw in [0.0, 0.5, 1.0, 5.0, -1.0]:
            cal = extractor.calibrate_score(raw, "bottle", thresholds)
            assert 0.0 <= cal <= 1.0, f"Calibrated score {cal} out of range"