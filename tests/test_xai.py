# tests/test_xai.py
import numpy as np
import pytest
from PIL import Image


def make_dummy_image(size=(224, 224)):
    return Image.fromarray(
        np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    )


class TestGradCAMPlusPlus:

    def test_gradcam_returns_correct_shape(self):
        """GradCAM++ must return [224, 224] array."""
        from src.xai import GradCAMPlusPlus
        g = GradCAMPlusPlus()
        g.load()

        img = make_dummy_image()
        cam = g.compute(img)

        assert cam is not None
        assert cam.shape == (224, 224)

    def test_gradcam_values_in_range(self):
        """GradCAM++ output must be normalised to [0, 1]."""
        from src.xai import GradCAMPlusPlus
        g = GradCAMPlusPlus()
        g.load()

        img = make_dummy_image()
        cam = g.compute(img)

        if cam is not None:
            assert cam.min() >= 0.0 - 1e-6
            assert cam.max() <= 1.0 + 1e-6

    def test_gradcam_returns_none_gracefully_when_model_missing(self):
        """GradCAM++ must return None (not raise) if model not loaded."""
        from src.xai import GradCAMPlusPlus
        g = GradCAMPlusPlus()
        # Do NOT call g.load()

        img    = make_dummy_image()
        result = g.compute(img)

        assert result is None


class TestSHAPExplainer:

    def test_build_feature_vector_shape(self):
        """Feature vector must have exactly 5 elements."""
        from src.xai import SHAPExplainer
        s = SHAPExplainer()

        patch_scores = np.random.rand(28, 28).astype(np.float32)
        depth_stats  = {"depth_variance": 0.3}
        fft_feats    = {"low_freq_ratio": 0.6}
        edge_feats   = {"edge_density": 0.1}

        vec = s.build_feature_vector(
            patch_scores, depth_stats, fft_feats, edge_feats
        )
        assert vec.shape == (5,)
        assert vec.dtype == np.float32

    def test_explain_returns_required_keys(self):
        """SHAP explain must return feature_names, shap_values, base_value."""
        from src.xai import SHAPExplainer
        s = SHAPExplainer()
        s.load_background()

        feat_vec = np.array([0.5, 0.8, 0.3, 0.1, 0.7], dtype=np.float32)
        result   = s.explain(feat_vec)

        assert "feature_names" in result
        assert "shap_values" in result
        assert "base_value" in result
        assert len(result["feature_names"]) == 5
        assert len(result["shap_values"])   == 5

    def test_heatmap_to_base64_returns_string(self):
        """Base64 encoder must return non-empty string."""
        from src.xai import heatmap_to_base64
        heatmap = np.random.rand(224, 224).astype(np.float32)
        b64     = heatmap_to_base64(heatmap)

        assert isinstance(b64, str)
        assert len(b64) > 100   # non-trivial content

    def test_heatmap_overlay_with_original(self):
        """Overlay must not raise and must return valid base64."""
        from src.xai import heatmap_to_base64
        heatmap  = np.random.rand(224, 224).astype(np.float32)
        orig_img = make_dummy_image()
        b64      = heatmap_to_base64(heatmap, orig_img)

        assert isinstance(b64, str)
        assert len(b64) > 100