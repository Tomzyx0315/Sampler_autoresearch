"""NVIDIA Inception features + unbiased covariance, matching EDM's FID protocol."""
import pickle
import numpy as np
import scipy.linalg
import torch
from . import model  # registers the vendored persistence imports before unpickling


def to_uint8(samples):
    # Exact EDM image quantization; feed these bytes to FID and saved PNGs alike.
    return (samples * 127.5 + 128).clamp(0, 255).to(torch.uint8)


def load_detector(path, device):
    with open(path, "rb") as stream:
        return pickle.load(stream).to(device).eval().requires_grad_(False)


class FeatureStats:
    def __init__(self, dim=2048):
        self.count = 0
        self.total = np.zeros(dim, dtype=np.float64)
        self.products = np.zeros((dim, dim), dtype=np.float64)

    def update(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.total) or not np.isfinite(features).all():
            raise ValueError("Invalid Inception features")
        self.count += len(features)
        self.total += features.sum(axis=0)
        self.products += features.T @ features

    def finalize(self):
        if self.count < 2:
            raise ValueError("FID requires at least two samples")
        mean = self.total / self.count
        covariance = (self.products - self.count * np.outer(mean, mean)) / (self.count - 1)
        return mean, (covariance + covariance.T) / 2


def fid(mean, covariance, reference_mean, reference_covariance):
    # SciPy <=1.16 returns ``(root, error)`` when disp=False; newer SciPy
    # removed that argument and returns the matrix directly.
    try:
        root = scipy.linalg.sqrtm(covariance @ reference_covariance, disp=False)
    except TypeError:
        root = scipy.linalg.sqrtm(covariance @ reference_covariance)
    if isinstance(root, tuple):
        root = root[0]
    if not np.isfinite(root).all():
        raise ValueError("Nonfinite covariance square root")
    if np.iscomplexobj(root):
        scale = max(1.0, float(np.max(np.abs(root.real))))
        if float(np.max(np.abs(root.imag))) > 1e-5 * scale:
            raise ValueError("Large imaginary component in covariance square root")
        root = root.real
    result = float(np.square(mean - reference_mean).sum() + np.trace(covariance + reference_covariance - 2 * root))
    if not np.isfinite(result) or result < -1e-5:
        raise ValueError(f"Invalid FID: {result}")
    return max(0.0, result)
