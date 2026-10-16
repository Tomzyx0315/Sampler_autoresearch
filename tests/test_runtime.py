"""Server tests: tensor execution, upstream parity, FID statistics, and counters.

Explicitly skipped if runtime dependencies are absent; not a GPU/FID smoke claim.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(m) is not None for m in ("torch", "numpy", "scipy", "PIL", "tqdm", "requests"))


@unittest.skipUnless(AVAILABLE, "Run on server after installing requirements.txt")
class RuntimeTests(unittest.TestCase):
    def test_tensor_reference_parity(self):
        import torch
        from harness.common import ROOT
        from harness.model import Denoiser
        from harness.verify_baselines import load_file
        from baselines.heun import sample as heun
        from baselines.dpmpp import sample as dpmpp
        class Toy(torch.nn.Module):
            sigma_min, sigma_max = 0, float("inf")
            def round_sigma(self, s):
                return torch.as_tensor(s)
            def forward(self, x, s, class_labels=None):
                s = torch.as_tensor(s, device=x.device).reshape(-1, 1, 1, 1)
                return x / (1 + s.square()) + 0.01 * x.tanh()
        net = Toy()
        noise = torch.randn(2, 3, 4, 4, dtype=torch.float64)
        n = 10
        levels = (80 ** (1/7) + torch.arange(n, dtype=torch.float64) / (n-1) * (0.002 ** (1/7) - 80 ** (1/7))) ** 7
        levels = torch.cat([levels, torch.zeros(1, dtype=torch.float64)])
        edm = load_file("test_edm", ROOT / "third_party/edm/generate.py")
        kd = load_file("test_kd", ROOT / "third_party/k_diffusion_reference/sampling.py")
        for ours, reference, calls in (
            (heun, edm.edm_sampler(net, noise, num_steps=n, randn_like=torch.zeros_like), 19),
            (dpmpp, kd.sample_dpmpp_2m(net, noise * levels[0], levels, disable=True), 10),
        ):
            counter = Denoiser(net)
            actual = ours(counter, noise * levels[0], levels.tolist())
            torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
            self.assertEqual(counter.calls, calls)
            self.assertEqual(counter.sample_evaluations, calls * 2)

    def test_feature_statistics_and_fid(self):
        import numpy as np
        from harness.metrics import FeatureStats, fid
        data = np.random.default_rng(5).normal(size=(31, 7))
        stats = FeatureStats(dim=7)
        stats.update(data[:9])
        stats.update(data[9:])
        mean, covariance = stats.finalize()
        np.testing.assert_allclose(mean, data.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(covariance, np.cov(data, rowvar=False), atol=1e-14)
        self.assertAlmostEqual(fid(mean, covariance, mean, covariance), 0, places=10)
        self.assertAlmostEqual(fid(mean, covariance, mean + 1, covariance), 7, places=10)

    def test_seed_batch_independence_and_budget(self):
        import torch
        from harness.model import noise_for_seeds, Denoiser
        a = noise_for_seeds([4, 5, 6], "cpu")
        b = torch.cat([noise_for_seeds([4], "cpu"), noise_for_seeds([5, 6], "cpu")])
        self.assertTrue(torch.equal(a, b))
        denoise = Denoiser(lambda x, s: x, max_nfe=1)
        denoise(a, 1.0)
        with self.assertRaises(RuntimeError):
            denoise(a, 1.0)

    def test_uint8_conversion_and_trace_analysis(self):
        import torch
        import numpy as np
        from harness.metrics import to_uint8
        from playground.analyze_trace import analyze
        self.assertEqual(to_uint8(torch.tensor([-2., -1., 0., 1., 2.])).tolist(), [0, 0, 128, 255, 255])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.npz"
            x = np.ones((4, 2, 3, 2, 2))
            sigma = np.asarray([4., 3., 2., 1.])
            np.savez(path, x=x, denoised=x - sigma[:, None, None, None, None], sigma=sigma)
            result = analyze(path)
            np.testing.assert_allclose(result["mean_adjacent_cosine"], 1)
            self.assertEqual(result["centered_top2_energy_fraction_per_trajectory"], [None, None])


if __name__ == "__main__":
    unittest.main()
