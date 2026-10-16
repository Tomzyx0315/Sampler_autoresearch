"""EDM loading and counted access to the frozen denoiser."""
import pickle
import sys
import torch
from .common import ROOT

# Needed by NVIDIA's self-contained model pickles. The source is pinned in Git.
EDM_ROOT = ROOT / "third_party/edm"
if str(EDM_ROOT) not in sys.path:
    sys.path.insert(0, str(EDM_ROOT))


def load_network(path, device):
    # Only use the official asset downloaded by prepare; pickle contains Python code.
    with open(path, "rb") as stream:
        net = pickle.load(stream)["ema"].to(device).eval().requires_grad_(False)
    if (net.img_resolution, net.img_channels, net.label_dim) != (32, 3, 0):
        raise ValueError("Expected the fixed unconditional CIFAR-10 model")
    return net


def noise_for_seeds(seeds, device):
    # CPU per-sample RNG makes initial noise independent of batch partition/device.
    return torch.stack([
        torch.randn(3, 32, 32, generator=torch.Generator(device="cpu").manual_seed(int(seed)))
        for seed in seeds
    ]).to(device=device, dtype=torch.float64)


class Denoiser:
    def __init__(self, net, trace_samples=0):
        self.net = net
        self.calls = 0
        self.sample_evaluations = 0
        self.trace_samples = trace_samples
        self.trace = []

    def __call__(self, x, sigma):
        sigma_tensor = torch.as_tensor(sigma, dtype=torch.float64, device=x.device)
        if not bool(torch.isfinite(sigma_tensor).all()) or not bool((sigma_tensor > 0).all()):
            raise ValueError("Denoiser sigma must be positive and finite")
        self.calls += 1
        self.sample_evaluations += x.shape[0]
        result = self.net(x, sigma_tensor).to(torch.float64)
        if self.trace_samples:
            if sigma_tensor.numel() != 1:
                raise ValueError("Trace recording currently requires scalar sigma per call")
            self.trace.append({
                "sigma": float(sigma_tensor),
                "x": x[:self.trace_samples].detach().float().cpu().numpy().copy(),
                "denoised": result[:self.trace_samples].detach().float().cpu().numpy().copy(),
            })
        return result


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
