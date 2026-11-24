import argparse
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


class TinyTextureMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(coords))


def make_test_texture(size: int = 128) -> np.ndarray:
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    r = 0.5 + 0.5 * np.sin(6.0 * x) * np.cos(6.0 * y)
    g = 0.5 + 0.5 * np.cos(4.0 * np.sqrt(x * x + y * y) * np.pi)
    b = 0.5 + 0.5 * np.sin(8.0 * (x + y))
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def psnr_db(orig: np.ndarray, recon: np.ndarray) -> float:
    orig_f = orig.astype(np.float64)
    recon_f = recon.astype(np.float64)
    mse = np.mean((orig_f - recon_f) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def run_bench(image_path: str | None, size: int, steps: int, hidden_dim: int, vram_cap_mb: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if image_path and Path(image_path).exists():
        img = Image.open(image_path).convert("RGB").resize((size, size))
        target_np = np.array(img, dtype=np.uint8)
    else:
        target_np = make_test_texture(size)

    h, w, _ = target_np.shape
    raw_bytes = target_np.nbytes

    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    target_t = torch.from_numpy(target_np.astype(np.float32) / 255.0).to(device).reshape(-1, 3)

    model = TinyTextureMLP(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(coords)
        loss = nn.functional.mse_loss(pred, target_t)
        loss.backward()
        optimizer.step()

    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        recon_t = model(coords).reshape(h, w, 3)
    if device.type == "cuda":
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t0) * 1000.0

    recon_np = np.clip(recon_t.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    ratio = raw_bytes / max(weight_bytes, 1)
    score_psnr = psnr_db(target_np, recon_np)

    vram_info = "CPU-only mode"
    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        total_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
        within_cap = alloc_mb <= vram_cap_mb
        vram_info = (
            f"{torch.cuda.get_device_name(device)} | alloc={alloc_mb:.1f} MB / "
            f"cap={vram_cap_mb:.1f} MB (total={total_mb:.0f} MB) | within_cap={within_cap}"
        )

    print(f"Device: {device} ({vram_info})")
    print(f"Texture: {w}x{h} RGB | Raw={raw_bytes} B | Weights={weight_bytes} B | Ratio={ratio:.2f}x")
    print(f"Decode time: {decode_ms:.2f} ms | PSNR: {score_psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTX NTC Prototype Benchmark")
    parser.add_argument("--image", type=str, default=None, help="Optional path to input RGB image")
    parser.add_argument("--size", type=int, default=128, help="Working square resolution")
    parser.add_argument("--steps", type=int, default=150, help="Training steps for quick prototype test")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden MLP width")
    parser.add_argument("--vram-cap-mb", type=float, default=3500.0, help="Safe VRAM budget cap in MB")
    args = parser.parse_args()
    run_bench(args.image, args.size, args.steps, args.hidden_dim, args.vram_cap_mb)
