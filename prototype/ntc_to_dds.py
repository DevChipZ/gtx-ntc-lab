import argparse
import io
import math
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from dds_export import write_dds_rgba8
from ntc_bench import make_test_texture, psnr_db


class FourierTextureMLP(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_frequencies: int = 64, sigma: float = 10.0):
        super().__init__()
        # Fixed Gaussian frequency matrix B so the network can learn sharp edges
        torch.manual_seed(42)
        B = torch.randn(2, num_frequencies) * sigma
        self.register_buffer("B", B)

        in_dim = num_frequencies * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        proj = (2.0 * math.pi) * (coords @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        features = self.encode(coords)
        return torch.sigmoid(self.net(features))


def run_ntc_to_dds(
    image_path: str | None,
    out_dds: str,
    size: int,
    steps: int,
    hidden_dim: int,
    sigma: float,
    vram_cap_mb: float,
) -> None:
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

    model = FourierTextureMLP(hidden_dim=hidden_dim, num_frequencies=64, sigma=sigma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

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

    recon_rgb = np.clip(recon_t.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    recon_rgba = np.concatenate([recon_rgb, alpha], axis=-1)

    out_path = Path(out_dds)
    write_dds_rgba8(out_path, recon_rgba)

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    packed_bytes = len(zlib.compress(buf.getvalue(), level=6))
    ratio = raw_bytes / max(packed_bytes, 1)
    score_psnr = psnr_db(target_np, recon_rgb)

    vram_note = "CPU mode"
    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        within_cap = alloc_mb <= vram_cap_mb
        vram_note = f"{torch.cuda.get_device_name(device)} | alloc={alloc_mb:.1f} MB / cap={vram_cap_mb:.1f} MB | within_cap={within_cap}"

    print(f"Device: {device} ({vram_note})")
    print(f"NTC->DDS: {out_path} ({out_path.stat().st_size} B) | PackedWeights={packed_bytes} B | Ratio={ratio:.2f}x")
    print(f"Decode time: {decode_ms:.2f} ms | PSNR: {score_psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/decode Fourier NTC texture and export DDS")
    parser.add_argument("--image", type=str, default=None, help="Optional input RGB image path")
    parser.add_argument("--out", type=str, default="ntc_recon.dds", help="Output .dds path")
    parser.add_argument("--size", type=int, default=256, help="Working square resolution")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden MLP width")
    parser.add_argument("--sigma", type=float, default=10.0, help="Fourier frequency bandwidth")
    parser.add_argument("--vram-cap-mb", type=float, default=3500.0, help="Safe VRAM budget cap in MB")
    args = parser.parse_args()
    run_ntc_to_dds(
        args.image,
        args.out,
        args.size,
        args.steps,
        args.hidden_dim,
        args.sigma,
        args.vram_cap_mb,
    )
