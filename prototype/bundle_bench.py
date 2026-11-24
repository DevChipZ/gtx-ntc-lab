import argparse
import io
import math
import time
import zlib

import numpy as np
import torch
import torch.nn as nn


class BundleTextureMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32, out_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(coords))


def make_rgba_bundle(size: int = 128) -> np.ndarray:
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    r = 0.5 + 0.5 * np.sin(6.0 * x) * np.cos(6.0 * y)
    g = 0.5 + 0.5 * np.cos(4.0 * np.sqrt(x * x + y * y) * np.pi)
    b = 0.5 + 0.5 * np.sin(8.0 * (x + y))
    # 4th channel simulates roughness / micro-detail map
    roughness = 0.5 + 0.5 * np.sin(12.0 * x - 7.0 * y)
    bundle = np.stack([r, g, b, roughness], axis=-1)
    return np.clip(bundle * 255.0, 0, 255).astype(np.uint8)


def psnr_db(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def compressed_weights_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(zlib.compress(buf.getvalue(), level=6))


def run_bundle_bench(size: int, steps: int, hidden_dim: int, vram_cap_mb: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_np = make_rgba_bundle(size)
    h, w, c = target_np.shape
    raw_bytes = target_np.nbytes

    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    target_t = torch.from_numpy(target_np.astype(np.float32) / 255.0).to(device).reshape(-1, c)

    model = BundleTextureMLP(hidden_dim=hidden_dim, out_channels=c).to(device)
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
        recon_t = model(coords).reshape(h, w, c)
    if device.type == "cuda":
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t0) * 1000.0

    recon_np = np.clip(recon_t.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    packed_bytes = compressed_weights_bytes(model)
    ratio = raw_bytes / max(packed_bytes, 1)
    rgb_psnr = psnr_db(target_np[..., :3], recon_np[..., :3])
    ch4_psnr = psnr_db(target_np[..., 3], recon_np[..., 3])

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
    print(f"Bundle: {w}x{h} x{c}ch | Raw={raw_bytes} B | PackedWeights={packed_bytes} B | Ratio={ratio:.2f}x")
    print(f"Decode time: {decode_ms:.2f} ms | RGB PSNR: {rgb_psnr:.2f} dB | Ch4 PSNR: {ch4_psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-Channel Texture Bundle Benchmark")
    parser.add_argument("--size", type=int, default=128, help="Working square resolution")
    parser.add_argument("--steps", type=int, default=150, help="Training steps")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden MLP width")
    parser.add_argument("--vram-cap-mb", type=float, default=3500.0, help="Safe VRAM budget cap in MB")
    args = parser.parse_args()
    run_bundle_bench(args.size, args.steps, args.hidden_dim, args.vram_cap_mb)
