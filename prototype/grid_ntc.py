import argparse
import io
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from dds_export import write_dds_rgba8
from ntc_bench import make_test_texture, psnr_db


class GridTextureNTC(nn.Module):
    """Compact multi-scale 2D feature grid + tiny MLP decoder (real NTC architecture)."""

    def __init__(self, grid_size: int = 64, feat_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        # Two spatial scales: coarse (grid_size // 2) + fine (grid_size)
        self.coarse_grid = nn.Parameter(torch.randn(1, feat_dim, grid_size // 2, grid_size // 2) * 0.05)
        self.fine_grid = nn.Parameter(torch.randn(1, feat_dim, grid_size, grid_size) * 0.05)

        in_dim = feat_dim * 2
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (N, 2) in [-1, 1] -> reshape to (1, 1, N, 2) for bilinear grid_sample
        sample_grid = coords.view(1, 1, -1, 2)
        f_coarse = F.grid_sample(self.coarse_grid, sample_grid, mode="bilinear", align_corners=True)
        f_fine = F.grid_sample(self.fine_grid, sample_grid, mode="bilinear", align_corners=True)
        feats = torch.cat([f_coarse, f_fine], dim=1).squeeze(0).squeeze(1).transpose(0, 1)
        return torch.sigmoid(self.decoder(feats))


def run_grid_ntc(
    image_path: str | None,
    out_dds: str,
    size: int,
    grid_size: int,
    steps: int,
    vram_cap_mb: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if image_path and Path(image_path).exists():
        img = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
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

    model = GridTextureNTC(grid_size=grid_size, feat_dim=8, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(coords)
        loss = F.mse_loss(pred, target_t)
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

    # Quantize grids to fp16 before saving so packed weight ratio reflects real storage
    fp16_state = {k: v.half().cpu() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(fp16_state, buf)
    packed_bytes = len(zlib.compress(buf.getvalue(), level=6))
    ratio = raw_bytes / max(packed_bytes, 1)
    score_psnr = psnr_db(target_np, recon_rgb)

    vram_note = "CPU mode"
    if device.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        within_cap = alloc_mb <= vram_cap_mb
        vram_note = f"{torch.cuda.get_device_name(device)} | alloc={alloc_mb:.1f} MB / cap={vram_cap_mb:.1f} MB | within_cap={within_cap}"

    print(f"Device: {device} ({vram_note})")
    print(f"GridNTC->DDS: {out_path} ({out_path.stat().st_size} B) | PackedWeights(fp16)={packed_bytes} B | Ratio={ratio:.2f}x")
    print(f"Decode time: {decode_ms:.2f} ms | PSNR: {score_psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-scale Feature Grid NTC compressor")
    parser.add_argument("--image", type=str, default=None, help="Optional input RGB image path")
    parser.add_argument("--out", type=str, default="grid_recon.dds", help="Output .dds path")
    parser.add_argument("--size", type=int, default=512, help="Target texture resolution (e.g. 512)")
    parser.add_argument("--grid-size", type=int, default=128, help="Fine feature grid resolution")
    parser.add_argument("--steps", type=int, default=600, help="Training steps")
    parser.add_argument("--vram-cap-mb", type=float, default=3500.0, help="Safe VRAM budget cap in MB")
    args = parser.parse_args()
    run_grid_ntc(args.image, args.out, args.size, args.grid_size, args.steps, args.vram_cap_mb)
