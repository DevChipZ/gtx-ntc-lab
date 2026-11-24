import argparse
import io
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from grid_ntc import GridTextureNTC
from ntc_bench import make_test_texture, psnr_db
from vram_budget import TextureAsset, plan_vram_budget


def benchmark_single(
    name: str,
    image_path: Path | None,
    size: int,
    grid_size: int,
    steps: int,
    device: torch.device,
) -> dict:
    if image_path and image_path.exists():
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
    fp16_state = {k: v.half().cpu() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(fp16_state, buf)
    packed_bytes = len(zlib.compress(buf.getvalue(), level=6))

    ratio = raw_bytes / max(packed_bytes, 1)
    score_psnr = psnr_db(target_np, recon_rgb)
    # Estimate working VRAM in MB (packed weights + decoded RGBA surface buffer + headroom)
    est_vram_mb = (packed_bytes + (h * w * 4)) / (1024.0 * 1024.0) + 12.0

    return {
        "name": name,
        "size": f"{w}x{h}",
        "packed_kb": packed_bytes / 1024.0,
        "ratio": ratio,
        "decode_ms": decode_ms,
        "psnr": score_psnr,
        "est_vram_mb": est_vram_mb,
    }


def run_batch(
    images_dir: str | None,
    size: int,
    grid_size: int,
    steps: int,
    vram_cap_mb: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items: list[tuple[str, Path | None]] = []

    if images_dir and Path(images_dir).is_dir():
        files = sorted(
            [p for p in Path(images_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        )[:4]
        for f in files:
            items.append((f.name, f))

    if not items:
        items = [
            ("synth_surface_a", None),
            ("synth_surface_b", None),
            ("synth_surface_c", None),
        ]

    results = [benchmark_single(name, path, size, grid_size, steps, device) for name, path in items]

    print(f"Batch NTC Benchmark (device={device}, cap={vram_cap_mb:.1f} MB):")
    for r in results:
        print(
            f"  - {r['name']} [{r['size']}]: Packed={r['packed_kb']:.1f} KB | "
            f"Ratio={r['ratio']:.2f}x | Decode={r['decode_ms']:.2f} ms | "
            f"PSNR={r['psnr']:.2f} dB | EstVRAM={r['est_vram_mb']:.1f} MB"
        )

    assets = [
        TextureAsset(name=r["name"], width=size, height=size, channels=4, bytes_per_channel=1, priority=idx + 1)
        for idx, r in enumerate(results)
    ]
    plan = plan_vram_budget(assets, vram_cap_mb=vram_cap_mb)
    print("VRAM Budget Plan:", plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch NTC benchmark + VRAM budget check")
    parser.add_argument("--images-dir", type=str, default=None, help="Optional folder of PNG/JPG textures")
    parser.add_argument("--size", type=int, default=256, help="Working square resolution")
    parser.add_argument("--grid-size", type=int, default=128, help="Fine feature grid resolution")
    parser.add_argument("--steps", type=int, default=200, help="Training steps per item")
    parser.add_argument("--vram-cap-mb", type=float, default=3500.0, help="Safe VRAM budget cap in MB")
    args = parser.parse_args()
    run_batch(args.images_dir, args.size, args.grid_size, args.steps, args.vram_cap_mb)

