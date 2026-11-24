import argparse
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def write_dds_rgba8(out_path: Path, rgba: np.ndarray) -> None:
    """Write a valid uncompressed 32-bit DDS (DXGI_FORMAT_R8G8B8A8_UNORM / legacy DDSURFACEDESC2)."""
    h, w, c = rgba.shape
    assert c == 4, f"Expected 4 channels (RGBA), got {c}"

    # DDS magic + 124-byte header (DDSURFACEDESC2)
    DDS_MAGIC = b"DDS "
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_PIXELFORMAT = 0x1000
    DDSD_LINEARSIZE = 0x80000

    DDPF_RGB = 0x40
    DDPF_ALPHAPIXELS = 0x1
    DDSCAPS_TEXTURE = 0x1000

    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    pitch_or_linear = w * 4

    # Pixel format (32 bytes): size, flags, fourCC, rgbBitCount, rMask, gMask, bMask, aMask
    pf_flags = DDPF_RGB | DDPF_ALPHAPIXELS
    pixel_format = struct.pack(
        "<II4sIIIII",
        32,
        pf_flags,
        b"\x00\x00\x00\x00",
        32,
        0x000000FF,
        0x0000FF00,
        0x00FF0000,
        0xFF000000,
    )

    header = struct.pack(
        "<IIIIIII44s32sIIII20s",
        124,
        flags,
        h,
        w,
        pitch_or_linear,
        0,  # depth
        0,  # mipmap count
        b"\x00" * 44,
        pixel_format,
        DDSCAPS_TEXTURE,
        0,
        0,
        0,
        b"\x00" * 20,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(DDS_MAGIC + header + rgba.tobytes())


def export_texture_to_dds(
    image_path: str | None,
    out_dds: str = "out_texture.dds",
    size: int = 128,
    bc_format: str = "BC7_UNORM",
) -> Path:
    if image_path and Path(image_path).exists():
        img = Image.open(image_path).convert("RGBA").resize((size, size))
        rgba = np.array(img, dtype=np.uint8)
    else:
        y, x = np.meshgrid(
            np.linspace(-1.0, 1.0, size, dtype=np.float32),
            np.linspace(-1.0, 1.0, size, dtype=np.float32),
            indexing="ij",
        )
        r = np.clip((0.5 + 0.5 * np.sin(6.0 * x) * np.cos(6.0 * y)) * 255.0, 0, 255).astype(np.uint8)
        g = np.clip((0.5 + 0.5 * np.cos(4.0 * np.sqrt(x * x + y * y) * np.pi)) * 255.0, 0, 255).astype(np.uint8)
        b = np.clip((0.5 + 0.5 * np.sin(8.0 * (x + y))) * 255.0, 0, 255).astype(np.uint8)
        a = np.full_like(r, 255)
        rgba = np.stack([r, g, b, a], axis=-1)

    out_path = Path(out_dds)
    temp_png = out_path.with_suffix(".tmp.png")
    temp_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(temp_png)

    texconv = shutil.which("texconv")
    if texconv:
        cmd = [
            texconv,
            "-f",
            bc_format,
            "-o",
            str(out_path.parent),
            str(temp_png),
        ]
        subprocess.run(cmd, check=True)
        produced = out_path.parent / f"{temp_png.stem}.DDS"
        if produced.exists():
            produced.replace(out_path)
    else:
        write_dds_rgba8(out_path, rgba)

    if temp_png.exists():
        temp_png.unlink()

    print(f"Exported DDS: {out_path} ({out_path.stat().st_size} bytes) | mode={bc_format}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export texture to DDS")
    parser.add_argument("--image", type=str, default=None, help="Optional input image path")
    parser.add_argument("--out", type=str, default="out_texture.dds", help="Output .dds path")
    parser.add_argument("--size", type=int, default=128, help="Square resolution")
    parser.add_argument("--bc-format", type=str, default="BC7_UNORM", help="Target BC format when texconv is present")
    args = parser.parse_args()
    export_texture_to_dds(args.image, args.out, args.size, args.bc_format)
