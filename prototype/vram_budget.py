from dataclasses import dataclass


@dataclass
class TextureAsset:
    name: str
    width: int
    height: int
    channels: int
    bytes_per_channel: int = 1
    priority: int = 1  # 1 = high (hero/close-up), higher = lower priority

    @property
    def raw_bytes(self) -> int:
        return self.width * self.height * self.channels * self.bytes_per_channel


def plan_vram_budget(assets: list[TextureAsset], vram_cap_mb: float = 3500.0) -> list[dict]:
    cap_bytes = int(vram_cap_mb * 1024 * 1024)
    sorted_assets = sorted(assets, key=lambda a: a.priority)
    used_bytes = 0
    plan = []

    for asset in sorted_assets:
        scale = 1.0
        w, h = asset.width, asset.height
        est_bytes = asset.raw_bytes

        while used_bytes + est_bytes > cap_bytes and (w > 128 and h > 128):
            w //= 2
            h //= 2
            scale *= 0.5
            est_bytes = w * h * asset.channels * asset.bytes_per_channel

        within_cap = (used_bytes + est_bytes) <= cap_bytes
        if within_cap:
            used_bytes += est_bytes

        plan.append(
            {
                "name": asset.name,
                "priority": asset.priority,
                "target_res": f"{w}x{h}",
                "scale": scale,
                "est_mb": round(est_bytes / (1024 * 1024), 2),
                "loaded_to_vram": within_cap,
                "cumulative_vram_mb": round(used_bytes / (1024 * 1024), 2),
            }
        )

    return plan


if __name__ == "__main__":
    sample_assets = [
        TextureAsset("hero_armor_bundle", 2048, 2048, 4, priority=1),
        TextureAsset("weapon_blade_bundle", 2048, 2048, 4, priority=1),
        TextureAsset("terrain_ground_bundle", 2048, 2048, 4, priority=2),
        TextureAsset("distant_cliff_bundle", 2048, 2048, 4, priority=3),
        TextureAsset("foliage_cluster_bundle", 1024, 1024, 4, priority=3),
    ]
    for row in plan_vram_budget(sample_assets, vram_cap_mb=3500.0):
        print(row)
