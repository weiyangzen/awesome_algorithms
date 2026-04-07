"""Minimal runnable MVP for Mipmap generation and sampling."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def generate_checkerboard_texture(size: int = 256, checks: int = 64) -> Array:
    """Create a high-frequency RGB checkerboard in [0, 1]."""
    if size <= 0 or checks <= 0:
        raise ValueError("size and checks must be positive")

    y = np.arange(size, dtype=np.float64)[:, None]
    x = np.arange(size, dtype=np.float64)[None, :]
    cell = size / checks

    pattern = ((np.floor(x / cell) + np.floor(y / cell)) % 2.0).astype(np.float64)
    red = pattern
    green = 1.0 - pattern
    blue = 0.2 + 0.8 * pattern
    return np.stack([red, green, blue], axis=-1)


def downsample_2x2_box(image: Array) -> Array:
    """Downsample one mip level with 2x2 box filtering."""
    if image.ndim != 3:
        raise ValueError("image must be HxWxC")

    h, w, c = image.shape
    if h <= 0 or w <= 0 or c <= 0:
        raise ValueError("image shape must be positive")

    if h == 1 and w == 1:
        return image.copy()

    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    return 0.25 * (
        image[0::2, 0::2]
        + image[1::2, 0::2]
        + image[0::2, 1::2]
        + image[1::2, 1::2]
    )


def build_mipmap(base_level: Array) -> list[Array]:
    """Build mip levels from base to 1x1."""
    if base_level.ndim != 3:
        raise ValueError("base_level must be HxWxC")

    levels = [base_level.astype(np.float64, copy=True)]
    while levels[-1].shape[0] > 1 or levels[-1].shape[1] > 1:
        levels.append(downsample_2x2_box(levels[-1]))
    return levels


def bilinear_sample_repeat(image: Array, u: float, v: float) -> Array:
    """Sample image using repeat wrap and bilinear interpolation."""
    h, w, _ = image.shape

    u_wrapped = float(u % 1.0)
    v_wrapped = float(v % 1.0)

    x = u_wrapped * w - 0.5
    y = v_wrapped * h - 0.5

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    tx = x - np.floor(x)
    ty = y - np.floor(y)

    c00 = image[y0 % h, x0 % w]
    c10 = image[y0 % h, x1 % w]
    c01 = image[y1 % h, x0 % w]
    c11 = image[y1 % h, x1 % w]

    c0 = (1.0 - tx) * c00 + tx * c10
    c1 = (1.0 - tx) * c01 + tx * c11
    return (1.0 - ty) * c0 + ty * c1


def sample_no_mipmap(levels: list[Array], u: float, v: float) -> Array:
    return bilinear_sample_repeat(levels[0], u, v)


def sample_with_mipmap(levels: list[Array], u: float, v: float, rho_texels: float) -> Array:
    """Trilinear-style sample: bilinear in two levels + linear blend by fractional LOD."""
    max_level = len(levels) - 1
    lod = float(np.log2(max(rho_texels, 1e-8)))
    lod = float(np.clip(lod, 0.0, float(max_level)))

    l0 = int(np.floor(lod))
    l1 = min(l0 + 1, max_level)
    t = lod - l0

    c0 = bilinear_sample_repeat(levels[l0], u, v)
    c1 = bilinear_sample_repeat(levels[l1], u, v)
    return (1.0 - t) * c0 + t * c1


def render_scene(
    levels: list[Array], out_size: int = 64, tiling: float = 6.0, supersample: int = 8
) -> tuple[Array, Array, Array, float]:
    """Render no-mipmap, mipmap, and supersampled reference images."""
    if out_size <= 0 or supersample <= 0 or tiling <= 0:
        raise ValueError("out_size, supersample, and tiling must be positive")

    base = levels[0]
    h0, w0, _ = base.shape
    rho = max(tiling * w0 / out_size, tiling * h0 / out_size)

    no_mipmap = np.empty((out_size, out_size, 3), dtype=np.float64)
    with_mipmap = np.empty((out_size, out_size, 3), dtype=np.float64)
    reference = np.empty((out_size, out_size, 3), dtype=np.float64)

    inv_out = 1.0 / out_size
    ss_norm = 1.0 / (supersample * supersample)

    for j in range(out_size):
        for i in range(out_size):
            u_center = (i + 0.5) * inv_out * tiling
            v_center = (j + 0.5) * inv_out * tiling

            no_mipmap[j, i] = sample_no_mipmap(levels, u_center, v_center)
            with_mipmap[j, i] = sample_with_mipmap(levels, u_center, v_center, rho)

            acc = np.zeros(3, dtype=np.float64)
            for sy in range(supersample):
                for sx in range(supersample):
                    u = (i + (sx + 0.5) / supersample) * inv_out * tiling
                    v = (j + (sy + 0.5) / supersample) * inv_out * tiling
                    acc += bilinear_sample_repeat(base, u, v)
            reference[j, i] = acc * ss_norm

    np.clip(no_mipmap, 0.0, 1.0, out=no_mipmap)
    np.clip(with_mipmap, 0.0, 1.0, out=with_mipmap)
    np.clip(reference, 0.0, 1.0, out=reference)
    return no_mipmap, with_mipmap, reference, rho


def mse(a: Array, b: Array) -> float:
    return float(np.mean((a - b) ** 2))


def format_level_shapes(levels: list[Array]) -> str:
    return " -> ".join([f"{lv.shape[1]}x{lv.shape[0]}" for lv in levels])


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    base_size = 256
    checks = 64
    out_size = 64
    tiling = 6.3
    supersample = 8

    base_texture = generate_checkerboard_texture(size=base_size, checks=checks)
    levels = build_mipmap(base_texture)
    no_mipmap, with_mipmap, reference, rho = render_scene(
        levels=levels,
        out_size=out_size,
        tiling=tiling,
        supersample=supersample,
    )

    mse_no = mse(no_mipmap, reference)
    mse_mip = mse(with_mipmap, reference)
    improvement = (mse_no - mse_mip) / max(mse_no, 1e-12)

    print("=== Mipmap MVP Demo ===")
    print(f"Base texture size: {base_texture.shape[1]}x{base_texture.shape[0]}")
    print(f"Mipmap levels: {len(levels)}")
    print(f"Level chain: {format_level_shapes(levels)}")
    print(f"Output size: {out_size}x{out_size}")
    print(f"Tiling: {tiling}")
    print(f"Estimated rho (texels/pixel): {rho:.4f}")
    print(f"Estimated lod=log2(rho): {np.log2(rho):.4f}")
    print()
    print(f"MSE(no mipmap, reference): {mse_no:.8f}")
    print(f"MSE(mipmap, reference):    {mse_mip:.8f}")
    print(f"Improvement ratio:         {improvement * 100:.2f}%")

    y, x = out_size // 3, out_size // 3
    print()
    print(f"Sample pixel @ ({x}, {y}):")
    print(f"  no_mipmap = {no_mipmap[y, x]}")
    print(f"  mipmap    = {with_mipmap[y, x]}")
    print(f"  reference = {reference[y, x]}")

    assert mse_mip <= mse_no + 1e-12, "Expected mipmap error to be <= no-mipmap error"


if __name__ == "__main__":
    main()
