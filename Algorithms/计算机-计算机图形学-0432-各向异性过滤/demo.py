"""各向异性过滤（Anisotropic Filtering）最小可运行 MVP。

实现内容：
1) 生成程序纹理
2) 构建 mipmap 金字塔
3) 三线性（近似各向同性）采样
4) 各向异性多抽样（沿主轴分段积分）
5) 输出对比图与误差指标

运行：
    uv run python demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_procedural_texture(size: int = 256) -> np.ndarray:
    """生成带高频细节的程序纹理，范围 [0,1]。"""
    t = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float64)
    u, v = np.meshgrid(t, t)

    checker = ((np.floor(u * 36) + np.floor(v * 36)) % 2).astype(np.float64)
    stripes = 0.5 + 0.5 * np.sin(2.0 * np.pi * (u * 70.0 + v * 12.0))
    rings = 0.5 + 0.5 * np.sin(2.0 * np.pi * np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) * 85.0)

    r = 0.55 * checker + 0.45 * stripes
    g = 0.60 * (1.0 - checker) + 0.40 * rings
    b = 0.50 * stripes + 0.50 * rings

    tex = np.stack([r, g, b], axis=-1)
    return np.clip(tex, 0.0, 1.0)


def downsample_2x(img: np.ndarray) -> np.ndarray:
    """2x2 盒滤波下采样。"""
    h, w, _ = img.shape
    h2 = max(1, h // 2)
    w2 = max(1, w // 2)

    if h == 1 and w == 1:
        return img

    # 若尺寸为奇数，丢弃最后一行/列，保证可整除
    h_even = h2 * 2
    w_even = w2 * 2
    crop = img[:h_even, :w_even]
    out = (
        crop[0::2, 0::2]
        + crop[1::2, 0::2]
        + crop[0::2, 1::2]
        + crop[1::2, 1::2]
    ) * 0.25
    return out


def build_mipmaps(base: np.ndarray) -> list[np.ndarray]:
    """从 base 纹理构建 mipmap 金字塔。"""
    mips = [base]
    cur = base
    while cur.shape[0] > 1 or cur.shape[1] > 1:
        cur = downsample_2x(cur)
        mips.append(cur)
    return mips


def bilinear_sample(tex: np.ndarray, u: float, v: float) -> np.ndarray:
    """对单层纹理做双线性采样（repeat wrap）。"""
    h, w, _ = tex.shape

    u = u % 1.0
    v = v % 1.0

    x = u * w - 0.5
    y = v * h - 0.5

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - x0
    fy = y - y0

    x0 %= w
    x1 %= w
    y0 %= h
    y1 %= h

    c00 = tex[y0, x0]
    c10 = tex[y0, x1]
    c01 = tex[y1, x0]
    c11 = tex[y1, x1]

    c0 = c00 * (1.0 - fx) + c10 * fx
    c1 = c01 * (1.0 - fx) + c11 * fx
    return c0 * (1.0 - fy) + c1 * fy


def trilinear_sample(mips: list[np.ndarray], u: float, v: float, lod: float) -> np.ndarray:
    """在 mipmap 上做三线性采样。"""
    level_max = len(mips) - 1
    lod = float(np.clip(lod, 0.0, float(level_max)))

    l0 = int(np.floor(lod))
    l1 = min(l0 + 1, level_max)
    t = lod - l0

    c0 = bilinear_sample(mips[l0], u, v)
    c1 = bilinear_sample(mips[l1], u, v)
    return c0 * (1.0 - t) + c1 * t


def make_uv_map(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """构造带透视压缩的 UV 映射，制造明显各向异性。"""
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height
    x, y = np.meshgrid(xs, ys)

    depth = 0.16 + 2.5 * x
    u = 10.0 * (x + 0.16 * y) / depth
    v = 10.0 * (y + 0.06 * np.sin(4.0 * np.pi * x)) / depth
    return u, v


def jacobian_from_uv(u_map: np.ndarray, v_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算每像素雅可比项：du/dx, dv/dx, du/dy, dv/dy。"""
    du_dy, du_dx = np.gradient(u_map)
    dv_dy, dv_dx = np.gradient(v_map)
    return du_dx, dv_dx, du_dy, dv_dy


def render_isotropic(
    mips: list[np.ndarray],
    u_map: np.ndarray,
    v_map: np.ndarray,
    du_dx: np.ndarray,
    dv_dx: np.ndarray,
    du_dy: np.ndarray,
    dv_dy: np.ndarray,
) -> np.ndarray:
    """各向同性近似：按最大轴估计 LOD，只做 1 次三线性采样。"""
    h, w = u_map.shape
    base_size = float(mips[0].shape[0])
    out = np.zeros((h, w, 3), dtype=np.float64)

    for j in range(h):
        for i in range(w):
            gx = np.hypot(du_dx[j, i], dv_dx[j, i])
            gy = np.hypot(du_dy[j, i], dv_dy[j, i])
            rho = max(gx, gy, 1e-8)
            lod = np.log2(max(rho * base_size, 1.0))
            out[j, i] = trilinear_sample(mips, u_map[j, i], v_map[j, i], lod)

    return out


def render_anisotropic(
    mips: list[np.ndarray],
    u_map: np.ndarray,
    v_map: np.ndarray,
    du_dx: np.ndarray,
    dv_dx: np.ndarray,
    du_dy: np.ndarray,
    dv_dy: np.ndarray,
    max_aniso: int = 8,
) -> np.ndarray:
    """近似各向异性过滤：沿主轴做多点积分，LOD 由次轴决定。"""
    h, w = u_map.shape
    base_size = float(mips[0].shape[0])
    out = np.zeros((h, w, 3), dtype=np.float64)

    for j in range(h):
        for i in range(w):
            jaco = np.array(
                [
                    [du_dx[j, i], du_dy[j, i]],
                    [dv_dx[j, i], dv_dy[j, i]],
                ],
                dtype=np.float64,
            )

            # J 的奇异值给出像素足迹椭圆主/次轴尺度
            _, svals, vh = np.linalg.svd(jaco)
            sigma_max = max(float(svals[0]), 1e-8)
            sigma_min = max(float(svals[1]), 1e-8)

            lod_minor = np.log2(max(sigma_min * base_size, 1.0))
            aniso_ratio = sigma_max / sigma_min
            taps = int(np.clip(np.ceil(aniso_ratio), 1, max_aniso))

            v1 = vh[0]  # 屏幕空间主方向
            major_vec = jaco @ v1
            norm = float(np.linalg.norm(major_vec))
            if norm < 1e-12:
                major_dir = np.array([1.0, 0.0], dtype=np.float64)
            else:
                major_dir = major_vec / norm

            accum = np.zeros(3, dtype=np.float64)
            for k in range(taps):
                t = (k + 0.5) / taps - 0.5
                ou = u_map[j, i] + major_dir[0] * t * sigma_max
                ov = v_map[j, i] + major_dir[1] * t * sigma_max
                accum += trilinear_sample(mips, ou, ov, lod_minor)

            out[j, i] = accum / taps

    return out


def write_ppm(path: Path, img: np.ndarray) -> None:
    """写出二进制 PPM，避免依赖图像库。"""
    h, w, _ = img.shape
    data = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(data.tobytes())


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-15:
        return 99.0
    return -10.0 * np.log10(mse)


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    texture = make_procedural_texture(size=256)
    mips = build_mipmaps(texture)

    width, height = 180, 120
    u_map, v_map = make_uv_map(width, height)
    du_dx, dv_dx, du_dy, dv_dy = jacobian_from_uv(u_map, v_map)

    img_iso = render_isotropic(mips, u_map, v_map, du_dx, dv_dx, du_dy, dv_dy)
    img_aniso = render_anisotropic(mips, u_map, v_map, du_dx, dv_dx, du_dy, dv_dy, max_aniso=8)

    # 用更高采样上限的同算法作为近似参考。
    img_ref = render_anisotropic(mips, u_map, v_map, du_dx, dv_dx, du_dy, dv_dy, max_aniso=24)

    write_ppm(out_dir / "output_isotropic.ppm", img_iso)
    write_ppm(out_dir / "output_anisotropic.ppm", img_aniso)
    write_ppm(out_dir / "output_reference.ppm", img_ref)
    write_ppm(out_dir / "output_absdiff.ppm", np.abs(img_iso - img_aniso) * 4.0)

    mse_iso = float(np.mean((img_iso - img_ref) ** 2))
    mse_aniso = float(np.mean((img_aniso - img_ref) ** 2))

    print("== 各向异性过滤 MVP ==")
    print(f"纹理尺寸: {texture.shape[1]}x{texture.shape[0]}")
    print(f"输出尺寸: {width}x{height}")
    print(f"mipmap 层数: {len(mips)}")
    print(f"MSE(各向同性 vs 参考): {mse_iso:.6f}")
    print(f"MSE(各向异性 vs 参考): {mse_aniso:.6f}")
    print(f"PSNR(各向同性, dB): {psnr(img_iso, img_ref):.3f}")
    print(f"PSNR(各向异性, dB): {psnr(img_aniso, img_ref):.3f}")
    if mse_aniso > 0:
        print(f"误差改进倍数(iso/aniso): {mse_iso / mse_aniso:.2f}x")

    print("输出文件:")
    print("- output_isotropic.ppm")
    print("- output_anisotropic.ppm")
    print("- output_reference.ppm")
    print("- output_absdiff.ppm")


if __name__ == "__main__":
    main()
