"""Minimal runnable MVP: midpoint ellipse rasterization."""

from __future__ import annotations

from typing import Iterable


Point = tuple[int, int]


def _add_symmetric_points(store: set[Point], xc: int, yc: int, x: int, y: int) -> None:
    """Insert 4-way symmetric points generated from the first-quadrant sample."""
    store.add((xc + x, yc + y))
    store.add((xc - x, yc + y))
    store.add((xc + x, yc - y))
    store.add((xc - x, yc - y))


def midpoint_ellipse_points(rx: int, ry: int, xc: int = 0, yc: int = 0) -> list[Point]:
    """Generate raster points for an ellipse using the midpoint ellipse algorithm."""
    if rx <= 0 or ry <= 0:
        raise ValueError("rx and ry must be positive integers")

    rx2 = rx * rx
    ry2 = ry * ry

    x = 0
    y = ry
    dx = 2 * ry2 * x
    dy = 2 * rx2 * y

    points: set[Point] = set()
    _add_symmetric_points(points, xc, yc, x, y)

    # Region 1
    d1 = ry2 - (rx2 * ry) + 0.25 * rx2
    while dx < dy:
        if d1 < 0:
            x += 1
            dx = dx + 2 * ry2
            d1 = d1 + dx + ry2
        else:
            x += 1
            y -= 1
            dx = dx + 2 * ry2
            dy = dy - 2 * rx2
            d1 = d1 + dx - dy + ry2
        _add_symmetric_points(points, xc, yc, x, y)

    # Region 2
    d2 = (ry2 * ((x + 0.5) * (x + 0.5))) + (rx2 * ((y - 1) * (y - 1))) - (rx2 * ry2)
    while y >= 0:
        if d2 > 0:
            y -= 1
            dy = dy - 2 * rx2
            d2 = d2 + rx2 - dy
        else:
            x += 1
            y -= 1
            dx = dx + 2 * ry2
            dy = dy - 2 * rx2
            d2 = d2 + dx - dy + rx2
        _add_symmetric_points(points, xc, yc, x, y)

    return sorted(points, key=lambda p: (p[1], p[0]))


def render_ascii(points: Iterable[Point], padding: int = 1) -> tuple[str, tuple[int, int, int, int]]:
    """Render points to an ASCII canvas and return canvas text with coordinate bounds."""
    point_list = list(points)
    if not point_list:
        return "", (0, 0, 0, 0)

    xs = [p[0] for p in point_list]
    ys = [p[1] for p in point_list]

    xmin = min(xs) - padding
    xmax = max(xs) + padding
    ymin = min(ys) - padding
    ymax = max(ys) + padding

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    grid = [["." for _ in range(width)] for _ in range(height)]
    for px, py in point_list:
        row = ymax - py
        col = px - xmin
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = "#"

    lines = ["".join(row) for row in grid]
    return "\n".join(lines), (xmin, xmax, ymin, ymax)


def implicit_error_stats(points: Iterable[Point], rx: int, ry: int, xc: int = 0, yc: int = 0) -> tuple[float, float]:
    """Return max and mean implicit-function residual for generated points."""
    rx2 = float(rx * rx)
    ry2 = float(ry * ry)

    residuals: list[float] = []
    for x, y in points:
        nx = (x - xc) * (x - xc) / rx2
        ny = (y - yc) * (y - yc) / ry2
        residuals.append(abs(nx + ny - 1.0))

    max_err = max(residuals) if residuals else 0.0
    mean_err = (sum(residuals) / len(residuals)) if residuals else 0.0
    return max_err, mean_err


def main() -> None:
    rx, ry = 18, 10
    xc, yc = 0, 0

    points = midpoint_ellipse_points(rx=rx, ry=ry, xc=xc, yc=yc)
    canvas, bounds = render_ascii(points, padding=1)
    max_err, mean_err = implicit_error_stats(points, rx=rx, ry=ry, xc=xc, yc=yc)

    print("Midpoint Ellipse MVP")
    print(f"center=({xc}, {yc}), rx={rx}, ry={ry}")
    print(f"points={len(points)}")
    print(f"bounds(xmin,xmax,ymin,ymax)={bounds}")
    print(f"implicit residual: max={max_err:.6f}, mean={mean_err:.6f}")
    print()
    print(canvas)


if __name__ == "__main__":
    main()
