from __future__ import annotations
import math
from .geometry import point_segment_distance


def circle_polygon_vertices(center, radius_m, segments=64):
    """Return a conservative polygon approximation of a circle in metres.

    The polygon is circumscribed (rather than inscribed), so the MPC's existing
    polygon-distance checks do not underestimate the requested circle radius.
    """
    try:
        cx, cy = float(center[0]), float(center[1])
        radius = float(radius_m)
        count = int(segments)
    except (TypeError, ValueError, IndexError):
        raise ValueError("圆心和半径必须是有效数值。")
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(radius)):
        raise ValueError("圆心和半径必须是有限数值。")
    if radius <= 0:
        raise ValueError("圆形障碍物半径必须大于 0。")
    if count < 16 or count > 512:
        raise ValueError("圆形障碍物边界分段数必须在 16～512 之间。")
    boundary_radius = radius / math.cos(math.pi / count)
    return [
        (cx + boundary_radius * math.cos(2.0 * math.pi * i / count),
         cy + boundary_radius * math.sin(2.0 * math.pi * i / count))
        for i in range(count)
    ]


def point_in_polygon(x, y, vertices):
    inside = False
    n = len(vertices)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def distance_to_polygon(x, y, vertices):
    if point_in_polygon(x, y, vertices):
        return 0.0
    distances = []
    for i in range(len(vertices)):
        ax, ay = vertices[i]
        bx, by = vertices[(i + 1) % len(vertices)]
        distances.append(point_segment_distance(x, y, ax, ay, bx, by))
    return min(distances, default=float("inf"))


def polygon_area(vertices):
    """Return the absolute shoelace area of a polygon in square metres."""
    if len(vertices) < 3:
        return 0.0
    return abs(sum(
        vertices[i][0] * vertices[(i + 1) % len(vertices)][1]
        - vertices[(i + 1) % len(vertices)][0] * vertices[i][1]
        for i in range(len(vertices))
    )) * 0.5


def _orientation(a, b, c, eps=1e-9):
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def _on_segment(a, b, p, eps=1e-9):
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def _segments_intersect(a, b, c, d):
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(a, b, c):
        return True
    if o2 == 0 and _on_segment(a, b, d):
        return True
    if o3 == 0 and _on_segment(c, d, a):
        return True
    if o4 == 0 and _on_segment(c, d, b):
        return True
    return False


def validate_polygon(vertices, min_area_m2=1.0):
    """Validate a user-drawn simple polygon; return (valid, reason)."""
    try:
        points = [(float(x), float(y)) for x, y in vertices]
    except (TypeError, ValueError):
        return False, "顶点坐标无效。"
    if len(points) < 3:
        return False, "多边形至少需要 3 个顶点。"
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in points):
        return False, "顶点坐标必须是有限数值。"
    # A repeated closing point is harmless; remove it before edge checks.
    if len(points) > 3 and math.dist(points[0], points[-1]) < 1e-6:
        points.pop()
    if len(points) < 3:
        return False, "多边形至少需要 3 个不同顶点。"
    for i in range(len(points)):
        if math.dist(points[i], points[(i + 1) % len(points)]) < 1e-3:
            return False, "存在重合或过近的相邻顶点，请撤销后重新绘制。"
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            # Adjacent edges share a legitimate endpoint; skip them.
            if j == i or j == (i + 1) % n or (j + 1) % n == i:
                continue
            c, d = points[j], points[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                return False, "多边形边线发生交叉，请重新绘制为不自交的封闭区域。"
    if polygon_area(points) < float(min_area_m2):
        return False, "多边形面积过小，请扩大障碍物范围。"
    return True, "有效多边形"
