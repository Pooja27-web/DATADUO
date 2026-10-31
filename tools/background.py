"""A still image of a particle cloud shaped like a twisted ring, used as the page background.

It follows the AXON reference (same camera, group transform and point-cloud look) but is drawn
once with numpy instead of being animated, so the page has no motion apart from hover colours.
"""
import base64
import io
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter

from tools.palette import ACCENT


def _hex(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def _sdf(p):
    """Signed distance to a smooth ring with one large opening, lobed and bent in depth."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    a = np.arctan2(y, x)
    r = np.hypot(x, y)
    major = 0.27 + 0.025 * np.sin(2 * a - 0.4)
    tube = 0.085 + 0.03 * np.sin(3 * a + 1.3) + 0.018 * np.sin(5 * a - 0.7)
    zc = 0.07 * np.sin(2 * a + 0.6)
    return np.hypot(r - major, z - zc) - tube


def _normal(p, e=1e-3):
    n = np.stack([
        _sdf(p + np.array([e, 0, 0])) - _sdf(p - np.array([e, 0, 0])),
        _sdf(p + np.array([0, e, 0])) - _sdf(p - np.array([0, e, 0])),
        _sdf(p + np.array([0, 0, e])) - _sdf(p - np.array([0, 0, e])),
    ], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def _rotation(deg):
    ax, ay, az = np.radians(deg)
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rx @ ry @ rz


@lru_cache(maxsize=4)
def background_data_uri(width=1600, height=900, n=170000, seed=7):
    rng = np.random.default_rng(seed)

    # 1. points pulled onto the surface, plus a faint halo of loose points
    p = rng.uniform(-0.45, 0.45, (n, 3))
    for _ in range(14):
        p -= _normal(p) * _sdf(p)[:, None] * 0.8
    p += rng.normal(0, 0.0035, p.shape)
    halo = rng.random(n) < 0.06
    p[halo] += rng.normal(0, 0.03, (int(halo.sum()), 3))

    # 2. group transform and camera from the reference (scale, then rotate, then translate)
    world = (p * np.array([5.0, 5.0, 6.0])) @ _rotation([-28.4, 390.0, 4.92]).T + np.array([0.343, 0.16, -1.0])
    cam = np.array([-1.17, 0.0, 6.0])
    f = -cam / np.linalg.norm(cam)
    right = np.cross(f, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    v = world - cam
    xv, yv, zv = v @ right, v @ up, v @ f
    t = np.tan(np.radians(15.0))
    px = ((xv / (zv * t * (width / height))) + 1) / 2 * width
    py = (1 - (yv / (zv * t))) / 2 * height

    # 3. grey points with a few accent points, depth-weighted
    r = rng.random((n, 3))
    weight = (0.25 + 0.75 * r[:, 0]) * np.clip(1.3 - 0.12 * (zv - 5.0), 0.5, 1.4)
    accent = r[:, 2] > 0.96
    weight = np.where(accent, weight * 2.2, weight)
    colour = np.where(accent[:, None], _hex(ACCENT), np.ones(3))
    ok = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    idx = py[ok].astype(int) * width + px[ok].astype(int)
    img = np.stack([
        np.bincount(idx, weights=(weight * colour[:, c])[ok], minlength=width * height)
        for c in range(3)
    ], axis=-1).reshape(height, width, 3)

    # 4. soft sprites and a faint glow, then tone-map
    base = Image.fromarray(np.clip(img / max(img.max(), 1e-9) * 255, 0, 255).astype("uint8"))
    small = np.asarray(base.filter(ImageFilter.GaussianBlur(0.9)), dtype=float)
    glow = np.asarray(base.filter(ImageFilter.GaussianBlur(3.0)), dtype=float)
    mix = small + 0.7 * glow
    lum = mix.mean(axis=2)
    ref = np.percentile(lum[lum > 0.5], 99.5) if (lum > 0.5).any() else 1.0
    out = (1 - np.exp(-mix / max(ref, 1e-6) * 1.6)) * 0.9
    jpg = Image.fromarray((out * 255).astype("uint8"))
    buf = io.BytesIO()
    jpg.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    uri = background_data_uri()
    open("background_preview.jpg", "wb").write(base64.b64decode(uri.split(",", 1)[1]))
    print("kb:", len(uri) * 3 // 4 // 1024)
