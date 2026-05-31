#!/usr/bin/env python3
"""Convert an image to the Nord colour palette via Oklab snapping."""

import argparse
import sys

import cv2
import numpy as np

LOG_EPS = 1e-4   # floor for log-space pigment transform; log(0+LOG_EPS) ≈ -9.2
KM_EPS  = 1e-3   # minimum reflectance before K/S conversion; K/S(KM_EPS) ≈ 499


def _lin_to_km(r):
    """Linear reflectance → Kubelka-Munk K/S ratio: (1-R)² / 2R."""
    r = np.maximum(r, KM_EPS)
    return (1.0 - r) ** 2 / (2.0 * r)


def _km_to_lin(ks):
    """Kubelka-Munk K/S ratio → linear reflectance.

    Numerically stable form: R = 1 / (1 + K/S + sqrt(K/S² + 2·K/S))
    avoids catastrophic cancellation for large K/S (dark colours).
    """
    ks = np.maximum(ks, 0.0)
    return np.clip(1.0 / (1.0 + ks + np.sqrt(ks * ks + 2.0 * ks + 1e-12)), 0.0, 1.0)


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)


def _linear_rgb_to_oklab(r, g, b):
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    L  = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a  = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ = 0.0259040371 * l_ + 0.4072165126 * m_ - 0.4331205297 * s_
    return L, a, b_


def _oklab_to_linear_rgb(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r  =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g  = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_ = -0.0041960863 * l - 0.7034186147 * m + 1.6956086611 * s
    return r, g, b_


def _bgr_to_oklab(bgr):
    """Convert an (…, 3) float32 BGR array in [0, 1] to Oklab."""
    r = _srgb_to_linear(bgr[..., 2])
    g = _srgb_to_linear(bgr[..., 1])
    b = _srgb_to_linear(bgr[..., 0])
    return np.stack(_linear_rgb_to_oklab(r, g, b), axis=-1)


def _oklab_to_bgr(lab):
    """Convert an (…, 3) Oklab array to float32 BGR in [0, 1]."""
    r, g, b = _oklab_to_linear_rgb(lab[..., 0], lab[..., 1], lab[..., 2])
    return np.stack([
        np.clip(_linear_to_srgb(b), 0.0, 1.0),
        np.clip(_linear_to_srgb(g), 0.0, 1.0),
        np.clip(_linear_to_srgb(r), 0.0, 1.0),
    ], axis=-1)


PALETTE_BGR = [
    (  0,   0,   0),  # black   #000000  (extended — full lightness range)
    ( 64,  52,  46),  # nord0   #2E3440  Polar Night
    ( 82,  66,  59),  # nord1   #3B4252  Polar Night
    ( 94,  76,  67),  # nord2   #434C5E  Polar Night
    (106,  86,  76),  # nord3   #4C566A  Polar Night
    (233, 222, 216),  # nord4   #D8DEE9  Snow Storm
    (240, 233, 229),  # nord5   #E5E9F0  Snow Storm
    (244, 239, 236),  # nord6   #ECEFF4  Snow Storm
    (187, 188, 143),  # nord7   #8FBCBB  Frost
    (208, 192, 136),  # nord8   #88C0D0  Frost
    (193, 161, 129),  # nord9   #81A1C1  Frost
    (172, 129,  94),  # nord10  #5E81AC  Frost
    (106,  97, 191),  # nord11  #BF616A  Aurora
    (112, 135, 208),  # nord12  #D08770  Aurora
    (139, 203, 235),  # nord13  #EBCB8B  Aurora
    (140, 190, 163),  # nord14  #A3BE8C  Aurora
    (173, 142, 180),  # nord15  #B48EAD  Aurora
]


def build_lookup():
    """Pre-compute Oklab (L, a, b) and Oklch hue H for each palette colour."""
    bgr = np.array(PALETTE_BGR, dtype=np.float32) / 255.0
    lab = _bgr_to_oklab(bgr)  # (16, 3)
    palette = []
    for row in lab:
        L, a, b = float(row[0]), float(row[1]), float(row[2])
        palette.append({'L': L, 'a': a, 'b': b, 'H': float(np.arctan2(b, a))})
    return palette



def _gen_blue_noise(size=64, sigma=1.5, seed=42):
    """Void-and-cluster blue-noise threshold texture, (size, size) float32 in [0, 1)."""
    N = size
    rng = np.random.default_rng(seed)

    fy = np.fft.fftfreq(N)[:, None]
    fx = np.fft.fftfreq(N)[None, :]
    gauss_fft = np.exp(-2.0 * np.pi**2 * sigma**2 * (fx**2 + fy**2))

    def _conv(m):
        return np.real(np.fft.ifft2(np.fft.fft2(m.astype(np.float64)) * gauss_fft)).astype(np.float32)

    flat = np.zeros(N * N, dtype=np.float32)
    flat[: N * N // 2] = 1.0
    rng.shuffle(flat)
    mask = flat.reshape(N, N)

    # Phase 1: relax — move 1s from tightest clusters to loosest voids
    for _ in range(N * N):
        e = _conv(mask)
        ci = np.unravel_index(np.where(mask > 0, e, -np.inf).argmax(), (N, N))
        vi = np.unravel_index(np.where(mask == 0, e,  np.inf).argmin(), (N, N))
        if e[ci] <= e[vi]:
            break
        mask[ci] = 0.0
        mask[vi] = 1.0

    # Phase 2: assign threshold ranks
    result = np.empty(N * N, dtype=np.float32)
    n = int(mask.sum())
    tmp = mask.copy()

    for rank in range(n - 1, -1, -1):           # rank tightest 1-pixels first
        e = _conv(tmp)
        ci = np.unravel_index(np.where(tmp > 0, e, -np.inf).argmax(), (N, N))
        result[ci[0] * N + ci[1]] = (rank + 0.5) / (N * N)
        tmp[ci] = 0.0

    for rank in range(n, N * N):                 # then rank loosest 0-pixels
        e = _conv(tmp)
        vi = np.unravel_index(np.where(tmp == 0, e, np.inf).argmin(), (N, N))
        result[vi[0] * N + vi[1]] = (rank + 0.5) / (N * N)
        tmp[vi] = 1.0

    return result.reshape(N, N)


_BN_TEXTURE: np.ndarray | None = None


def _threshold_texture(rows, cols):
    """Return a (rows, cols) float32 blue-noise threshold texture."""
    global _BN_TEXTURE
    if _BN_TEXTURE is None:
        print("  Generating blue-noise texture...", file=sys.stderr, flush=True)
        _BN_TEXTURE = _gen_blue_noise(64)
    s = _BN_TEXTURE.shape[0]
    return np.tile(_BN_TEXTURE, (rows // s + 1, cols // s + 1))[:rows, :cols]


def _dither(pix_L, pix_a, pix_b, pix_C, pal_L, pal_a, pal_b, pal_H, threshold):
    """Floyd-Steinberg error diffusion in Oklab (a, b) space, seeded with blue noise.
    threshold: (rows, cols) float32 in [0, 1) — blue-noise texture."""
    rows, cols = pix_L.shape
    scale = float(np.sqrt((np.var(pal_a) + np.var(pal_b)) / 2))
    err_a = np.zeros((rows, cols), dtype=np.float32)
    err_b = np.zeros((rows, cols), dtype=np.float32)
    out_a = np.empty((rows, cols), dtype=np.float32)
    out_b = np.empty((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            offset = (float(threshold[i, j]) - 0.5) * scale
            a_eff = pix_a[i, j] + err_a[i, j] + offset
            b_eff = pix_b[i, j] + err_b[i, j] + offset
            dists = (pix_L[i, j] - pal_L)**2 + (a_eff - pal_a)**2 + (b_eff - pal_b)**2
            k     = int(np.argmin(dists))
            C     = float(pix_C[i, j])
            a_out = C * float(np.cos(pal_H[k]))
            b_out = C * float(np.sin(pal_H[k]))
            out_a[i, j] = a_out
            out_b[i, j] = b_out
            ea = a_eff - a_out
            eb = b_eff - b_out
            if j + 1 < cols:
                err_a[i, j + 1]     += 7/16 * ea;  err_b[i, j + 1]     += 7/16 * eb
            if i + 1 < rows:
                if j > 0:
                    err_a[i+1, j-1] += 3/16 * ea;  err_b[i+1, j-1]     += 3/16 * eb
                err_a[i+1, j]       += 5/16 * ea;  err_b[i+1, j]       += 5/16 * eb
                if j + 1 < cols:
                    err_a[i+1, j+1] += 1/16 * ea;  err_b[i+1, j+1]     += 1/16 * eb

    return out_a, out_b




def _halfspace_eqs(points):
    """Half-space representation of the convex hull of 'points' (N, 3).

    Each row [nx, ny, nz, d] of the returned (F, 4) array satisfies
    nx*x + ny*y + nz*z + d <= 0 for all x inside the hull.
    O(N^4) — fine for small N (N = 18 palette entries).
    """
    pts = np.asarray(points, dtype=np.float64)
    N   = len(pts)
    centroid = pts.mean(axis=0)
    eqs = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                n  = np.cross(v1, v2)
                nrm = np.linalg.norm(n)
                if nrm < 1e-10:
                    continue
                n /= nrm
                d = float(-n @ pts[i])
                # Valid face: every point is on the non-positive side
                if not np.all(pts @ n + d <= 1e-8):
                    continue
                # Orient normal outward from centroid
                if float(n @ centroid) + d > 0:
                    n, d = -n, -d
                eqs.append(np.append(n, d).astype(np.float32))
    return np.array(eqs, dtype=np.float32) if eqs else np.zeros((0, 4), dtype=np.float32)


def _mix_strip(strip_lin, hull_eqs, space='linear', hull_eqs_km=None,
               max_iters=500, tol=1e-5):
    """Optimise out-of-hull pixels in the model's working space.

    strip_lin   : (M, 3) float32 — pixel values in the working space:
                    'linear' → linear RGB
                    'log'    → log(lin + LOG_EPS)
                    'km'     → log(1 + K/S)
    hull_eqs    : (F, 4) float32 — half-space equations in strip_lin's space.
    hull_eqs_km : (G, 4) float32 — (KM only) half-space equations in raw K/S space.
                    When supplied, each phase runs its first half in log(1+K/S)
                    (hull_eqs) then switches to raw K/S (hull_eqs_km) for the
                    second half, ending with strict KM mixing enforcement.
    Returns     : (M, 3) float32 — optimised values in strip_lin's space.
    """
    import mlx.core as mx
    M = strip_lin.shape[0]

    # ── In-hull test ──────────────────────────────────────────────────────────
    if hull_eqs.shape[0] > 0:
        in_hull = (strip_lin @ hull_eqs[:, :3].T + hull_eqs[:, 3] <= 1e-6).all(axis=-1)
    else:
        in_hull = np.ones(M, dtype=bool)

    out = strip_lin.copy()
    rem = np.where(~in_hull)[0]
    if len(rem) == 0:
        return out

    # ── Work only on out-of-hull pixels ───────────────────────────────────────
    M_r     = len(rem)
    normals = mx.array(hull_eqs[:, :3])
    d_vals  = mx.array(hull_eqs[:, 3])
    target  = mx.array(strip_lin[rem])
    color   = mx.array(strip_lin[rem])

    def _rgb_to_oklab(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        def cbrt(x):
            return mx.sign(x) * (mx.abs(x) + 1e-10) ** (1.0 / 3.0)
        l_, m_, s_ = cbrt(l), cbrt(m), cbrt(s)
        L  = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a  = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b_ = 0.0259040371 * l_ + 0.4072165126 * m_ - 0.4331205297 * s_
        return mx.stack([L, a, b_], axis=-1)

    def _oklab(c):
        if space == 'log':
            rgb = mx.clip(mx.exp(c) - LOG_EPS, 0.0, 1.0)
        elif space == 'km':
            ks  = mx.maximum(mx.exp(c) - 1.0, 0.0)
            rgb = mx.clip(1.0 / (1.0 + ks + mx.sqrt(ks * ks + 2.0 * ks + 1e-12)), 0.0, 1.0)
        else:
            rgb = c
        return _rgb_to_oklab(rgb)

    target_ok = _oklab(target)
    mx.eval(target_ok)

    L_target = target_ok[:, 0]
    C_target = mx.sqrt(target_ok[:, 1] ** 2 + target_ok[:, 2] ** 2 + 1e-16)
    a_hat_t  = target_ok[:, 1] / (C_target + 1e-8)
    b_hat_t  = target_ok[:, 2] / (C_target + 1e-8)
    mx.eval(L_target, C_target, a_hat_t, b_hat_t)

    def snap(c):
        for _ in range(20):
            v       = c @ normals.T + d_vals[None, :]
            per_px  = mx.max(v, axis=-1)
            worst_f = mx.argmax(v, axis=-1)
            c       = c - mx.maximum(per_px, 0.0)[:, None] * normals[worst_f]
        if space == 'km':
            c = mx.maximum(c, 0.0)  # log(1+K/S) ≥ 0
        return c

    # ── KM phased mode: raw-K/S snap + oklab ──────────────────────────────────
    km_phased = (space == 'km' and hull_eqs_km is not None)
    if km_phased:
        normals_km = mx.array(hull_eqs_km[:, :3])
        d_vals_km  = mx.array(hull_eqs_km[:, 3])

        def _oklab_km_raw(c):
            ks  = mx.maximum(c, 0.0)
            rgb = mx.clip(1.0 / (1.0 + ks + mx.sqrt(ks * ks + 2.0 * ks + 1e-12)), 0.0, 1.0)
            return _rgb_to_oklab(rgb)

        def snap_km_raw(c):
            for _ in range(20):
                v       = c @ normals_km.T + d_vals_km[None, :]
                per_px  = mx.max(v, axis=-1)
                worst_f = mx.argmax(v, axis=-1)
                c       = c - mx.maximum(per_px, 0.0)[:, None] * normals_km[worst_f]
            return mx.maximum(c, 0.0)  # K/S ≥ 0

    def _adam_run(loss_fn, c, snap_fn=None, n_iters=None):
        if snap_fn is None:
            snap_fn = snap
        if n_iters is None:
            n_iters = max_iters
        lag = mx.value_and_grad(loss_fn)
        lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
        m_a = mx.zeros((M_r, 3))
        v_a = mx.zeros((M_r, 3))
        prev = float('inf')
        for step in range(1, n_iters + 1):
            val, grad = lag(c)
            m_a = b1 * m_a + (1 - b1) * grad
            v_a = b2 * v_a + (1 - b2) * grad * grad
            c = snap_fn(c - lr * (m_a / (1 - b1 ** step)) / (mx.sqrt(v_a / (1 - b2 ** step)) + eps))
            mx.eval(c, m_a, v_a, val)
            curr = float(val)
            if abs(curr - prev) < tol:
                break
            prev = curr
        return c

    def _run_phase(loss_log, loss_km, c):
        """First half in log(1+K/S), second half in raw K/S, return log(1+K/S)."""
        half = max_iters // 2
        c = _adam_run(loss_log, c, snap_fn=snap, n_iters=half)
        c_km = snap_km_raw(mx.maximum(mx.exp(c) - 1.0, 0.0))
        mx.eval(c_km)
        c_km = _adam_run(loss_km, c_km, snap_fn=snap_km_raw, n_iters=max_iters - half)
        return mx.log(mx.maximum(c_km, 0.0) + 1.0)

    color = snap(color)
    mx.eval(color)

    # ── Phase 1: match lightness ──────────────────────────────────────────────
    if km_phased:
        color = _run_phase(
            lambda c: ((_oklab(c)        [:, 0] - L_target) ** 2).mean(),
            lambda c: ((_oklab_km_raw(c) [:, 0] - L_target) ** 2).mean(),
            color)
    else:
        color = _adam_run(lambda c: ((_oklab(c)[:, 0] - L_target) ** 2).mean(), color)

    # ── Phase 2: match hue, preserve L ───────────────────────────────────────
    def _loss_H(oklab_fn, c):
        lab   = oklab_fn(c)
        cross = lab[:, 1] * b_hat_t - lab[:, 2] * a_hat_t
        return (cross ** 2).mean() + 1000.0 * ((lab[:, 0] - L_target) ** 2).mean()

    if km_phased:
        color = _run_phase(lambda c: _loss_H(_oklab,        c),
                           lambda c: _loss_H(_oklab_km_raw, c), color)
    else:
        color = _adam_run(lambda c: _loss_H(_oklab, c), color)

    # ── Phase 3: match chroma, preserve H and L ───────────────────────────────
    def _loss_C(oklab_fn, c):
        lab   = oklab_fn(c)
        C_mix = mx.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2 + 1e-16)
        cross = lab[:, 1] * b_hat_t - lab[:, 2] * a_hat_t
        return (  (C_mix - C_target) ** 2).mean() \
             + 1000.0 * (cross ** 2).mean() \
             + 1000.0 * ((lab[:, 0] - L_target) ** 2).mean()

    if km_phased:
        color = _run_phase(lambda c: _loss_C(_oklab,        c),
                           lambda c: _loss_C(_oklab_km_raw, c), color)
    else:
        color = _adam_run(lambda c: _loss_C(_oklab, c), color)

    mx.eval(color)
    out[rem] = np.array(color)
    return out


def mix_convert(image, model='additive', strip_h=256):
    """Palette mixing gamut mapping.

    model='additive' — convex hull in linear RGB (light/additive mixing).
    model='pigment'  — convex hull in log(linear_RGB + LOG_EPS); convex
                       combinations = Beer's law transparent-pigment mixing:
                       T_mix = T_A^w * T_B^(1-w).
    model='opaque'   — convex hull in Kubelka-Munk K/S = (1-R)²/2R; convex
                       combinations = opaque paint mixing per K-M theory.

    Pixels inside the hull are left unchanged; others are remapped to the
    nearest in-hull colour by optimising lightness, hue, and chroma sequentially.
    """
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        print("Error: --mix requires MLX (pip install mlx)", file=sys.stderr)
        sys.exit(1)

    space = {'additive': 'linear', 'pigment': 'log', 'opaque': 'km'}[model]

    # Palette in linear RGB (R, G, B order)
    pal_bgr = np.array(PALETTE_BGR, dtype=np.float32) / 255.0  # (P, 3) sRGB
    pal_lin = np.stack([
        _srgb_to_linear(pal_bgr[:, 2]),
        _srgb_to_linear(pal_bgr[:, 1]),
        _srgb_to_linear(pal_bgr[:, 0]),
    ], axis=-1).astype(np.float32)                              # (P, 3) linear RGB

    # Build hull in the working space
    black = np.zeros((1, 3), dtype=np.float32)
    white = np.ones((1, 3),  dtype=np.float32)
    if space == 'log':
        pal_work = np.log(pal_lin + LOG_EPS)
        pal_ext  = np.vstack([pal_work,
                              np.log(black + LOG_EPS),
                              np.log(white + LOG_EPS)])
    elif space == 'km':
        pal_km    = _lin_to_km(pal_lin)
        pal_work  = np.log(1.0 + pal_km)
        black_km  = _lin_to_km(black)
        white_km  = _lin_to_km(white)
        # Log(1+K/S) hull — well-conditioned range [0, log(500)≈6.2].
        pal_ext   = np.vstack([pal_work,
                               np.log(1.0 + black_km),
                               np.log(1.0 + white_km)])
        # Raw K/S hull — used for the strict-KM second half of each phase.
        pal_ext_km = np.vstack([pal_km, black_km, white_km])
    else:
        pal_work = pal_lin
        pal_ext  = np.vstack([pal_lin, black, white])
    hull_eqs    = _halfspace_eqs(pal_ext)                       # (F, 4) — computed once
    hull_eqs_km = _halfspace_eqs(pal_ext_km) if space == 'km' else None

    # Image in linear RGB (R, G, B order)
    img_f   = image.astype(np.float32) / 255.0
    img_lin = np.stack([
        _srgb_to_linear(img_f[:, :, 2]),
        _srgb_to_linear(img_f[:, :, 1]),
        _srgb_to_linear(img_f[:, :, 0]),
    ], axis=-1).astype(np.float32)                              # (H, W, 3) linear RGB

    # Transform image to working space.
    # km: use log(1+K/S) for the optimiser variable (hull/snap use raw K/S).
    if space == 'log':
        img_work = np.log(img_lin + LOG_EPS)
    elif space == 'km':
        img_work = np.log(1.0 + _lin_to_km(img_lin))
    else:
        img_work = img_lin

    rows, cols = img_work.shape[:2]
    out_work = np.empty((rows, cols, 3), dtype=np.float32)

    print(f"  [mix] model={model} device=mlx", file=sys.stderr)
    for r0 in range(0, rows, strip_h):
        r1 = min(r0 + strip_h, rows)
        h  = r1 - r0
        print(f"  [mix] rows {r0}–{r1} / {rows}", file=sys.stderr, flush=True)
        strip  = img_work[r0:r1].reshape(-1, 3)
        result = _mix_strip(strip, hull_eqs, space=space, hull_eqs_km=hull_eqs_km)
        out_work[r0:r1] = result.reshape(h, cols, 3)

    # Convert working space back to linear RGB.
    if space == 'log':
        out_lin = np.clip(np.exp(out_work) - LOG_EPS, 0.0, 1.0)
    elif space == 'km':
        out_lin = _km_to_lin(np.maximum(np.exp(out_work) - 1.0, 0.0))
    else:
        out_lin = out_work

    # Linear RGB → sRGB → BGR uint8
    out_srgb = np.stack([
        _linear_to_srgb(out_lin[:, :, 0]),
        _linear_to_srgb(out_lin[:, :, 1]),
        _linear_to_srgb(out_lin[:, :, 2]),
    ], axis=-1)                                                  # (H, W, 3) sRGB RGB
    out_bgr = np.clip(out_srgb[:, :, ::-1], 0.0, 1.0)          # (H, W, 3) BGR
    return (out_bgr * 255.0).astype(np.uint8)


def _nighttime(image):
    """Darken and cool: L→1-√(1-L), b shifted toward blue inversely proportional to L."""
    lab = _bgr_to_oklab(image.astype(np.float32) / 255.0)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    L = np.clip(L, 0.0, 1.0)

    b_min, b_max = float(b.min()), float(b.max())
    if b_max - b_min > 1e-6:
        b_norm = (b - b_min) / (b_max - b_min)   # 0 = most blue, 1 = most yellow
        # dark pixels (low L) get full squaring; bright pixels (L→1) get no shift
        b_norm_new = b_norm * (L + b_norm * (1.0 - L))
        b_new = b_norm_new * (b_max - b_min) + b_min
    else:
        b_new = b

    out_lab = np.stack([1.0 - np.sqrt(1.0 - L), a, b_new], axis=-1)
    return np.clip(_oklab_to_bgr(out_lab) * 255.0, 0, 255).astype(np.uint8)


def convert(image, palette, dither=None):
    """Map every pixel to the nearest Nord colour by 3D Oklab distance,
    then snap the hue while keeping the pixel's own chroma and lightness."""
    lab = _bgr_to_oklab(image.astype(np.float32) / 255.0)
    pix_L = lab[:, :, 0]
    pix_a = lab[:, :, 1]
    pix_b = lab[:, :, 2]
    pix_C = np.sqrt(pix_a**2 + pix_b**2)

    pal_L = np.array([c['L'] for c in palette], dtype=np.float32)
    pal_a = np.array([c['a'] for c in palette], dtype=np.float32)
    pal_b = np.array([c['b'] for c in palette], dtype=np.float32)
    pal_H = np.array([c['H'] for c in palette], dtype=np.float32)

    if dither == "fs":
        rows, cols = pix_L.shape
        threshold = _threshold_texture(rows, cols)
        out_a, out_b = _dither(pix_L, pix_a, pix_b, pix_C, pal_L, pal_a, pal_b, pal_H, threshold)
    else:
        dL = pix_L[:, :, None] - pal_L
        da = pix_a[:, :, None] - pal_a
        db = pix_b[:, :, None] - pal_b
        nearest   = np.argmin(dL**2 + da**2 + db**2, axis=-1)
        matched_H = pal_H[nearest]
        out_a = pix_C * np.cos(matched_H)
        out_b = pix_C * np.sin(matched_H)

    out_lab = np.stack([pix_L, out_a, out_b], axis=-1)
    return (_oklab_to_bgr(out_lab) * 255.0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Convert an image to the Nord colour palette via Oklab snapping."
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", required=True, help="Output image path")
    parser.add_argument("--dither", choices=["fs"],
                        help="Dithering: 'fs' (Floyd-Steinberg with blue noise)")
    parser.add_argument("--mix", nargs="?", const="additive",
                        choices=["additive", "pigment", "opaque"],
                        help="Palette mixing model: 'additive' (linear RGB hull, default), "
                             "'pigment' (log-space hull, Beer's law transparent-pigment mixing), "
                             "or 'opaque' (Kubelka-Munk K/S hull, opaque paint mixing). "
                             "Ignores --dither.")
    parser.add_argument("--night", action="store_true",
                        help="Nighttime preprocessing: darken (L→L²) and cool (b shifted toward blue)")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    if args.night:
        image = _nighttime(image)

    if args.mix is not None:
        result = mix_convert(image, model=args.mix)
    else:
        palette = build_lookup()
        result = convert(image, palette, dither=args.dither)

    ok = cv2.imwrite(args.output, result)
    if not ok:
        print(f"Error: cannot write image '{args.output}'", file=sys.stderr)
        sys.exit(1)

    print(f"Saved '{args.output}'")


if __name__ == "__main__":
    main()
