#!/usr/bin/env python3
"""Convert an image to the Nord colour palette via Oklab snapping."""

import argparse
import sys

import cv2
import numpy as np

KM_EPS  = 1e-3   # minimum reflectance before K/S conversion; K/S(KM_EPS) ≈ 499


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

# ── CIE 1931 2° standard observer and D65 illuminant (400–700 nm, 10 nm) ────
_LAMBDA = np.arange(400, 710, 10, dtype=np.float32)  # (31,)

_CIE_CMF = np.array([
    #    x̄          ȳ          z̄
    [0.014310, 0.000396, 0.067850],
    [0.043510, 0.001210, 0.207300],
    [0.134380, 0.004000, 0.645600],
    [0.283900, 0.011600, 1.385800],
    [0.348280, 0.023000, 1.747060],
    [0.336200, 0.038000, 1.772110],
    [0.290800, 0.060000, 1.669200],
    [0.195360, 0.090980, 1.287640],
    [0.095640, 0.139020, 0.812950],
    [0.032010, 0.208020, 0.465180],
    [0.004900, 0.323000, 0.272000],
    [0.009650, 0.503000, 0.158200],
    [0.063100, 0.710000, 0.078200],
    [0.165500, 0.862000, 0.042200],
    [0.290400, 0.954000, 0.020300],
    [0.433450, 0.994950, 0.008750],
    [0.594500, 0.995000, 0.003900],
    [0.762100, 0.952000, 0.002100],
    [0.916300, 0.870000, 0.001650],
    [1.026300, 0.757000, 0.001100],
    [1.062200, 0.631000, 0.000800],
    [1.002600, 0.503000, 0.000340],
    [0.854450, 0.381000, 0.000190],
    [0.642400, 0.265000, 0.000050],
    [0.447900, 0.175000, 0.000020],
    [0.283500, 0.107000, 0.000000],
    [0.164900, 0.061000, 0.000000],
    [0.087400, 0.032000, 0.000000],
    [0.046770, 0.017000, 0.000000],
    [0.022700, 0.008200, 0.000000],
    [0.011350, 0.004100, 0.000000],
], dtype=np.float32)  # (31, 3)

_D65 = np.array([
     82.7549,  91.4860,  93.4318,  86.6823, 104.8650, 117.0080,
    117.8120, 114.8610, 115.9230, 108.8110, 109.3540, 107.8020,
    104.7900, 107.6890, 104.4050, 104.0460, 100.0000,  96.3342,
     95.7880,  88.6856,  90.0062,  89.5991,  87.6987,  83.2886,
     83.6992,  80.0268,  80.2146,  82.2778,  78.2842,  69.7213,
     71.6091,
], dtype=np.float32)  # (31,)

# IEC 61966-2-1 sRGB ↔ CIE XYZ (D65) matrices; convention: XYZ = lin_rgb @ M.T
_M_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float32)

_M_XYZ_TO_RGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float32)

_PALETTE_KS: np.ndarray | None = None  # (N_palette, 31) K/S spectra, fitted on first use


def _fit_palette_ks():
    """Fit a reflectance model to each palette colour.

    Spectral colours use a single Gaussian:
      R(λ) = clip(R_base + A·exp(−(λ−λ₀)²/2σ²), KM_EPS, 1)

    Extra-spectral colours (purple/magenta: g < r AND g < b) use a bi-Gaussian
    with one red lobe (λ ∈ [560, 700] nm) and one blue/violet lobe (λ ∈ [400, 490] nm).
    A single Gaussian cannot simultaneously stimulate the L and S cones while
    leaving the M cone depressed; the bi-Gaussian is the physically correct model.

    Returns (N_palette, 31) float32 K/S ratios.
    """
    lam    = _LAMBDA.astype(np.float64)
    k_norm = float((_D65 * _CIE_CMF[:, 1]).sum())
    D65n   = _D65.astype(np.float64) / k_norm       # normalised so Y(white) = 1
    cmf    = _CIE_CMF.astype(np.float64)

    def _xyz_from_refl(R):
        return (R[:, None] * D65n[:, None] * cmf).sum(axis=0)

    try:
        from scipy.optimize import minimize as _sp_min
        _have_scipy = True
    except ImportError:
        _have_scipy = False

    pal_ks = np.zeros((len(PALETTE_BGR), 31), dtype=np.float32)

    for i, (b8, g8, r8) in enumerate(PALETTE_BGR):
        r_lin = float(_srgb_to_linear(r8 / 255.0))
        g_lin = float(_srgb_to_linear(g8 / 255.0))
        b_lin = float(_srgb_to_linear(b8 / 255.0))
        xyz_tgt = np.array([r_lin, g_lin, b_lin]) @ _M_RGB_TO_XYZ.T.astype(np.float64)

        # Purple/magenta: green is the minimum channel → extra-spectral.
        # Needs one red lobe + one blue/violet lobe.
        is_extraspectral = (g_lin < r_lin) and (g_lin < b_lin)

        if is_extraspectral:
            def loss(p, xyz_tgt=xyz_tgt):
                Rb, Ar, lr_, sgr, Ab, lb_, sgb = p
                R = np.clip(
                    Rb
                    + Ar * np.exp(-(lam - lr_)**2 / (2.0 * sgr**2))
                    + Ab * np.exp(-(lam - lb_)**2 / (2.0 * sgb**2)),
                    KM_EPS, 1.0)
                return float(((_xyz_from_refl(R) - xyz_tgt)**2).sum())

            brightness = (r_lin + g_lin + b_lin) / 3.0
            x0   = [max(0.02, brightness * 0.1),
                    max(0.02, r_lin * 0.9), 625.0, 40.0,
                    max(0.02, b_lin * 0.9), 445.0, 40.0]
            bnds = [(0.0, 0.99),
                    (0.0, 0.99), (560.0, 700.0), (20.0, 100.0),
                    (0.0, 0.99), (400.0, 490.0), (20.0, 100.0)]
        else:
            def loss(p, xyz_tgt=xyz_tgt):
                Rb, A, l0, sg = p
                R = np.clip(Rb + A * np.exp(-(lam - l0)**2 / (2.0 * sg**2)), KM_EPS, 1.0)
                return float(((_xyz_from_refl(R) - xyz_tgt)**2).sum())

            if r_lin >= g_lin and r_lin >= b_lin:
                l0_init = 620.0
            elif g_lin >= r_lin and g_lin >= b_lin:
                l0_init = 540.0
            else:
                l0_init = 460.0
            brightness = (r_lin + g_lin + b_lin) / 3.0
            x0   = [max(0.02, brightness * 0.2), max(0.02, brightness * 0.8), l0_init, 60.0]
            bnds = [(0.0, 0.99), (0.0, 0.99), (400.0, 700.0), (20.0, 150.0)]

        if _have_scipy:
            res    = _sp_min(loss, x0, method='L-BFGS-B', bounds=bnds,
                             options={'maxiter': 300, 'ftol': 1e-14})
            p_best = res.x
        else:
            rng    = np.random.default_rng(i)
            best_l = loss(x0)
            p_best = np.array(x0)
            for _ in range(500):
                x = np.array([rng.uniform(lo, hi) for lo, hi in bnds])
                lv = loss(x)
                if lv < best_l:
                    best_l, p_best = lv, x.copy()
            for _ in range(100):
                for j in range(len(bnds)):
                    for frac in [0.05, 0.005]:
                        for s in (1, -1):
                            lo, hi = bnds[j]
                            xn = p_best.copy()
                            xn[j] = np.clip(xn[j] + s * frac * (hi - lo), lo, hi)
                            lv = loss(xn)
                            if lv < best_l:
                                best_l, p_best = lv, xn

        if is_extraspectral:
            Rb, Ar, lr_, sgr, Ab, lb_, sgb = p_best
            R_fit = np.clip(
                Rb
                + Ar * np.exp(-(lam - lr_)**2 / (2.0 * sgr**2))
                + Ab * np.exp(-(lam - lb_)**2 / (2.0 * sgb**2)),
                KM_EPS, 1.0)
        else:
            Rb, A, l0, sg = p_best
            R_fit = np.clip(Rb + A * np.exp(-(lam - l0)**2 / (2.0 * sg**2)), KM_EPS, 1.0)

        pal_ks[i] = ((1.0 - R_fit)**2 / (2.0 * R_fit)).astype(np.float32)

    return pal_ks


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




def _simplex_project_mlx(c):
    """Project rows of (M, N) MLX array onto the probability simplex (Duchi et al., 2008)."""
    import mlx.core as mx
    N  = c.shape[1]
    u  = -mx.sort(-c, axis=-1)                                     # (M, N) descending
    cs = mx.cumsum(u, axis=-1)                                     # (M, N)
    j  = mx.arange(1, N + 1, dtype=mx.float32)[None, :]           # (1, N)
    rho = mx.sum((u * j > cs - 1.0).astype(mx.float32),
                 axis=-1, keepdims=True)                           # (M, 1)
    rho_idx = mx.maximum(rho.astype(mx.int32) - 1, 0)             # (M, 1)
    oh  = (mx.arange(N, dtype=mx.int32)[None, :] ==
           rho_idx).astype(mx.float32)                             # (M, N)
    theta = ((oh * cs).sum(axis=-1, keepdims=True) - 1.0) / rho   # (M, 1)
    return mx.maximum(c - theta, 0.0)


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


def _mix_strip_spectral(strip_bgr, palette_ks_mx, D65n_cmf_mx, M_xyz2rgb_mx,
                        c_anchor=None, reg_lambda=0.05,
                        max_iters=500, tol=1e-5, optimizer='adam', adam_lr=0.02,
                        lr_final_frac=0.05, progress_label=None):
    """KM spectral mixing on a flat (M, 3) BGR sRGB strip.

    Optimises simplex weights c over palette K/S spectra so that the mixed
    colour (integrated through CIE observer) minimises Oklab distance to the
    target.  Returns (M, N_pal) float32 simplex weights.

    c_anchor   : (M, N_pal) float32 — spatially-smoothed initialisation.
                 When provided, c starts here and a regularisation term
                 reg_lambda * ||c − c_anchor||² is added to the loss.
                 In flat regions the oklab loss ≈ 0, so regularisation
                 dominates and keeps nearby pixels near their shared
                 smooth anchor → no contouring.  In sharp regions the
                 oklab gradient dominates → fine detail is preserved.
    reg_lambda : regularisation strength.
    """
    import mlx.core as mx
    import time
    import math

    M, N_pal = strip_bgr.shape[0], palette_ks_mx.shape[0]

    # Linear RGB (R, G, B) of input strip
    strip_lin = np.stack([
        _srgb_to_linear(strip_bgr[:, 2]),
        _srgb_to_linear(strip_bgr[:, 1]),
        _srgb_to_linear(strip_bgr[:, 0]),
    ], axis=-1).astype(np.float32)                                # (M, 3) R,G,B

    # Target Oklab
    tgt_ok = np.stack(
        _linear_rgb_to_oklab(strip_lin[:, 0], strip_lin[:, 1], strip_lin[:, 2]),
        axis=-1).astype(np.float32)                               # (M, 3)
    target = mx.array(tgt_ok)

    # Initialisation: use supplied anchor if available, else one-hot nearest palette
    if c_anchor is not None:
        c = mx.array(c_anchor.astype(np.float32))
        anchor_mx = mx.array(c_anchor.astype(np.float32))
    else:
        pal_bgr_f = np.array(PALETTE_BGR, dtype=np.float32) / 255.0
        pal_lin = np.stack([
            _srgb_to_linear(pal_bgr_f[:, 2]),
            _srgb_to_linear(pal_bgr_f[:, 1]),
            _srgb_to_linear(pal_bgr_f[:, 0]),
        ], axis=-1)
        pal_ok = np.stack(
            _linear_rgb_to_oklab(pal_lin[:, 0], pal_lin[:, 1], pal_lin[:, 2]),
            axis=-1)
        dists   = ((tgt_ok[:, None, :] - pal_ok[None, :, :]) ** 2).sum(axis=-1)
        nearest = np.argmin(dists, axis=-1)
        c_init  = np.zeros((M, N_pal), dtype=np.float32)
        c_init[np.arange(M), nearest] = 1.0
        c = mx.array(c_init)
        anchor_mx = None

    def _forward(c):
        ks_mix = mx.maximum(c @ palette_ks_mx, 0.0)              # (M, 31)
        R_mix  = 1.0 / (1.0 + ks_mix + mx.sqrt(ks_mix * ks_mix + 2.0 * ks_mix + 1e-12))
        XYZ    = R_mix @ D65n_cmf_mx                              # (M, 3)
        rgb    = mx.clip(XYZ @ M_xyz2rgb_mx.T, 0.0, 1.0)         # (M, 3) linear R,G,B
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
        m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
        s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
        def cbrt(x): return (x + 1e-8)**(1.0/3.0)
        l_, m_, s_ = cbrt(l), cbrt(m), cbrt(s)
        L  = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
        a  = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
        b_ = 0.0259040371*l_ + 0.4072165126*m_ - 0.4331205297*s_
        return mx.stack([L, a, b_], axis=-1)

    def _reg(c):
        return reg_lambda * ((c - anchor_mx) ** 2).mean() if anchor_mx is not None else 0.0

    def loss_single(c):
        pred = _forward(c)
        return ((target - pred)**2).mean() + _reg(c)

    def _adam_run(loss_fn, c, n_steps, phase_label):
        lr0, b1, b2, eps_a = adam_lr, 0.9, 0.999, 1e-8
        lr_min = adam_lr * lr_final_frac
        denom  = max(n_steps - 1, 1)
        m_a    = mx.zeros((M, N_pal))
        v_a    = mx.zeros((M, N_pal))
        lag    = mx.value_and_grad(loss_fn)
        prev   = float('inf')
        t0     = time.time()
        for step in range(1, n_steps + 1):
            lr = lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * (step - 1) / denom))
            val, grad = lag(c)
            m_a   = b1 * m_a + (1.0 - b1) * grad
            v_a   = b2 * v_a + (1.0 - b2) * grad * grad
            c     = _simplex_project_mlx(c - lr * (m_a / (1.0 - b1**step)) / (mx.sqrt(v_a / (1.0 - b2**step)) + eps_a))
            mx.eval(c, m_a, v_a, val)
            curr  = float(val)
            if progress_label is not None and (step % 10 == 0 or step == n_steps):
                ips = step / (time.time() - t0 + 1e-9)
                print(f"\r  [mix] {progress_label} {phase_label} step {step:4d}/{n_steps} "
                      f"loss={curr:.6f} ({ips:5.1f} it/s)   ",
                      end="", file=sys.stderr, flush=True)
            if abs(curr - prev) < tol:
                break
            prev = curr
        if progress_label is not None:
            print(file=sys.stderr, flush=True)
        return c

    if optimizer == 'gd':
        lag = mx.value_and_grad(lambda c: ((target - _forward(c))**2).mean() + _reg(c))
        prev = float('inf')
        lr   = 0.05
        for step in range(1, max_iters + 1):
            val, grad = lag(c)
            g_norm  = mx.sqrt((grad * grad).sum(axis=-1, keepdims=True) + 1e-12)
            clipped = grad * mx.minimum(1.0 / g_norm, 1.0)
            c = _simplex_project_mlx(c - lr * clipped)
            mx.eval(c, val)
            if abs(float(val) - prev) < tol:
                break
            prev = float(val)
    else:
        c = _adam_run(loss_single, c, max_iters, '')

    return np.array(c)   # (M, N_pal) float32 simplex weights


def _simplex_project_np(c):
    """Project rows of (M, N) array onto the probability simplex."""
    c  = c.astype(np.float64)
    u  = np.sort(c, axis=-1)[:, ::-1]
    cs = np.cumsum(u, axis=-1)
    N  = c.shape[1]
    rho = np.maximum(np.sum(u * np.arange(1, N + 1) > cs - 1.0, axis=-1), 1)
    theta = (cs[np.arange(len(c)), rho - 1] - 1.0) / rho
    return np.maximum(c - theta[:, None], 0.0).astype(np.float32)


def _recon_from_weights(weights, palette_ks, D65n_cmf):
    """(M, N_pal) simplex weights → (M, 3) float32 BGR sRGB in [0, 1]."""
    ks_f  = np.maximum(weights @ palette_ks, 0.0)
    R_f   = _km_to_lin(ks_f)
    XYZ_f = R_f @ D65n_cmf
    rgb_f = np.clip(XYZ_f @ _M_XYZ_TO_RGB.T, 0.0, 1.0)
    return np.stack([
        np.clip(_linear_to_srgb(rgb_f[:, 2]), 0.0, 1.0),  # B
        np.clip(_linear_to_srgb(rgb_f[:, 1]), 0.0, 1.0),  # G
        np.clip(_linear_to_srgb(rgb_f[:, 0]), 0.0, 1.0),  # R
    ], axis=-1).astype(np.float32)


def mix_convert_spectral(image, strip_h=256,
                         init_sigma=14.0, n_phases=1,
                         steps_per_phase=200, phase_sigma=20.0,
                         phase_shrink=0.0, final_lr=0.01):
    """Gaussian-reflectance Kubelka-Munk spectral mixing.

    Fits a Gaussian reflectance spectrum to each palette colour, then for every
    pixel optimises a simplex (Σcᵢ = 1) over palette K/S spectra to minimise
    Oklab distance to the target.

    Pipeline:
      1. Build augmented palette: original N colours + all N*(N-1)/2 pairwise
         50/50 K/S mixtures.  Each augmented entry carries a source vector in
         the original palette space (one-hot for pure colours, 0.5/0.5 for
         pairs).
      2. Snap each pixel to the nearest augmented-palette entry → source
         vector in original-palette space (no dithering needed: boundary
         pixels land on pairwise-mixture entries directly).
      3. Blur (σ=init_sigma) → simplex snap.
      4. Repeat n_phases times:
           a. Adam lr=0.02 (all but last), lr=0.001 (last) for steps_per_phase
              steps per strip (no early exit).  The reduced lr on the final
              phase limits divergence between adjacent blurred-region pixels.
           b. Between phases (skipped after last): blur σ=phase_sigma + snap.
    """
    try:
        import mlx.core as mx
    except ImportError:
        print("Error: --mix spectral requires MLX (pip install mlx)", file=sys.stderr)
        sys.exit(1)

    global _PALETTE_KS
    if _PALETTE_KS is None:
        print("  Fitting spectral K/S for palette...", file=sys.stderr, flush=True)
        _PALETTE_KS = _fit_palette_ks()

    k_norm   = float((_D65 * _CIE_CMF[:, 1]).sum())
    D65n_cmf = (_D65[:, None] / k_norm * _CIE_CMF).astype(np.float32)  # (31, 3)

    pal_ks_mx    = mx.array(_PALETTE_KS)
    D65n_cmf_mx  = mx.array(D65n_cmf)
    M_xyz2rgb_mx = mx.array(_M_XYZ_TO_RGB)

    img_f = image.astype(np.float32) / 255.0
    rows, cols = img_f.shape[:2]
    N_pal = _PALETTE_KS.shape[0]

    # ── Build augmented palette: pure colours + pairwise 50/50 K/S mixtures ──
    print("  [mix] building augmented palette + init...", file=sys.stderr, flush=True)
    img_ok = _bgr_to_oklab(img_f)                                    # (H, W, 3)
    pal_bgr_f = np.array(PALETTE_BGR, dtype=np.float32) / 255.0
    pal_lin = np.stack([
        _srgb_to_linear(pal_bgr_f[:, 2]),
        _srgb_to_linear(pal_bgr_f[:, 1]),
        _srgb_to_linear(pal_bgr_f[:, 0]),
    ], axis=-1)
    pal_ok = np.stack(
        _linear_rgb_to_oklab(pal_lin[:, 0], pal_lin[:, 1], pal_lin[:, 2]),
        axis=-1)                                                      # (N_pal, 3)

    # Pairwise 50/50 K/S mixtures
    pi, pj = np.triu_indices(N_pal, k=1)                             # (N_pairs,) each
    ks_pairs  = 0.5 * (_PALETTE_KS[pi] + _PALETTE_KS[pj])           # (N_pairs, 31)
    R_pairs   = _km_to_lin(ks_pairs)                                 # (N_pairs, 31)
    XYZ_pairs = R_pairs @ D65n_cmf                                   # (N_pairs, 3)
    rgb_pairs = np.clip(XYZ_pairs @ _M_XYZ_TO_RGB.T, 0.0, 1.0)     # (N_pairs, 3)
    ok_pairs  = np.stack(
        _linear_rgb_to_oklab(rgb_pairs[:, 0], rgb_pairs[:, 1], rgb_pairs[:, 2]),
        axis=-1)                                                      # (N_pairs, 3)

    # Source vectors: pure colours → one-hot; pairs → 0.5 at each component
    N_pairs = pi.shape[0]
    src_pure  = np.eye(N_pal, dtype=np.float32)                      # (N_pal, N_pal)
    src_pairs = np.zeros((N_pairs, N_pal), dtype=np.float32)
    src_pairs[np.arange(N_pairs), pi] = 0.5
    src_pairs[np.arange(N_pairs), pj] = 0.5

    aug_ok  = np.concatenate([pal_ok,  ok_pairs],  axis=0)           # (N_aug, 3)
    aug_src = np.concatenate([src_pure, src_pairs], axis=0)          # (N_aug, N_pal)

    # ── Snap to nearest augmented-palette entry, blur, snap ──────────────────
    # No dithering needed: boundary pixels are directly assigned to the
    # pairwise-mixture entry, giving interior-simplex starting weights.
    init = np.empty((rows, cols, N_pal), dtype=np.float32)
    DSTRIP = 64
    for r0 in range(0, rows, DSTRIP):
        r1   = min(r0 + DSTRIP, rows)
        dist = (img_ok[r0:r1, :, 0, None] - aug_ok[None, None, :, 0]) ** 2
        dist += (img_ok[r0:r1, :, 1, None] - aug_ok[None, None, :, 1]) ** 2
        dist += (img_ok[r0:r1, :, 2, None] - aug_ok[None, None, :, 2]) ** 2
        init[r0:r1] = aug_src[np.argmin(dist, axis=-1)]

    # Blend snapped init 50-50 with a field of random palette mixtures so pixels
    # do not start the optimisation already stuck in local optima near palette edges.
    rand_w = np.random.exponential(1.0, (rows, cols, N_pal)).astype(np.float32)
    rand_w /= rand_w.sum(axis=-1, keepdims=True)
    init = 0.5 * init + 0.5 * rand_w
    init = _simplex_project_np(init.reshape(-1, N_pal)).reshape(rows, cols, N_pal)

    if init_sigma > 0:
        for k in range(N_pal):
            init[:, :, k] = cv2.GaussianBlur(init[:, :, k], (0, 0), init_sigma)
        init = _simplex_project_np(init.reshape(-1, N_pal)).reshape(rows, cols, N_pal)

    # ── Phased Adam: optimize → blur → optimize → blur → … ───────────────────
    print("  [mix] model=spectral device=mlx", file=sys.stderr)
    weights_hw = init.copy()
    n_strips = (rows + strip_h - 1) // strip_h
    for phase in range(n_phases):
        print(f"  [mix] phase {phase + 1}/{n_phases}", file=sys.stderr, flush=True)
        for si, r0 in enumerate(range(0, rows, strip_h)):
            r1 = min(r0 + strip_h, rows)
            h  = r1 - r0
            strip = img_f[r0:r1].reshape(-1, 3)
            lr = final_lr if phase == n_phases - 1 else 0.02
            w = _mix_strip_spectral(strip, pal_ks_mx, D65n_cmf_mx, M_xyz2rgb_mx,
                                    c_anchor=weights_hw[r0:r1].reshape(-1, N_pal),
                                    reg_lambda=0.0,
                                    max_iters=steps_per_phase, tol=0.0,
                                    adam_lr=lr,
                                    progress_label=f"strip {si + 1}/{n_strips}")
            weights_hw[r0:r1] = w.reshape(h, cols, N_pal)
        if phase < n_phases - 1:
            for k in range(N_pal):
                weights_hw[:, :, k] = cv2.GaussianBlur(
                    weights_hw[:, :, k], (0, 0), phase_sigma)
            weights_hw = _simplex_project_np(
                weights_hw.reshape(-1, N_pal)).reshape(rows, cols, N_pal)
            weights_hw = (1.0 - phase_shrink) * weights_hw + phase_shrink / N_pal

    flat = _simplex_project_np(weights_hw.reshape(-1, N_pal))
    bgr  = _recon_from_weights(flat, _PALETTE_KS, D65n_cmf)
    return np.clip(bgr.reshape(rows, cols, 3) * 255.0, 0, 255).astype(np.uint8)


def _mix_strip_additive(strip_lin, hull_eqs, max_iters=300, lr=0.02, tol=1e-6,
                        progress_label=None):
    """Projected gradient descent in linear RGB, minimising Oklab distance to target.

    Pixels are first projected onto the palette's convex hull in linear RGB
    (a no-op for pixels already inside it — only out-of-hull pixels are
    processed). Optimisation then proceeds in two sequential phases —
    luminance, then chrominance (a, b jointly) — each running up to
    max_iters steps. Every step computes the analytic gradient of that
    phase's loss, proposes color - lr*grad, then projects that candidate
    back onto the hull via the half-space representation — so the
    effective, gamut-clamped step is (projected_candidate - color) rather
    than the raw gradient step.

    strip_lin : (M, 3) float32 — pixel values in linear RGB (R, G, B).
    hull_eqs  : (F, 4) float32 — half-space equations in linear RGB space.
    Returns   : (M, 3) float32 — optimised linear RGB values.
    """
    import mlx.core as mx

    M = strip_lin.shape[0]

    if hull_eqs.shape[0] > 0:
        in_hull = (strip_lin @ hull_eqs[:, :3].T + hull_eqs[:, 3] <= 1e-6).all(axis=-1)
    else:
        in_hull = np.ones(M, dtype=bool)

    out = strip_lin.copy()
    rem = np.where(~in_hull)[0]
    if len(rem) == 0:
        return out

    normals = mx.array(hull_eqs[:, :3])
    d_vals  = mx.array(hull_eqs[:, 3])

    def _project(c):
        for _ in range(20):
            v      = c @ normals.T + d_vals[None, :]
            per_px = mx.max(v, axis=-1)
            worst  = mx.argmax(v, axis=-1)
            c      = c - mx.maximum(per_px, 0.0)[:, None] * normals[worst]
        return mx.clip(c, 0.0, 1.0)

    def _oklab(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        def cbrt(x):
            return (x + 1e-8) ** (1.0 / 3.0)
        l_, m_, s_ = cbrt(l), cbrt(m), cbrt(s)
        L  = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a  = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b_ = 0.0259040371 * l_ + 0.4072165126 * m_ - 0.4331205297 * s_
        return mx.stack([L, a, b_], axis=-1)

    target    = mx.array(strip_lin[rem])
    target_ok = _oklab(target)
    color     = _project(target)
    mx.eval(target_ok, color)
    baseline_color = color   # naive hull-clamp, before any gradient descent

    L_target = target_ok[:, 0]
    a_target = target_ok[:, 1]
    b_target = target_ok[:, 2]
    mx.eval(L_target, a_target, b_target)

    def _run_phase(loss_fn, c, phase_label):
        """Adam, projecting the candidate onto the hull after every step —
        so the effective, gamut-clamped step is (projected_candidate - c)
        rather than the raw Adam update. Adam's per-coordinate adaptivity is
        needed here: phase 2 combines a primary term with a 1000x penalty
        term, and that ill-conditioning stalls plain gradient descent.

        Gradients are norm-clipped before feeding Adam's moment estimates.
        Near black, Oklab's cube root has a very steep (though finite) slope,
        so a pixel whose trajectory passes close to (0,0,0) can produce one
        enormous gradient. Left unclipped, that single step dominates the
        exponential moving averages — especially the slowly-decaying second
        moment (beta2=0.999) — for dozens of subsequent steps, during which
        Adam keeps moving in the wrong direction."""
        lag       = mx.value_and_grad(loss_fn)
        b1, b2, eps_a = 0.9, 0.999, 1e-8
        max_grad_norm = 20.0
        m_a       = mx.zeros(c.shape)
        v_a       = mx.zeros(c.shape)
        prev      = float('inf')
        for step in range(1, max_iters + 1):
            val, grad = lag(c)
            grad_norm = mx.sqrt((grad * grad).sum(axis=-1, keepdims=True) + 1e-12)
            grad = grad * mx.minimum(1.0, max_grad_norm / grad_norm)
            m_a = b1 * m_a + (1.0 - b1) * grad
            v_a = b2 * v_a + (1.0 - b2) * grad * grad
            update = (m_a / (1.0 - b1 ** step)) / (mx.sqrt(v_a / (1.0 - b2 ** step)) + eps_a)
            c      = _project(c - lr * update)
            mx.eval(c, m_a, v_a, val)
            curr = float(val)
            if progress_label is not None and (step % 10 == 0 or step == max_iters):
                print(f"\r  [mix] {progress_label} {phase_label} step {step:4d}/{max_iters} "
                      f"loss={curr:.6f}   ", end="", file=sys.stderr, flush=True)
            if abs(curr - prev) < tol:
                break
            prev = curr
        if progress_label is not None:
            print(file=sys.stderr, flush=True)
        return c

    # Phase 1: match luminance
    color = _run_phase(lambda c: ((_oklab(c)[:, 0] - L_target) ** 2).mean(), color, "L")

    # Phase 2: match chrominance (a, b jointly — hue and chroma together), preserve luminance
    def _loss_ab(c):
        lab = _oklab(c)
        return ((lab[:, 1] - a_target) ** 2 + (lab[:, 2] - b_target) ** 2).mean() \
             + 1000.0 * ((lab[:, 0] - L_target) ** 2).mean()
    color = _run_phase(_loss_ab, color, "ab")

    # Safety net: the two-phase optimisation should only ever improve on the
    # naive hull-clamp, but near black Oklab's cube root is steep enough that
    # occasional pixels can diverge instead (see module docs / BACKGROUND.md).
    # Never let optimisation leave a pixel worse off than simply clamping it.
    optimised_dist = ((_oklab(color) - target_ok) ** 2).sum(axis=-1)
    baseline_dist  = ((_oklab(baseline_color) - target_ok) ** 2).sum(axis=-1)
    color = mx.where((optimised_dist > baseline_dist)[:, None], baseline_color, color)
    mx.eval(color)

    out[rem] = np.array(color)
    return out


def mix_convert_additive(image, strip_h=256, max_iters=300, lr=0.02):
    """Additive (linear-light) palette mixing: convex hull in linear RGB.

    Pixels are clamped onto the palette's convex hull in linear RGB, then
    walked back toward their original colour in Oklab space via projected
    gradient descent (see _mix_strip_additive). Pixels already inside the
    hull are left unchanged.
    """
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        print("Error: --mix additive requires MLX (pip install mlx)", file=sys.stderr)
        sys.exit(1)

    pal_bgr = np.array(PALETTE_BGR, dtype=np.float32) / 255.0  # (P, 3) sRGB
    pal_lin = np.stack([
        _srgb_to_linear(pal_bgr[:, 2]),
        _srgb_to_linear(pal_bgr[:, 1]),
        _srgb_to_linear(pal_bgr[:, 0]),
    ], axis=-1).astype(np.float32)                              # (P, 3) linear RGB

    black = np.zeros((1, 3), dtype=np.float32)
    white = np.ones((1, 3),  dtype=np.float32)
    pal_ext  = np.vstack([pal_lin, black, white])
    hull_eqs = _halfspace_eqs(pal_ext)                          # (F, 4) — computed once

    img_f   = image.astype(np.float32) / 255.0
    rows, cols = img_f.shape[:2]
    img_lin = np.stack([
        _srgb_to_linear(img_f[:, :, 2]),
        _srgb_to_linear(img_f[:, :, 1]),
        _srgb_to_linear(img_f[:, :, 0]),
    ], axis=-1).astype(np.float32)                               # (H, W, 3) R,G,B

    print("  [mix] model=additive device=mlx", file=sys.stderr)
    out_lin  = img_lin.copy()
    n_strips = (rows + strip_h - 1) // strip_h
    for si, r0 in enumerate(range(0, rows, strip_h)):
        r1    = min(r0 + strip_h, rows)
        strip = img_lin[r0:r1].reshape(-1, 3)
        out   = _mix_strip_additive(strip, hull_eqs, max_iters=max_iters, lr=lr,
                                    progress_label=f"strip {si + 1}/{n_strips}")
        out_lin[r0:r1] = out.reshape(r1 - r0, cols, 3)

    bgr = np.stack([
        np.clip(_linear_to_srgb(out_lin[:, :, 2]), 0.0, 1.0),  # B
        np.clip(_linear_to_srgb(out_lin[:, :, 1]), 0.0, 1.0),  # G
        np.clip(_linear_to_srgb(out_lin[:, :, 0]), 0.0, 1.0),  # R
    ], axis=-1)
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)


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
    parser.add_argument("--mix", nargs="?", const="spectral", choices=["spectral", "additive"],
                        help="Palette mixing (requires MLX; ignores --dither). "
                             "'spectral' (default): optimise simplex weights over "
                             "Gaussian-reflectance Kubelka-Munk palette spectra. "
                             "'additive': clamp to the palette's convex hull in linear RGB, "
                             "then nudge back toward the original colour in Oklab space via "
                             "gamut-projected gradient descent.")
    parser.add_argument("--night", action="store_true",
                        help="Nighttime preprocessing: darken (L→L²) and cool (b shifted toward blue)")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    if args.night:
        image = _nighttime(image)

    if args.mix == "spectral":
        result = mix_convert_spectral(image)
    elif args.mix == "additive":
        result = mix_convert_additive(image)
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
