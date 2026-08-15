#!/usr/bin/env python3
"""Host driver for the night-colouring pipeline (see README.md).
`prepare(image_tex, size)` runs everything independent of `amount`
(oklab conversion + the downsampled multi-scale DoG light-peak
detection, ~100ms+ at demo resolutions — dominated by 11 separable-
Gaussian blur passes); `resolve(amount)` is the cheap (~0.1ms measured)
single-pass darken/cool-or-brighten/warm transform that reads
prepare()'s cached results, safe to call repeatedly for a live
"night <-> day" slider without rerunning detection each time.
`run(image_tex, size, amount)` is prepare()+resolve() in one call, for
one-shot use. Detection itself runs at reduced resolution — see
prepare()'s own comment.

Requires moderngl (`pip install moderngl`).
"""
import math
import os

import moderngl
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))

SIGMAS = (2.5, 4.0, 6.0, 10.0, 16.0, 24.0)   # palettize._dog_light_peaks default
THRESHOLD = 0.12
STRENGTH_SCALE = 3.0
SPREAD = 1.5
LIGHT_BOOST = 0.2

# Detection (blur stack + DoG + protection map) runs at 1/DOWNSAMPLE
# resolution -- see run()'s own comment for why. Not exposed as a public
# knob: 2 was validated (timing + peak-detection accuracy) against the
# full-resolution reference; a larger factor risks losing the smallest
# detection scale (sigma=2.5, already near the documented noise floor
# where sub-2px scales pick up specular glints instead of real lights --
# see palettize.py's _dog_light_peaks docs) to downsample blur entirely.
DOWNSAMPLE = 2

LEVEL_SIGMAS = [math.sqrt(SIGMAS[i] * SIGMAS[i + 1]) for i in range(len(SIGMAS) - 1)]


def _read(name):
    with open(os.path.join(_DIR, name)) as f:
        return f.read()


def _inject_common(src, common_src):
    nl = src.index('\n')
    return src[:nl + 1] + common_src + '\n' + src[nl + 1:]


class NightPipeline:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        common = _read('common.glsl')
        fullscreen_vert = _read(os.path.join('..', 'fullscreen.vert'))

        def frag(name):
            return _inject_common(_read(name), common)

        names = ['oklab_convert', 'minmax_seed', 'minmax_reduce',
                 'downsample2x', 'upsample_bilinear',
                 'gaussian_blur_h', 'gaussian_blur_v', 'dog_nms',
                 'combine_max', 'nighttime_resolve']
        self.prog = {n: ctx.program(vertex_shader=fullscreen_vert, fragment_shader=frag(f'{n}.frag'))
                    for n in names}
        self._vao = {n: ctx.vertex_array(p, []) for n, p in self.prog.items()}

    def _tex(self, size, components=4):
        return self.ctx.texture(size, components, dtype='f4')

    def _fbo(self, tex):
        return self.ctx.framebuffer(color_attachments=[tex])

    def _draw(self, name, target_fbo, uniforms):
        prog = self.prog[name]
        unit = 0
        for k, v in uniforms.items():
            if k not in prog:
                continue
            u = prog[k]
            if isinstance(v, moderngl.Texture):
                v.use(location=unit)
                u.value = unit
                unit += 1
            else:
                u.value = v
        target_fbo.use()
        self.ctx.viewport = (0, 0, target_fbo.size[0], target_fbo.size[1])
        self.ctx.disable(moderngl.BLEND)
        self._vao[name].render(mode=moderngl.TRIANGLES, vertices=3)

    def _minmax_b(self, oklab_tex, W, H):
        seed = self._tex((W, H), 2)
        self._draw('minmax_seed', self._fbo(seed), {'u_oklab': oklab_tex})
        cur, cur_w, cur_h = seed, W, H
        while cur_w > 1 or cur_h > 1:
            nw, nh = max(1, math.ceil(cur_w / 2)), max(1, math.ceil(cur_h / 2))
            dst = self._tex((nw, nh), 2)
            self._draw('minmax_reduce', self._fbo(dst), {'u_src': cur, 'u_srcSize': (cur_w, cur_h)})
            cur, cur_w, cur_h = dst, nw, nh
        return cur

    def _blur(self, src_tex, size, sigma, normalize):
        W, H = size
        radius = int(math.ceil(3.0 * sigma))
        h_out = self._tex(size, 4)
        self._draw('gaussian_blur_h', self._fbo(h_out), dict(
            u_src=src_tex, u_imageSize=size, u_sigma=sigma, u_radius=radius, u_normalize=normalize))
        v_out = self._tex(size, 4)
        self._draw('gaussian_blur_v', self._fbo(v_out), dict(
            u_src=h_out, u_imageSize=size, u_sigma=sigma, u_radius=radius, u_normalize=normalize))
        return v_out

    def prepare(self, image_tex, size):
        """image_tex: moderngl.Texture, RGBA, sRGB [0,1]. size: (W, H).
        Runs everything independent of `amount`: oklab conversion and the
        downsampled multi-scale DoG light-peak detection, ending in a
        full-resolution protection map. Stores results on self;
        resolve() (cheap -- see its own comment) reads them, as many
        times as you like for different `amount` values, without
        rerunning any of this. Call this once per image, not on every
        tick of a live amount slider.
        """
        W, H = size
        self._size = size

        oklab = self._tex(size, 4)
        self._draw('oklab_convert', self._fbo(oklab), {'u_image': image_tex})
        self.oklab = oklab

        self.b_minmax = self._minmax_b(oklab, W, H)

        # Detection (every blur + DoG + protection-map pass below) runs at
        # 1/DOWNSAMPLE resolution -- confirmed by profiling that the 11
        # separable-Gaussian passes here (radii up to ~90px at full
        # resolution) dominate this pipeline's total cost, since blur cost
        # scales as pixels x radius. Downsampling once, halving every sigma
        # to match (same *physical* detection scale, fewer pixels and a
        # proportionally smaller radius), then upsampling only the final
        # protect map back to full resolution keeps oklab_convert/
        # _minmax_b/nighttime_resolve untouched (they still see the real
        # full-resolution image) while cutting the dominant cost roughly
        # DOWNSAMPLE^3 (DOWNSAMPLE^2 fewer pixels x DOWNSAMPLE smaller
        # radius per blur pass).
        d_size = (max(1, math.ceil(W / DOWNSAMPLE)), max(1, math.ceil(H / DOWNSAMPLE)))
        oklab_small = self._tex(d_size, 4)
        self._draw('downsample2x', self._fbo(oklab_small), dict(u_src=oklab, u_srcSize=size))

        blurred = [self._blur(oklab_small, d_size, s / DOWNSAMPLE, normalize=True) for s in SIGMAS]

        peaks = []
        for level in range(5):
            peak = self._tex(d_size, 4)
            self._draw('dog_nms', self._fbo(peak), dict(
                u_blur0=blurred[0], u_blur1=blurred[1], u_blur2=blurred[2],
                u_blur3=blurred[3], u_blur4=blurred[4], u_blur5=blurred[5],
                u_imageSize=d_size, u_level=level, u_threshold=THRESHOLD,
                u_strengthScale=STRENGTH_SCALE))
            peaks.append(peak)

        protects = [self._blur(peaks[i], d_size, LEVEL_SIGMAS[i] * SPREAD / DOWNSAMPLE, normalize=False)
                   for i in range(5)]

        protect_map_small = self._tex(d_size, 4)
        self._draw('combine_max', self._fbo(protect_map_small), dict(
            u_p0=protects[0], u_p1=protects[1], u_p2=protects[2],
            u_p3=protects[3], u_p4=protects[4]))

        protect_map = self._tex(size, 4)
        self._draw('upsample_bilinear', self._fbo(protect_map), dict(
            u_src=protect_map_small, u_srcSize=d_size, u_dstSize=size))
        self.protect_map = protect_map

    def resolve(self, amount=1.0):
        """Cheap, single-pass: applies the darken/cool (or brighten/warm)
        transform at `amount` (continuous in [-1, 1] -- see
        nighttime_resolve.frag's own header comment) using prepare()'s
        cached detection results. Measured ~0.1ms in isolation (vs.
        ~100ms+ for the detection stage prepare() runs). Call prepare()
        first. Returns an (H, W, 3) float32 array in [0, 1], sRGB.
        """
        W, H = self._size
        out_tex = self._tex(self._size, 4)
        self._draw('nighttime_resolve', self._fbo(out_tex), dict(
            u_oklab=self.oklab, u_protect=self.protect_map, u_bMinMax=self.b_minmax,
            u_lightBoost=LIGHT_BOOST, u_amount=amount))
        data = np.frombuffer(out_tex.read(), dtype=np.float32).reshape(H, W, 4)
        return np.clip(data[:, :, :3], 0.0, 1.0)

    def run(self, image_tex, size, amount=1.0):
        """Convenience: prepare() + resolve() in one call, for one-shot use."""
        self.prepare(image_tex, size)
        return self.resolve(amount)
