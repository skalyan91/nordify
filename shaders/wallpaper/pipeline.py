#!/usr/bin/env python3
"""Host driver for the wallpaper-pipeline shader passes (see README.md /
the plan this was built from). Sequences passes 0 (depth min/max) through
13 (GMM depth layers) once per image/depth/segmentation/target-size
("setup"), then pass 14 (crop+resize+blur composite) on every call to
render() -- the only pass that needs to rerun when focal depth changes.

Every value that crosses a pass boundary is a GPU texture; nothing is
read back to the CPU between passes, so setup() and render() are both
safe to call from a real-time loop (setup on image/target-size change,
render on focal-depth change or every frame).

Requires moderngl (`pip install moderngl`).
"""
import math
import os

import moderngl
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_SHADERS_ROOT = os.path.dirname(_DIR)

NUM_BINS = 256          # must match every shader's own `const int NUM_BINS`
MAX_SEGMENTS = 64        # must match segment_score.vert / argmax_1d.frag's MAX_SEGMENTS
NUM_CANDIDATES = 32      # must match combine_and_argmin.frag's NUM_CANDIDATES
MAX_K = 8                # must match gmm_iterate.frag / sort_and_bounds.frag's MAX_K


def _read(*parts):
    with open(os.path.join(*parts)) as f:
        return f.read()


def _inject_common(src, common_src):
    """Insert common.glsl's contents right after the `#version` line --
    GLSL requires #version to be the source's literal first line, so it
    can't simply be prepended."""
    nl = src.index('\n')
    return src[:nl + 1] + common_src + '\n' + src[nl + 1:]


class WallpaperPipeline:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        common = _read(_DIR, 'common.glsl')
        fullscreen_vert = _read(_SHADERS_ROOT, 'fullscreen.vert')

        def frag(name):
            return _inject_common(_read(_DIR, name), common)

        def vert(name):
            return _inject_common(_read(_DIR, name), common)

        # "Fullscreen" passes: one fragment per output texel, driven by
        # the shared fullscreen-triangle vertex shader.
        fs_names = [
            'minmax_reduce', 'argmax_1d', 'median_1d',
            'luminance_blur_h', 'luminance_blur_v', 'gradient_mag',
            'prefix_sum_1d', 'weighted_mag_for_offset', 'reduce_pairwise',
            'entropy_seed', 'entropy_1d', 'combine_and_argmin',
            'gmm_init', 'gmm_iterate', 'sort_and_bounds',
            'composite',
        ]
        self.prog = {}
        for name in fs_names:
            self.prog[name] = ctx.program(vertex_shader=fullscreen_vert,
                                          fragment_shader=frag(f'{name}.frag'))
        # minmax_seed reads straight from the source depth texture, not a
        # previous reduction step -- still a fullscreen pass, just listed
        # separately since it has no ping-pong partner shape.
        self.prog['minmax_seed'] = ctx.program(vertex_shader=fullscreen_vert,
                                               fragment_shader=frag('minmax_seed.frag'))

        # GL_POINTS scatter passes: own vertex shader (gl_VertexID-driven,
        # no vertex buffer), additive-blended fragment output.
        for name in ['segment_score', 'depth_hist_masked', 'figure_count_1d',
                     'depth_hist_cropped']:
            self.prog[name] = ctx.program(vertex_shader=vert(f'{name}.vert'),
                                          fragment_shader=frag(f'{name}.frag'))

        self._empty_vao = {name: ctx.vertex_array(p, [])
                           for name, p in self.prog.items()}

        # Populated by setup(); consumed by render().
        self._setup_done = False

    # -- small helpers --------------------------------------------------

    def _tex(self, size, components=4):
        return self.ctx.texture(size, components, dtype='f4')

    def _fbo(self, tex):
        return self.ctx.framebuffer(color_attachments=[tex])

    def _draw_fullscreen(self, name, target_fbo, uniforms, viewport=None):
        prog = self.prog[name]
        prog._next_unit = 0   # texture units are per-draw, not cumulative across calls
        for k, v in uniforms.items():
            self._set_uniform(prog, k, v)
        target_fbo.use()
        self.ctx.viewport = viewport or (0, 0, target_fbo.size[0], target_fbo.size[1])
        self.ctx.disable(moderngl.BLEND)
        self._empty_vao[name].render(mode=moderngl.TRIANGLES, vertices=3)

    def _draw_scatter(self, name, target_fbo, uniforms, num_points):
        prog = self.prog[name]
        prog._next_unit = 0   # texture units are per-draw, not cumulative across calls
        for k, v in uniforms.items():
            self._set_uniform(prog, k, v)
        target_fbo.use()
        self.ctx.viewport = (0, 0, target_fbo.size[0], target_fbo.size[1])
        target_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        self._empty_vao[name].render(mode=moderngl.POINTS, vertices=num_points)
        self.ctx.disable(moderngl.BLEND)

    @staticmethod
    def _set_uniform(prog, name, value):
        if name not in prog:
            return
        u = prog[name]
        if isinstance(value, moderngl.Texture):
            unit = getattr(prog, '_next_unit', 0)
            value.use(location=unit)
            u.value = unit
            prog._next_unit = unit + 1
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            u.write(np.array(value, dtype='f4').tobytes())
        else:
            u.value = value

    def _reduce_2d_minmax(self, seed_tex, w, h):
        """Log-step 2x2 min/max reduction (minmax_seed.frag then repeated
        minmax_reduce.frag steps) down to a 1x1 RG32F texture."""
        cur = seed_tex
        cur_w, cur_h = w, h
        while cur_w > 1 or cur_h > 1:
            next_w, next_h = max(1, math.ceil(cur_w / 2)), max(1, math.ceil(cur_h / 2))
            dst = self._tex((next_w, next_h), 2)
            self._draw_fullscreen('minmax_reduce', self._fbo(dst),
                                  {'u_src': cur, 'u_srcSize': (cur_w, cur_h)})
            cur, cur_w, cur_h = dst, next_w, next_h
        return cur

    def _reduce_1d_sum(self, src_tex, length, other, axis_is_x):
        """Log-step pairwise-sum reduction (reduce_pairwise.frag) along
        one axis of an RG32F buffer, down to size 1 along that axis."""
        cur = src_tex
        cur_len = length
        while cur_len > 1:
            next_len = max(1, math.ceil(cur_len / 2))
            size = (next_len, other) if axis_is_x else (other, next_len)
            src_size = (cur_len, other) if axis_is_x else (other, cur_len)
            dst = self._tex(size, 4)
            self._draw_fullscreen('reduce_pairwise', self._fbo(dst),
                                  {'u_src': cur, 'u_srcSize': src_size,
                                   'u_axisIsX': 1 if axis_is_x else 0})
            cur, cur_len = dst, next_len
        return cur

    def _prefix_sum_1d(self, src_tex, length):
        cur = src_tex
        step = 1
        while step < length:
            dst = self._tex((length, 1), 4)
            self._draw_fullscreen('prefix_sum_1d', self._fbo(dst),
                                  {'u_src': cur, 'u_step': step})
            cur = dst
            step *= 2
        return cur

    # -- setup: passes 0-13 ---------------------------------------------

    def setup(self, image_tex, depth_tex, segmentation_tex, image_size, target_size,
             ratio_w, ratio_h, k_layers=5, em_iters=8, figure_penalty_weight=4.0,
             viewing_distance_factor=1.5, e2_degrees=2.3):
        """image_tex, depth_tex, segmentation_tex: moderngl.Texture (RGBA/R32F/R32F).
        image_size: (W, H) of the source textures. target_size: (W, H) of
        the eventual render() output. ratio_w/ratio_h: target aspect
        (mirrors entropy_crop.py's find_crop_offset args)."""
        W, H = image_size
        gp = dict(u_viewingDistanceFactor=viewing_distance_factor, u_e2Degrees=e2_degrees)

        # Pass 0: depth min/max.
        seed = self._tex((W, H), 2)
        self._draw_fullscreen('minmax_seed', self._fbo(seed), {'u_depth': depth_tex})
        self.depth_minmax = self._reduce_2d_minmax(seed, W, H)

        # Passes 1-2: figure identification.
        seg_accum = self._tex((MAX_SEGMENTS, 1), 4)
        self._draw_scatter('segment_score', self._fbo(seg_accum), dict(
            u_depth=depth_tex, u_segmentation=segmentation_tex,
            u_depthMinMax=self.depth_minmax, u_imageSize=(W, H), **gp), W * H)
        self.figure_info = self._tex((1, 1), 4)
        self._draw_fullscreen('argmax_1d', self._fbo(self.figure_info), dict(
            u_accum=seg_accum, u_count=MAX_SEGMENTS, u_startIndex=1, u_findMax=True))

        # Passes 3-4: figure median depth.
        depth_hist = self._tex((NUM_BINS, 1), 4)
        self._draw_scatter('depth_hist_masked', self._fbo(depth_hist), dict(
            u_depth=depth_tex, u_segmentation=segmentation_tex,
            u_depthMinMax=self.depth_minmax, u_figureInfo=self.figure_info,
            u_imageSize=(W, H)), W * H)
        self.figure_median_depth = self._tex((1, 1), 4)
        self._draw_fullscreen('median_1d', self._fbo(self.figure_median_depth),
                              dict(u_hist=depth_hist))

        # Determine crop axis/lengths on the host (plain arithmetic on
        # known sizes -- entropy_crop.py:182-189's axis selection, not a
        # GPU reduction, so a CPU computation here doesn't reintroduce
        # any readback).
        if W * ratio_h > H * ratio_w:
            axis_is_x, length, other = True, W, H
            new_length = other * ratio_w // ratio_h
        else:
            axis_is_x, length, other = False, H, W
            new_length = other * ratio_h // ratio_w
        new_length = max(1, min(int(new_length), length))
        excess = length - new_length
        self.crop_axis_is_x = axis_is_x
        self.new_length = new_length

        # Pass 5: gradient magnitude of the crop edge-source (the image).
        gray_h = self._tex((W, H), 1)
        self._draw_fullscreen('luminance_blur_h', self._fbo(gray_h),
                              dict(u_image=image_tex, u_imageSize=(W, H)))
        gray_hv = self._tex((W, H), 1)
        self._draw_fullscreen('luminance_blur_v', self._fbo(gray_hv),
                              dict(u_gray=gray_h, u_imageSize=(W, H)))
        grad_mag = self._tex((W, H), 1)
        self._draw_fullscreen('gradient_mag', self._fbo(grad_mag),
                              dict(u_grayBlurred=gray_hv, u_imageSize=(W, H)))

        # Pass 6-7: figure-pixel count by crop-axis position, prefix-summed.
        fig_count = self._tex((length, 1), 4)
        self._draw_scatter('figure_count_1d', self._fbo(fig_count), dict(
            u_segmentation=segmentation_tex, u_figureInfo=self.figure_info,
            u_imageSize=(W, H), u_cropAxisIsX=int(axis_is_x),
            u_cropAxisLength=length), W * H)
        fig_prefix = self._prefix_sum_1d(fig_count, length)

        # Pass 8: per-candidate weighted-entropy reduction.
        if excess <= 0:
            candidate_offsets = [0] * NUM_CANDIDATES
        else:
            candidate_offsets = [int(round(x)) for x in
                                 np.linspace(0, excess, NUM_CANDIDATES)]
        candidate_scores = self._tex((NUM_CANDIDATES, 1), 4)
        candidate_scores_fbo = self._fbo(candidate_scores)
        for c, offset in enumerate(candidate_offsets):
            weighted = self._tex((new_length, other), 2)
            self._draw_fullscreen('weighted_mag_for_offset', self._fbo(weighted), dict(
                u_gradMag=grad_mag, u_cropAxisIsX=int(axis_is_x), u_offset=offset,
                u_newLength=new_length, u_other=other, **gp))
            profile = self._reduce_1d_sum(weighted, new_length, other, axis_is_x=True)
            seeded = self._tex((1, other), 4)
            self._draw_fullscreen('entropy_seed', self._fbo(seeded), dict(u_profile=profile))
            moments = self._reduce_1d_sum(seeded, other, 1, axis_is_x=False)
            self._draw_fullscreen('entropy_1d', candidate_scores_fbo, dict(u_moments=moments),
                                  viewport=(c, 0, 1, 1))

        # Pass 9: combine entropy + figure-cut penalty, argmin.
        self.crop_offset = self._tex((1, 1), 4)
        self._draw_fullscreen('combine_and_argmin', self._fbo(self.crop_offset), dict(
            u_candidateScores=candidate_scores, u_figureCountPrefix=fig_prefix,
            u_candidateOffsets=candidate_offsets, u_newLength=new_length,
            u_cropAxisLength=length, u_maxEntropyBits=math.log2(max(other, 2)),
            u_figurePenaltyWeight=figure_penalty_weight))

        # Pass 10-13: adaptive GMM depth layers, restricted to the crop window.
        cropped_hist = self._tex((NUM_BINS, 1), 4)
        self._draw_scatter('depth_hist_cropped', self._fbo(cropped_hist), dict(
            u_depth=depth_tex, u_depthMinMax=self.depth_minmax,
            u_cropOffset=self.crop_offset, u_imageSize=(W, H),
            u_cropAxisIsX=int(axis_is_x), u_newLength=new_length, **gp), W * H)

        self.k = min(k_layers, MAX_K)
        components = self._tex((self.k, 1), 4)   # (mean, variance, weight, _) per K
        self._draw_fullscreen('gmm_init', self._fbo(components), dict(u_k=self.k))
        for _ in range(em_iters):
            nxt = self._tex((self.k, 1), 4)
            self._draw_fullscreen('gmm_iterate', self._fbo(nxt),
                                  dict(u_hist=cropped_hist, u_components=components, u_k=self.k))
            components = nxt
        self.layer_centers = self._tex((self.k, 1), 4)
        self._draw_fullscreen('sort_and_bounds', self._fbo(self.layer_centers),
                              dict(u_centroids=components, u_k=self.k))

        self.image_tex, self.depth_tex = image_tex, depth_tex
        self.image_size, self.target_size = image_size, target_size
        self._setup_done = True

    # -- per-frame: pass 14 ----------------------------------------------

    def render(self, focal_depth=None, sigma_max=24.0):
        """focal_depth: None to fall back to the identified figure's
        median depth, else a float in [0,1]. Returns an (H, W, 3) uint8
        sRGB array. The only pass re-run here; safe to call every frame."""
        assert self._setup_done, "call setup() first"
        out = self._tex(self.target_size, 4)
        self._draw_fullscreen('composite', self._fbo(out), dict(
            u_image=self.image_tex, u_depth=self.depth_tex,
            u_depthMinMax=self.depth_minmax, u_layerCenters=self.layer_centers,
            u_cropOffset=self.crop_offset, u_figureMedianDepth=self.figure_median_depth,
            u_imageSize=self.image_size, u_targetSize=self.target_size,
            u_cropAxisIsX=int(self.crop_axis_is_x), u_newLength=self.new_length,
            u_k=self.k, u_sigmaMax=sigma_max,
            u_focalDepth=focal_depth if focal_depth is not None else 0.0,
            u_focalDepthIsSet=focal_depth is not None))
        data = out.read()
        w, h = self.target_size
        arr = np.frombuffer(data, dtype=np.float32).reshape(h, w, 4)[:, :, :3]
        return np.clip(arr, 0.0, 1.0)
