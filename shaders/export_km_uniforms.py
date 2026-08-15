#!/usr/bin/env python3
"""Generate subtractive_mix.frag's uniform data from palettize.py's own
palette K/S spectra (`_fit_palette_ks`) -- the single source of truth for
the shader's candidate triangles and pure-palette fallback colours, so the
shader and the Python `--mix spectral` path share the same underlying
palette data (though not the same search algorithm -- see the shader's
own docs on why this is a heuristic approximation, not a direct port).

Usage:
    python3 export_km_uniforms.py            # print GLSL-ready literals
    python3 export_km_uniforms.py --upload    # print PyOpenGL upload code

Import `build_uniforms()` directly if you're wiring this into a real host
app instead of copy-pasting literals.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import palettize as nf


def build_uniforms(palette_name=None):
    """Returns a dict of numpy arrays ready to upload as subtractive_mix.frag's
    uniforms, for the named palette (default: nf.DEFAULT_PALETTE/Nord).
    mat3 entries are already transposed to GLSL's column-major
    layout (upload with transpose=False)."""
    ks = nf._fit_palette_ks(nf._palette_bgr(palette_name)).astype(np.float64)   # (N_PAL, 31)
    n_pal = ks.shape[0]

    k_norm = float((nf._D65 * nf._CIE_CMF[:, 1]).sum())
    d65n_cmf = (nf._D65[:, None] / k_norm * nf._CIE_CMF).astype(np.float64)   # (31, 3)

    R = nf._km_to_lin(ks)                                    # (N_PAL, 31)
    xyz = R @ d65n_cmf                                        # (N_PAL, 3)
    palette_rgb = np.clip(xyz @ nf._M_XYZ_TO_RGB.T, 0.0, 1.0)   # (N_PAL, 3) linear RGB
    palette_oklab = np.stack(
        nf._linear_rgb_to_oklab(palette_rgb[:, 0], palette_rgb[:, 1], palette_rgb[:, 2]),
        axis=-1)                                                # (N_PAL, 3)

    _, hull_tris = nf._halfspace_eqs(palette_oklab.astype(np.float32))   # (F,4), (F,3)

    ks0 = ks[hull_tris[:, 0]]                                 # (F, 31)
    p = ks[hull_tris[:, 1]] - ks0
    q = ks[hull_tris[:, 2]] - ks0

    # u_kmTriangles: (NUM_BANDS, NUM_FACES) RGB32F texture, texel(band, f)
    # = (ks0, p, q) -- see subtractive_mix.frag's own comment on why this
    # is a texture rather than a `uniform float[]` array (GLSL pads every
    # element of a default-block array to a full vec4, which blows the
    # 4096-component uniform limit on real hardware for data this size).
    # GL texture upload wants row-major (height, width, channels) --
    # height = NUM_FACES, width = NUM_BANDS, so texel(band, f) lands at
    # ivec2(band, f) as intended: NO transpose (ks0/p/q are already
    # (NUM_FACES, NUM_BANDS)).
    km_triangles_tex = np.stack([ks0, p, q], axis=-1)          # (NUM_FACES, NUM_BANDS, 3)
    d65n_cmf_tex = d65n_cmf[None, :, :]                          # (1, NUM_BANDS, 3)

    return {
        'n_pal':       n_pal,
        'num_faces':   hull_tris.shape[0],
        'ks0':         ks0.astype(np.float32), 'p': p.astype(np.float32), 'q': q.astype(np.float32),
        'd65n_cmf':    d65n_cmf.astype(np.float32),                # (31, 3)
        'km_triangles_texture': km_triangles_tex.astype(np.float32),   # (NUM_FACES, NUM_BANDS, 3) = (height, width, 3)
        'd65n_cmf_texture':     d65n_cmf_tex.astype(np.float32),       # (1, NUM_BANDS, 3) = (height, width, 3)
        'palette_rgb':   palette_rgb.astype(np.float32),           # (N_PAL, 3)
        'palette_oklab': palette_oklab.astype(np.float32),         # (N_PAL, 3)
        'rgb2lms':     nf._M_RGB_TO_LMS.T.astype(np.float32),       # (3, 3), GLSL column-major
        'lms2oklab':   nf._M_LMS_TO_OKLAB.T.astype(np.float32),     # (3, 3), GLSL column-major
        'xyz2rgb':     nf._M_XYZ_TO_RGB.T.astype(np.float32),       # (3, 3), GLSL column-major
    }


def _flist(arr):
    return [float(x) for x in np.round(np.asarray(arr).flatten(), 9)]


def _glsl_vec3_array(name, arr):
    lines = [f"const vec3 {name}[{arr.shape[0]}] = vec3[]("]
    rows = [f"    vec3({v[0]:.9g}, {v[1]:.9g}, {v[2]:.9g})" for v in arr]
    lines.append(",\n".join(rows))
    lines.append(");")
    return "\n".join(lines)


def print_glsl_literals():
    u = build_uniforms()
    print(f"// N_PAL = {u['n_pal']}, NUM_FACES = {u['num_faces']} -- "
          f"update the shader's `const int` declarations to match.\n"
          f"// u_kmTriangles and u_d65nCmf are TEXTURES, not uniform arrays "
          f"(see subtractive_mix.frag's own comment on why) -- there is no\n"
          f"// GLSL literal form for texture contents; use --upload for the "
          f"upload code, or call build_uniforms() directly and upload\n"
          f"// km_triangles_texture / d65n_cmf_texture as RGB32F textures "
          f"of shape (NUM_BANDS, NUM_FACES) / (NUM_BANDS, 1).")
    print(_glsl_vec3_array('u_paletteRgb', u['palette_rgb']))
    print(_glsl_vec3_array('u_paletteOklab', u['palette_oklab']))


def print_pyopengl_upload(program_var='program'):
    """Covers all of subtractive_mix.frag's own uniforms -- see
    shaders/wallpaper/web/palettize.js's SubtractivePipeline for a complete,
    working example (WebGL2) of wiring this shader up for real."""
    u = build_uniforms()
    num_faces, num_bands = u['km_triangles_texture'].shape[:2]   # (height, width)
    print(f"""\
# PyOpenGL upload example -- call once after linking `{program_var}`.
# u_kmTriangles / u_d65nCmf are RGB32F textures (see subtractive_mix.frag's
# comment on why plain uniform arrays don't fit this data), bound to
# whichever texture units your host isn't already using for u_image.
from OpenGL.GL import *
import numpy as np

def upload_km_uniforms({program_var}, km_triangles_unit=1, d65n_cmf_unit=2):
    glUseProgram({program_var})
    glUniformMatrix3fv(glGetUniformLocation({program_var}, "u_rgb2lms"),   1, GL_FALSE, {_flist(u['rgb2lms'])})
    glUniformMatrix3fv(glGetUniformLocation({program_var}, "u_lms2oklab"), 1, GL_FALSE, {_flist(u['lms2oklab'])})
    glUniformMatrix3fv(glGetUniformLocation({program_var}, "u_xyz2rgb"),   1, GL_FALSE, {_flist(u['xyz2rgb'])})
    glUniform3fv(glGetUniformLocation({program_var}, "u_paletteRgb"),   {u['n_pal']}, {_flist(u['palette_rgb'])})
    glUniform3fv(glGetUniformLocation({program_var}, "u_paletteOklab"), {u['n_pal']}, {_flist(u['palette_oklab'])})

    def _rgb32f_texture(unit, w, h, data_flat_rgb):
        glActiveTexture(GL_TEXTURE0 + unit)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT,
                    np.array(data_flat_rgb, dtype=np.float32))
        return tex

    _rgb32f_texture(km_triangles_unit, {num_bands}, {num_faces}, {_flist(u['km_triangles_texture'])})
    glUniform1i(glGetUniformLocation({program_var}, "u_kmTriangles"), km_triangles_unit)
    _rgb32f_texture(d65n_cmf_unit, {num_bands}, 1, {_flist(u['d65n_cmf_texture'])})
    glUniform1i(glGetUniformLocation({program_var}, "u_d65nCmf"), d65n_cmf_unit)
""")


if __name__ == '__main__':
    if '--upload' in sys.argv:
        print_pyopengl_upload()
    else:
        print_glsl_literals()
