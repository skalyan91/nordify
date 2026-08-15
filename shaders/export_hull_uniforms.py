#!/usr/bin/env python3
"""Generate the additive_mix.frag uniform data from palettize.py's own palette.

This is the single source of truth for the shader's hull geometry: it
reuses palettize.py's `_halfspace_eqs` / `_face_geometry` directly rather
than re-deriving the palette hull, so the shader and the Python
`--mix additive` path can never silently drift apart.

Usage:
    python3 export_hull_uniforms.py            # print GLSL-ready literals
    python3 export_hull_uniforms.py --upload    # print PyOpenGL upload code

Import `build_uniforms()` directly if you're wiring this into a real host
app instead of copy-pasting literals.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import palettize as nf


def build_uniforms(palette_name=None):
    """Returns a dict of numpy arrays ready to upload as the shader's
    uniforms, for the named palette (default: nf.DEFAULT_PALETTE/Nord).
    mat3 entries are already transposed to GLSL's column-major layout, so
    they must be uploaded with transpose=False
    (glUniformMatrix3fv(loc, 1, GL_FALSE, data)) — do not transpose again."""
    pal_bgr = np.array(nf._palette_bgr(palette_name), dtype=np.float32) / 255.0
    pal_lin = np.stack([
        nf._srgb_to_linear(pal_bgr[:, 2]),
        nf._srgb_to_linear(pal_bgr[:, 1]),
        nf._srgb_to_linear(pal_bgr[:, 0]),
    ], axis=-1).astype(np.float32)

    black = np.zeros((1, 3), dtype=np.float32)
    white = np.ones((1, 3), dtype=np.float32)
    pal_ext = np.vstack([pal_lin, black, white])

    hull_eqs, hull_tris = nf._halfspace_eqs(pal_ext)   # (F, 4), (F, 3)
    geom = nf._face_geometry(pal_ext, hull_tris)        # V0, U, Wv, L0, P, Q — each (F, 3)

    return {
        'num_faces':  hull_eqs.shape[0],
        'hull_eqs':   hull_eqs.astype(np.float32),                       # (F, 4)
        'V0':         geom['V0'], 'U': geom['U'], 'Wv': geom['Wv'],       # (F, 3) each
        'L0':         geom['L0'], 'P': geom['P'], 'Q':  geom['Q'],        # (F, 3) each
        'rgb2lms':    nf._M_RGB_TO_LMS.T.astype(np.float32),              # (3, 3), GLSL column-major
        'lms2oklab':  nf._M_LMS_TO_OKLAB.T.astype(np.float32),            # (3, 3), GLSL column-major
    }


def _glsl_vec3_array(name, arr):
    lines = [f"const vec3 {name}[{arr.shape[0]}] = vec3[](" ]
    rows = [f"    vec3({v[0]:.9g}, {v[1]:.9g}, {v[2]:.9g})" for v in arr]
    lines.append(",\n".join(rows))
    lines.append(");")
    return "\n".join(lines)


def _glsl_vec4_array(name, arr):
    lines = [f"const vec4 {name}[{arr.shape[0]}] = vec4[]("]
    rows = [f"    vec4({v[0]:.9g}, {v[1]:.9g}, {v[2]:.9g}, {v[3]:.9g})" for v in arr]
    lines.append(",\n".join(rows))
    lines.append(");")
    return "\n".join(lines)


def print_glsl_literals():
    """Alternative to uniform upload: paste these directly into the
    shader in place of the `uniform ...` declarations, if you'd rather
    bake the (fixed, palette-derived) hull geometry into the shader
    source than set it at runtime. NUM_FACES must match len(hull_eqs)."""
    u = build_uniforms()
    print(f"// NUM_FACES = {u['num_faces']} — update the shader's `const int NUM_FACES` to match.")
    print(_glsl_vec4_array('u_hullEqs', u['hull_eqs']))
    print(_glsl_vec3_array('u_V0', u['V0']))
    print(_glsl_vec3_array('u_U', u['U']))
    print(_glsl_vec3_array('u_Wv', u['Wv']))
    print(_glsl_vec3_array('u_L0', u['L0']))
    print(_glsl_vec3_array('u_P', u['P']))
    print(_glsl_vec3_array('u_Q', u['Q']))


def _flist(arr):
    return [float(x) for x in np.round(arr.flatten(), 9)]


def print_pyopengl_upload(program_var='program'):
    u = build_uniforms()
    print(f"""\
# PyOpenGL upload example — call once after linking `{program_var}`.
# (Any other binding — moderngl, glfw+PyOpenGL, etc. — needs the same
# five calls; only the wrapper syntax differs.)
from OpenGL.GL import *

def upload_hull_uniforms({program_var}):
    glUseProgram({program_var})
    glUniformMatrix3fv(glGetUniformLocation({program_var}, "u_rgb2lms"),   1, GL_FALSE, {_flist(u['rgb2lms'])})
    glUniformMatrix3fv(glGetUniformLocation({program_var}, "u_lms2oklab"), 1, GL_FALSE, {_flist(u['lms2oklab'])})
    glUniform4fv(glGetUniformLocation({program_var}, "u_hullEqs"), {u['num_faces']}, {_flist(u['hull_eqs'])})
    glUniform3fv(glGetUniformLocation({program_var}, "u_V0"), {u['num_faces']}, {_flist(u['V0'])})
    glUniform3fv(glGetUniformLocation({program_var}, "u_U"),  {u['num_faces']}, {_flist(u['U'])})
    glUniform3fv(glGetUniformLocation({program_var}, "u_Wv"), {u['num_faces']}, {_flist(u['Wv'])})
    glUniform3fv(glGetUniformLocation({program_var}, "u_L0"), {u['num_faces']}, {_flist(u['L0'])})
    glUniform3fv(glGetUniformLocation({program_var}, "u_P"),  {u['num_faces']}, {_flist(u['P'])})
    glUniform3fv(glGetUniformLocation({program_var}, "u_Q"),  {u['num_faces']}, {_flist(u['Q'])})
""")


if __name__ == '__main__':
    if '--upload' in sys.argv:
        print_pyopengl_upload()
    else:
        print_glsl_literals()
