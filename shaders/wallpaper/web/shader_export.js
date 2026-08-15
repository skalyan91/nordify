// Builds a downloadable .zip of exactly the GLSL shaders relevant to the
// demo's *current* state -- not the whole shaders/ directory regardless
// of what's actually in use. "Relevant" means: the mixing shader for
// whichever palette + mode is selected (only if palette mixing is
// enabled), the night-colouring shaders (only if night colouring is
// enabled), and the wallpaper crop/focus shaders (only if a depth map +
// segmentation map are actually driving the current render).
//
// Every exported .frag/.vert file is already fully self-contained --
// build_shaders.py inlines common.glsl's contents and the version/
// precision header into each shader string at build time, so nothing
// here needs to be assembled further before it'll compile on its own.

import { FULLSCREEN_VERT, WALLPAPER, NIGHT, MIXING } from './shaders.js';
import { PALETTE_GEOMETRY } from './palette_data.js';
import { withConstInts } from './palettize.js';

const FFLATE_CDN_URL = 'https://cdn.jsdelivr.net/npm/fflate@0.8.2/+esm';

// Reverses shaders.js's own _js_key() naming (build_shaders.py):
// "foo.frag" -> "foo_frag", "foo.vert" -> "foo_vert".
function _glslFileName(jsKey) {
  if (jsKey.endsWith('_frag')) return `${jsKey.slice(0, -5)}.frag`;
  if (jsKey.endsWith('_vert')) return `${jsKey.slice(0, -5)}.vert`;
  return jsKey;
}

function _addDir(files, dirName, sourceObj) {
  for (const [key, src] of Object.entries(sourceObj)) {
    files[`${dirName}/${_glslFileName(key)}`] = src;
  }
}

function _buildReadme(state, includedMixing) {
  const lines = [];
  lines.push('Shaders exported from the Wallpaper Ricer web demo\n');
  lines.push('===================================================\n\n');
  lines.push('Included (only what was actually active when this was downloaded):\n\n');
  if (state.wallpaperActive) {
    lines.push('  fullscreen.vert, wallpaper/  -- crop + figure-sensitive + depth-guided blur.\n');
    lines.push('    See shaders/wallpaper/README.md and wallpaper.js for the pass order and\n');
    lines.push('    uniform contract (u_segmentation etc).\n\n');
  }
  if (state.nightEnabled) {
    lines.push('  fullscreen.vert, night/      -- nighttime colour transform.\n');
    lines.push('    See shaders/night/README.md and night.js for the pass order.\n\n');
  }
  if (includedMixing) {
    lines.push(`  fullscreen.vert, mixing/     -- ${state.mixMode} palette mixing, compiled\n`);
    lines.push(`    specifically for the '${state.paletteName}' palette (GLSL array sizes are\n`);
    lines.push('    compile-time constants, so this shader is NOT reusable for a different\n');
    lines.push('    palette without recompiling against that palette\'s own facet/colour counts\n');
    lines.push('    -- see palettize.js\'s withConstInts()).\n\n');
    lines.push(`    mixing/${state.paletteName}_${state.mixMode}_geometry.json has the numeric\n`);
    lines.push('    uniform data this shader needs to actually run (hull/facet geometry or\n');
    lines.push('    K/S triangle data + the palette\'s own colours) -- see\n');
    lines.push('    shaders/export_hull_uniforms.py / export_km_uniforms.py (the Python code\n');
    lines.push('    that generated it) or this demo\'s own palettize.js\'s AdditivePipeline /\n');
    lines.push('    SubtractivePipeline for a working upload example.\n\n');
  }
  lines.push('Every .frag/.vert file already has its own #version/precision header and any\n');
  lines.push('shared GLSL (common.glsl) inlined by this project\'s build_shaders.py -- each\n');
  lines.push('file compiles on its own, no assembly step needed. For the full pipeline\n');
  lines.push('driver code that actually runs these passes end-to-end, see this project\'s\n');
  lines.push('own shaders/ directory (pipeline.py, or wallpaper.js / night.js / palettize.js\n');
  lines.push('for the WebGL2 ports this demo itself runs).\n');
  return lines.join('');
}

// state: { paletteName, mixEnabled, mixMode ('additive'|'subtractive'),
//          nightEnabled, wallpaperActive } -- see demo.js's onDownloadShaders().
export async function downloadShaderZip(state, onStatus) {
  const files = {};
  let includedAnything = false;
  let includedMixing = false;

  if (state.wallpaperActive) {
    files['fullscreen.vert'] = FULLSCREEN_VERT;
    _addDir(files, 'wallpaper', WALLPAPER);
    includedAnything = true;
  }
  if (state.nightEnabled) {
    files['fullscreen.vert'] = FULLSCREEN_VERT;
    _addDir(files, 'night', NIGHT);
    includedAnything = true;
  }
  if (state.mixEnabled) {
    const geometry = PALETTE_GEOMETRY[state.paletteName];
    if (!geometry) {
      onStatus?.(`Can't export: no precomputed geometry for palette '${state.paletteName}'.`);
      return;
    }
    files['fullscreen.vert'] = FULLSCREEN_VERT;
    if (state.mixMode === 'additive') {
      const g = geometry.additive;
      files[`mixing/${state.paletteName}_additive_mix.frag`] =
        withConstInts(MIXING.additive_mix_frag, { NUM_FACES: g.num_faces });
      files[`mixing/${state.paletteName}_additive_geometry.json`] = JSON.stringify(g, null, 2);
    } else {
      const g = geometry.subtractive;
      files[`mixing/${state.paletteName}_subtractive_mix.frag`] =
        withConstInts(MIXING.subtractive_mix_frag, { NUM_FACES: g.num_faces, N_PAL: g.n_pal });
      files[`mixing/${state.paletteName}_subtractive_geometry.json`] = JSON.stringify(g, null, 2);
    }
    includedAnything = true;
    includedMixing = true;
  }

  if (!includedAnything) {
    onStatus?.('Nothing to export -- enable palette mixing / night colouring, or load an '
      + 'image with a depth map + segmentation map, first.');
    return;
  }

  files['README.txt'] = _buildReadme(state, includedMixing);

  onStatus?.('Building shader zip…');
  const { zipSync, strToU8 } = await import(/* webpackIgnore: true */ FFLATE_CDN_URL);
  const zipInput = {};
  for (const [path, content] of Object.entries(files)) zipInput[path] = strToU8(content);
  const zipped = zipSync(zipInput, { level: 6 });

  const blob = new Blob([zipped], { type: 'application/zip' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `wallricer-shaders-${state.paletteName}.zip`;
  a.click();
  URL.revokeObjectURL(url);
  onStatus?.('Shaders downloaded.');
}
