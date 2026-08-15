// Shared functions for the night-colouring pipeline. Injected after each
// pass's `#version` line by pipeline.py's `_inject_common`, same pattern
// as shaders/wallpaper/common.glsl.

vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 linearToSrgb(vec3 c) {
    c = max(c, 0.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

// palettize.py's _linear_rgb_to_oklab / _oklab_to_linear_rgb, as scalar
// formulas rather than uniform matrices -- Oklab is a fixed colour
// space, not palette-derived data, so there's nothing for a Python
// export script to compute here (unlike additive_mix.frag /
// subtractive_mix.frag's palette-dependent matrices).

vec3 rgbLinToOklab(vec3 c) {
    float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    vec3 lms_ = sign(vec3(l, m, s)) * pow(abs(vec3(l, m, s)), vec3(1.0 / 3.0));
    float L = 0.2104542553 * lms_.x + 0.7936177850 * lms_.y - 0.0040720468 * lms_.z;
    float a = 1.9779984951 * lms_.x - 2.4285922050 * lms_.y + 0.4505937099 * lms_.z;
    float b = 0.0259040371 * lms_.x + 0.4072165126 * lms_.y - 0.4331205297 * lms_.z;
    return vec3(L, a, b);
}

vec3 oklabToRgbLin(vec3 lab) {
    float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    float r =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    float g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    float b = -0.0041960863 * l - 0.7034186147 * m + 1.6956086611 * s;
    return vec3(r, g, b);
}
