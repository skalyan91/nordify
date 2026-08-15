#version 330 core
// Pass 4: median of the masked histogram from depth_hist_masked.* --
// bin-resolution-limited (256 bins) approximation of
// `np.median(depth_raw[piece_mask])`, normalised to [0,1] the same way
// _depth_blur normalises depth_raw (depth_blur.py:893's d_focus). One
// 1x1 draw: cumulative-sum the 256 bins, take the bin where the running
// count first reaches half the total.
//
// Falls back to (0.0, 0.0) — same convention as depth_blur.py's "no
// candidate found" fallback to d_focus=0.0 -- when the histogram is
// empty (no figure was identified).

out vec4 fragColor;   // (medianDepthNormalized, totalCount, 0, 0)

uniform sampler2D u_hist;   // NUM_BINS x 1, R32F

const int NUM_BINS = 256;

void main() {
    float total = 0.0;
    for (int i = 0; i < NUM_BINS; i++) {
        total += texelFetch(u_hist, ivec2(i, 0), 0).r;
    }
    if (total <= 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float half_   = total * 0.5;
    float running = 0.0;
    int   medianBin = NUM_BINS - 1;
    for (int i = 0; i < NUM_BINS; i++) {
        running += texelFetch(u_hist, ivec2(i, 0), 0).r;
        if (running >= half_) { medianBin = i; break; }
    }

    float medianDepth = (float(medianBin) + 0.5) / float(NUM_BINS);
    fragColor = vec4(medianDepth, total, 0.0, 0.0);
}
