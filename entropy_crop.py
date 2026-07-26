#!/usr/bin/env python3
"""Pick a crop offset that minimises the entropy of the edges it would cut,
then apply that same crop to one or more images.

For a target aspect ratio, slides the crop window along whichever axis
needs narrowing and evaluates, at each candidate offset, the gradient
magnitude across the *entire* kept region — not just the single boundary
line itself, since a feature can sit a few pixels inside the cut
(antialiasing, blur, an object that isn't perfectly axis-aligned) and still
be the thing that gets clipped. Each row/column is weighted by a parabola
that is 0 at the crop's own centre and rises to 1 at each boundary, so
content right at the cut counts fully, content near it still counts
somewhat, and content deep in the kept interior counts for nothing — a
smooth falloff with no arbitrary cutoff width to choose. Treating that
weighted gradient magnitude as an unnormalised distribution over
rows/columns, low entropy means the cuts are concentrated in a few of them
(the boundary mostly runs through flat, featureless regions and only clips
something in a narrow band) rather than smeared evenly across many
different rows/columns, i.e. many different objects. Cutting no edges at
all is the best case and scores as zero entropy.

The offset is chosen from one "edge source", then applied identically to
every input given — so a matched set of images (e.g. day/night variants of
the same wallpaper) crop in lock-step. The edge source is typically a depth
map: its edges are exactly object silhouettes (foreground vs. background),
so cutting through one means clipping a real object — unlike a colour
image, whose Sobel edges also fire on brushwork, texture and painted detail
that have nothing to do with where objects actually are. A colour image
still works as an edge source (e.g. when no depth map is available) and is
handled identically — `_gradient_magnitude` accepts either.

Requirements:
    pip install opencv-python-headless numpy
"""
import argparse
import sys

import cv2
import numpy as np


def _gradient_magnitude(image):
    """Continuous Sobel gradient magnitude, robust to flat-colour images.

    Accepts a (H, W, 3) BGR colour image or a (H, W) single-channel array
    (e.g. a depth map) directly — a depth map's own scale is irrelevant
    here, only where it changes sharply.

    A thresholded edge detector (e.g. Canny) goes nearly empty on a
    palette-snapped image, where most of the frame is large flat colour
    regions — leaving pathological ties between candidate offsets. Sobel's
    continuous magnitude has no such dead zone.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float64)


def _entropy(profile):
    total = float(profile.sum())
    if total <= 1e-9:
        return 0.0, total
    p = profile / total
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum()), total


def find_crop_offset(edge_source, ratio_w, ratio_h):
    """Minimum-entropy crop offset for `ratio_w:ratio_h` from `edge_source`
    (a BGR colour image or a single-channel depth map — see `_gradient_magnitude`).

    Crops along whichever axis the target ratio requires narrowing — width
    if the source is relatively wider than the target, else height (mirrors
    depth_blur.py's _crop_to_aspect).

    Rather than scoring only the single boundary row/column, every
    row/column in the kept window contributes, weighted by a parabola
    `((x - centre) / (new_length/2))**2` (x = absolute position, centre =
    the crop window's own midpoint): 1 at each boundary, 0 at the centre.
    Expanding the square, the weighted sum over the window is a
    combination of three window sums — of the gradient magnitude itself,
    of position-weighted magnitude, and of position^2-weighted magnitude —
    each obtainable from a prefix sum along the crop axis in O(1), so
    scanning every candidate offset stays O(excess x other) despite scoring
    the entire window rather than a fixed-width margin.

    Returns (axis, offset, new_length, entropies, totals): axis is 'x' or
    'y'; entropies/totals are per-offset arrays covering the full excess
    range, for diagnostics.
    """
    h, w = edge_source.shape[:2]
    mag = _gradient_magnitude(edge_source)

    if w * ratio_h > h * ratio_w:   # image wider than target: crop width
        axis, length, other_ratio_num, other_ratio_den = 'x', w, ratio_w, ratio_h
        other = h
    else:                            # image taller than target: crop height
        axis, length, other_ratio_num, other_ratio_den = 'y', h, ratio_h, ratio_w
        other = w
    new_length = other * other_ratio_num // other_ratio_den
    excess = length - new_length
    half = new_length / 2.0

    # Work with the crop axis as columns (axis=1) regardless of 'x' or 'y',
    # so the same prefix-sum logic covers both.
    m = mag if axis == 'x' else mag.T
    xs = np.arange(m.shape[1], dtype=np.float64)

    def prefix(values):
        return np.concatenate([np.zeros((m.shape[0], 1)), np.cumsum(values, axis=1)], axis=1)

    cum0 = prefix(m)
    cum1 = prefix(m * xs[None, :])
    cum2 = prefix(m * xs[None, :] ** 2)

    def window_sum(cum, a, b):
        return cum[:, b] - cum[:, a]

    entropies = np.zeros(excess + 1)
    totals = np.zeros(excess + 1)
    for o in range(excess + 1):
        centre = o + half
        s0 = window_sum(cum0, o, o + new_length)
        s1 = window_sum(cum1, o, o + new_length)
        s2 = window_sum(cum2, o, o + new_length)
        # sum_x mag(x) * ((x - centre) / half)^2, expanded
        profile = (s2 - 2 * centre * s1 + centre * centre * s0) / (half * half)
        profile = np.clip(profile, 0, None)  # guard against float rounding
        entropies[o], totals[o] = _entropy(profile)

    # Primary key: entropy. Secondary: total edge weight cut (breaks the
    # many-way ties zero-entropy offsets otherwise produce).
    best = int(np.lexsort((totals, entropies))[0])
    return axis, best, new_length, entropies, totals


def _apply_crop(image, axis, offset, new_length):
    return image[offset:offset + new_length, :] if axis == 'y' else image[:, offset:offset + new_length]


def main():
    parser = argparse.ArgumentParser(
        description="Pick a crop that minimises the entropy of the edges it would cut, "
                    "then apply the same crop to one or more images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("edge_source",
                        help="Image to run edge detection on and choose the crop offset from "
                             "(typically a depth map, e.g. from depth_blur.py --save-depth)")
    parser.add_argument("--aspect", default="16:9", metavar="W:H", help="Target aspect ratio")
    parser.add_argument("images", nargs='+', metavar="IN OUT",
                        help="Input/output path pairs to crop identically (must be pixel-aligned "
                             "with edge_source); include edge_source itself if it should be cropped too")
    parser.add_argument("--visualize", metavar="PATH",
                        help="Save a copy of edge_source with the chosen crop boundaries drawn on it")

    args = parser.parse_args()
    if len(args.images) % 2 != 0:
        parser.error("images must be given as IN OUT pairs")

    ratio_w, ratio_h = (int(x) for x in args.aspect.split(':'))

    edge_img = cv2.imread(args.edge_source)
    if edge_img is None:
        print(f"Error: cannot read image '{args.edge_source}'", file=sys.stderr)
        sys.exit(1)
    h, w = edge_img.shape[:2]

    axis, offset, new_length, entropies, totals = find_crop_offset(edge_img, ratio_w, ratio_h)
    excess = len(entropies) - 1
    print(f"  [crop] edge source '{args.edge_source}' ({w}x{h}) -> target {ratio_w}:{ratio_h}",
          file=sys.stderr)
    print(f"  [crop] narrowing {axis}-axis to {new_length}px (excess {excess}px); "
          f"chosen offset {offset} (centre would be {excess // 2})", file=sys.stderr)
    print(f"  [crop] entropy at chosen offset: {entropies[offset]:.4f} bits, "
          f"edge weight cut: {totals[offset]:.1f}", file=sys.stderr)

    if args.visualize:
        vis = edge_img.copy()
        p0, p1 = offset, offset + new_length - 1
        if axis == 'x':
            cv2.line(vis, (p0, 0), (p0, h - 1), (0, 0, 255), 4)
            cv2.line(vis, (p1, 0), (p1, h - 1), (0, 0, 255), 4)
        else:
            cv2.line(vis, (0, p0), (w - 1, p0), (0, 0, 255), 4)
            cv2.line(vis, (0, p1), (w - 1, p1), (0, 0, 255), 4)
        cv2.imwrite(args.visualize, vis)
        print(f"Saved boundary visualisation '{args.visualize}'")

    for i in range(0, len(args.images), 2):
        in_path, out_path = args.images[i], args.images[i + 1]
        img = cv2.imread(in_path)
        if img is None:
            print(f"Error: cannot read image '{in_path}'", file=sys.stderr)
            sys.exit(1)
        if img.shape[:2] != (h, w):
            print(f"Error: '{in_path}' ({img.shape[1]}x{img.shape[0]}) is not pixel-aligned "
                  f"with edge source ({w}x{h})", file=sys.stderr)
            sys.exit(1)
        crop = _apply_crop(img, axis, offset, new_length)
        ok = cv2.imwrite(out_path, crop)
        if not ok:
            print(f"Error: cannot write image '{out_path}'", file=sys.stderr)
            sys.exit(1)
        print(f"Saved '{out_path}' ({crop.shape[1]}x{crop.shape[0]})")


if __name__ == "__main__":
    main()
