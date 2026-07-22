#!/usr/bin/env python3
"""Pick a crop offset that minimises the entropy of the edges it would cut,
then apply that same crop to one or more images.

For a target aspect ratio, slides the crop window along whichever axis
needs narrowing and evaluates, at each candidate offset, the two boundary
lines the crop would cut through. Wherever a boundary line crosses a strong
image gradient, that row (or column) "loses" part of a feature. Treating
the boundary-line gradient magnitude as an unnormalised distribution over
rows/columns, low entropy means the cuts are concentrated in a few of them
(the boundary mostly runs through flat, featureless regions and only clips
something in a narrow band) rather than smeared evenly across many
different rows/columns, i.e. many different objects. Cutting no edges at
all is the best case and scores as zero entropy.

The offset is chosen from one "edge source" image, then applied identically
to every input given — so a matched set of images (e.g. day/night variants
of the same wallpaper) crop in lock-step.

Requirements:
    pip install opencv-python-headless numpy
"""
import argparse
import sys

import cv2
import numpy as np


def _gradient_magnitude(image_bgr):
    """Continuous Sobel gradient magnitude, robust to flat-colour images.

    A thresholded edge detector (e.g. Canny) goes nearly empty on a
    palette-snapped image, where most of the frame is large flat colour
    regions — leaving pathological ties between candidate offsets. Sobel's
    continuous magnitude has no such dead zone.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
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


def find_crop_offset(image_bgr, ratio_w, ratio_h):
    """Minimum-entropy crop offset for `ratio_w:ratio_h` from `image_bgr`.

    Crops along whichever axis the target ratio requires narrowing — width
    if the source is relatively wider than the target, else height (mirrors
    depth_blur.py's _crop_to_aspect).

    Returns (axis, offset, new_length, entropies, totals): axis is 'x' or
    'y'; entropies/totals are per-offset arrays covering the full excess
    range, for diagnostics.
    """
    h, w = image_bgr.shape[:2]
    mag = _gradient_magnitude(image_bgr)

    if w * ratio_h > h * ratio_w:   # image wider than target: crop width
        axis, length, other_ratio_num, other_ratio_den = 'x', w, ratio_w, ratio_h
        other = h
    else:                            # image taller than target: crop height
        axis, length, other_ratio_num, other_ratio_den = 'y', h, ratio_h, ratio_w
        other = w
    new_length = other * other_ratio_num // other_ratio_den
    excess = length - new_length

    entropies = np.zeros(excess + 1)
    totals = np.zeros(excess + 1)
    for o in range(excess + 1):
        o1 = o + new_length - 1
        profile = (mag[:, o] + mag[:, o1]) if axis == 'x' else (mag[o, :] + mag[o1, :])
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
                        help="Image to run edge detection on and choose the crop offset from")
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
