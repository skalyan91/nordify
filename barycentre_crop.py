#!/usr/bin/env python3
"""Pick a crop offset that centres the barycentre of a depth map, then
apply that same crop to one or more images.

Treats the depth map as a mass distribution over the image — each pixel's
value (white = near/foreground, black = far/background, matching
depth_blur.py's --save-depth output) is its weight — and computes the
weighted centroid along whichever axis the target aspect ratio needs to
narrow. The crop window is then centred on that coordinate (clamped so it
stays in bounds). Because near/foreground content is usually a photo's
subject, this tends to keep the subject centred without any separate
saliency model — reusing the depth map depth_blur.py already computes for
the blur pass, rather than re-deriving "what matters" from image edges.

The offset is chosen from one depth-map image, then applied identically to
every input given — so a matched set of images (e.g. day/night variants of
the same wallpaper) crop in lock-step.

Requirements:
    pip install opencv-python-headless numpy
"""
import argparse
import sys

import cv2
import numpy as np


def find_crop_offset(depth_map, ratio_w, ratio_h):
    """Barycentre-centred crop offset for `ratio_w:ratio_h` from `depth_map`.

    Crops along whichever axis the target ratio requires narrowing — width
    if the map is relatively wider than the target, else height (mirrors
    depth_blur.py's _crop_to_aspect). A depth map with zero total mass (e.g.
    perfectly flat) falls back to a centred crop.

    Returns (axis, offset, new_length, barycentre): axis is 'x' or 'y'.
    """
    h, w = depth_map.shape[:2]
    weights = depth_map.astype(np.float64)

    if w * ratio_h > h * ratio_w:   # map wider than target: crop width
        axis, length, other, other_num, other_den = 'x', w, h, ratio_w, ratio_h
        marginal = weights.sum(axis=0)   # (W,) mass per column
    else:                             # map taller than target: crop height
        axis, length, other, other_num, other_den = 'y', h, w, ratio_h, ratio_w
        marginal = weights.sum(axis=1)   # (H,) mass per row

    new_length = other * other_num // other_den
    excess = length - new_length

    total = marginal.sum()
    if total <= 1e-9:
        barycentre = length / 2.0
    else:
        coords = np.arange(length, dtype=np.float64)
        barycentre = float((coords * marginal).sum() / total)

    offset = int(round(barycentre - new_length / 2.0))
    offset = max(0, min(excess, offset))
    return axis, offset, new_length, barycentre


def _apply_crop(image, axis, offset, new_length):
    return image[offset:offset + new_length, :] if axis == 'y' else image[:, offset:offset + new_length]


def main():
    parser = argparse.ArgumentParser(
        description="Pick a crop that centres the depth map's barycentre, "
                    "then apply the same crop to one or more images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("depth_map",
                        help="Greyscale depth map (e.g. from depth_blur.py --save-depth) to "
                             "compute the barycentre from")
    parser.add_argument("--aspect", default="16:9", metavar="W:H", help="Target aspect ratio")
    parser.add_argument("images", nargs='+', metavar="IN OUT",
                        help="Input/output path pairs to crop identically (must be pixel-aligned "
                             "with depth_map); include depth_map itself in the pairs if it should "
                             "be cropped too")
    parser.add_argument("--visualize", metavar="PATH",
                        help="Save a copy of depth_map with the barycentre and chosen crop "
                             "boundaries drawn on it")

    args = parser.parse_args()
    if len(args.images) % 2 != 0:
        parser.error("images must be given as IN OUT pairs")

    ratio_w, ratio_h = (int(x) for x in args.aspect.split(':'))

    depth_img = cv2.imread(args.depth_map, cv2.IMREAD_GRAYSCALE)
    if depth_img is None:
        print(f"Error: cannot read image '{args.depth_map}'", file=sys.stderr)
        sys.exit(1)
    h, w = depth_img.shape[:2]

    axis, offset, new_length, barycentre = find_crop_offset(depth_img, ratio_w, ratio_h)
    excess = (w if axis == 'x' else h) - new_length
    print(f"  [crop] depth map '{args.depth_map}' ({w}x{h}) -> target {ratio_w}:{ratio_h}",
          file=sys.stderr)
    print(f"  [crop] narrowing {axis}-axis to {new_length}px (excess {excess}px); "
          f"barycentre at {barycentre:.1f}, chosen offset {offset} (centre would be {excess // 2})",
          file=sys.stderr)

    if args.visualize:
        vis = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
        p0, p1, bc = offset, offset + new_length - 1, int(round(barycentre))
        if axis == 'x':
            cv2.line(vis, (bc, 0), (bc, h - 1), (0, 255, 0), 2)
            cv2.line(vis, (p0, 0), (p0, h - 1), (0, 0, 255), 4)
            cv2.line(vis, (p1, 0), (p1, h - 1), (0, 0, 255), 4)
        else:
            cv2.line(vis, (0, bc), (w - 1, bc), (0, 255, 0), 2)
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
                  f"with depth map ({w}x{h})", file=sys.stderr)
            sys.exit(1)
        crop = _apply_crop(img, axis, offset, new_length)
        ok = cv2.imwrite(out_path, crop)
        if not ok:
            print(f"Error: cannot write image '{out_path}'", file=sys.stderr)
            sys.exit(1)
        print(f"Saved '{out_path}' ({crop.shape[1]}x{crop.shape[0]})")


if __name__ == "__main__":
    main()
