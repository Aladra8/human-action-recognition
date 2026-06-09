#!/usr/bin/env python3

import argparse
import os
import re
import sys

try:
    import cv2
    import numpy as np
except ImportError:
    print("\n[ERROR] opencv-python is not installed.")
    print("        Fix:  pip3 install opencv-python\n")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

FRAMES_PER_SEQ = 40
GT_FRAME_IDX   = 19   # 0-indexed → frame 20 (median frame)

CLASS_IDS = {
    "walking":      1,
    "jogging":      2,
    "running":      3,
    "boxing":       4,
    "handwaving":   5,
    "handclapping": 6,
}

SELECTIONS = {
    "d1": [1, 2, 3],
    "d2": [4, 5, 6],
    "d3": [7, 8, 9],
    "d4": [10, 11, 12],
}

KTH_PATTERN = re.compile(
    r"person(\d{2})_([a-z]+)_(d[1-4])_uncomp\.avi$", re.IGNORECASE
)

# ── Smart frame-window selection ──────────────────────────────────────────────

def read_all_frames(avi_path: str) -> list:
    """Read every frame from the AVI file."""
    cap = cv2.VideoCapture(avi_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames

def score_windows(frames: list, window: int = 40) -> int:
    """
    Find the start index of the 40-frame window where the human is most
    clearly and stably visible.

    Strategy:
      1. Build a median background from ALL frames.
      2. Compute foreground pixel count per frame.
      3. Score each window = median(fg_area) - 0.4 * std(fg_area)
         (rewards stable presence, penalises flickering/transitions).
      4. Return the start index of the highest-scoring window.
    """
    n = len(frames)
    if n <= window:
        return 0

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
             for f in frames]

    # Pixel-wise median background from all frames
    bg = np.median(np.stack(grays, 0).astype(np.float32), 0).astype(np.uint8)

    # Foreground area per frame
    fg = np.zeros(n, dtype=np.float32)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for i, g in enumerate(grays):
        diff = cv2.absdiff(g, bg)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        fg[i] = float(np.count_nonzero(mask))

    # Score each window
    best_score = -1.0
    best_start = 0
    for start in range(n - window + 1):
        w = fg[start: start + window]
        score = float(np.median(w)) - 0.4 * float(np.std(w))
        if score > best_score:
            best_score = score
            best_start = start

    return best_start

# ── Ground-truth bbox generation ──────────────────────────────────────────────

def auto_gt_bbox(frames: list) -> tuple:
    """
    Normalised (xc, yc, w, h) for frame 20 via background subtraction.
    Mirrors the median model in detector.cpp exactly.
    """
    if not frames:
        return (0.5, 0.5, 0.4, 0.8)

    img_h, img_w = frames[0].shape[:2]
    idx = min(GT_FRAME_IDX, len(frames) - 1)

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
             for f in frames]

    bg   = np.median(np.stack(grays, 0).astype(np.float32), 0).astype(np.uint8)
    diff = cv2.absdiff(grays[idx], bg)
    _, mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = None
    for c in contours:
        if cv2.contourArea(c) >= 400:
            r = cv2.boundingRect(c)
            if bbox is None:
                bbox = list(r)
            else:
                x1 = min(bbox[0], r[0])
                y1 = min(bbox[1], r[1])
                x2 = max(bbox[0]+bbox[2], r[0]+r[2])
                y2 = max(bbox[1]+bbox[3], r[1]+r[3])
                bbox = [x1, y1, x2-x1, y2-y1]

    if bbox is None:
        return (0.5, 0.5, 0.35, 0.75)

    bx, by, bw, bh = bbox
    xc = max(0.01, min(0.99, (bx + bw / 2) / img_w))
    yc = max(0.01, min(0.99, (by + bh / 2) / img_h))
    wn = max(0.01, min(0.95, bw / img_w))
    hn = max(0.01, min(0.95, bh / img_h))
    return (xc, yc, wn, hn)

# ── Frame extractor ───────────────────────────────────────────────────────────

def process_avi(avi_path: str, out_dir: str, class_id: int) -> tuple:
    """
    Find the best 40-frame window in the AVI, save frames as JPEGs,
    write gt.txt.  Returns (n_frames, start_frame_index).
    """
    os.makedirs(out_dir, exist_ok=True)

    all_frames = read_all_frames(avi_path)
    if not all_frames:
        return 0, 0

    # Find the best window
    start = score_windows(all_frames, FRAMES_PER_SEQ)
    frames = all_frames[start: start + FRAMES_PER_SEQ]

    if not frames:
        return 0, 0

    # Save JPEGs
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(out_dir, f"frame_{i+1:03d}.jpg"), f,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Auto-generate GT bbox at frame 20 of this window
    xc, yc, wn, hn = auto_gt_bbox(frames)
    with open(os.path.join(out_dir, "gt.txt"), "w") as fp:
        fp.write(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    return len(frames), start

# ── AVI file scanner ──────────────────────────────────────────────────────────

def scan_avi_files(raw_dir: str) -> dict:
    index = {}
    for root, _, files in os.walk(raw_dir):
        for fname in files:
            m = KTH_PATTERN.match(fname)
            if not m:
                continue
            pid    = int(m.group(1))
            action = m.group(2).lower()
            scen   = m.group(3).lower()
            if action in CLASS_IDS:
                index[(action, pid, scen)] = os.path.join(root, fname)
    return index

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Process unzipped KTH AVI files → 40-frame dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw",    required=True,
                        help="Folder containing unzipped KTH AVI files")
    parser.add_argument("--output", default="./dataset",
                        help="Output dataset folder  (default: ./dataset)")
    parser.add_argument("--force",  action="store_true",
                        help="Re-extract even if sequence already exists")
    args = parser.parse_args()

    raw_dir  = os.path.abspath(args.raw)
    out_root = os.path.abspath(args.output)

    if not os.path.isdir(raw_dir):
        print(f"\n[ERROR] Not found: {raw_dir}\n"); sys.exit(1)

    os.makedirs(out_root, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║   KTH Dataset Processor  (smart windows)     ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Raw AVI source : {raw_dir}")
    print(f"  Output         : {out_root}")
    print(f"  Force re-extract: {'yes' if args.force else 'no'}")
    print()

    print("  Scanning for AVI files …", end=" ", flush=True)
    avi_index = scan_avi_files(raw_dir)
    print(f"found {len(avi_index)} KTH video files.")
    print()

    if not avi_index:
        print("[ERROR] No KTH AVI files found. Expected names like:")
        print("  person01_boxing_d1_uncomp.avi")
        sys.exit(1)

    total = sum(len(v) for v in SELECTIONS.values()) * len(CLASS_IDS)
    done = succeeded = 0
    failed = []
    missing = []

    for action in CLASS_IDS:
        class_id = CLASS_IDS[action]
        print(f"  ╌╌ {action.upper()} (class {class_id}) ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌")

        for scenario, persons in SELECTIONS.items():
            for pid in persons:
                done += 1
                key      = (action, pid, scenario)
                seq_name = f"{action}_{scenario}_p{pid:02d}"
                seq_dir  = os.path.join(out_root, action, seq_name)
                prefix   = f"  [{done:2d}/{total}] {seq_name}"

                # Skip if already done (unless --force)
                if not args.force and os.path.isdir(seq_dir):
                    imgs = [f for f in os.listdir(seq_dir) if f.endswith(".jpg")]
                    if len(imgs) >= 10 and os.path.isfile(
                            os.path.join(seq_dir, "gt.txt")):
                        print(f"{prefix}  ✔ already done ({len(imgs)} frames)")
                        succeeded += 1
                        continue

                if key not in avi_index:
                    print(f"{prefix}  ✘ AVI not found")
                    missing.append(seq_name); continue

                print(f"{prefix}  selecting best window …", end=" ", flush=True)
                n, start = process_avi(avi_index[key], seq_dir, class_id)

                if n > 0:
                    print(f"{n} frames from t={start} ✔")
                    succeeded += 1
                else:
                    print("failed ✘")
                    failed.append(seq_name)

        print()

    print("╔══════════════════════════════════════════════╗")
    print("║   Done                                       ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Succeeded : {succeeded} / {total}")
    print(f"  Missing   : {len(missing)}")
    print(f"  Failed    : {len(failed)}")

    if succeeded == total:
        print()
        print("  ✔ Dataset ready!")
        print(f"  1. Verify:  ./dataset_check.sh {out_root}")
        print(f"  2. Rebuild: cd build && make -j$(sysctl -n hw.ncpu) && cd ..")
        print(f"  3. Run:     ./har {out_root} --loocv --visualize")
    print()

if __name__ == "__main__":
    main()