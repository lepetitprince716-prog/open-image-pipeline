"""Frame-to-frame motion energy for the performance-layer A/B.

Contact sheets show whether beats land; they cannot show whether a clip is
alive between beats. This decodes each clip to small grayscale frames and
reports the mean absolute difference between consecutive frames, split into
thirds so a dead final hold is visible as a near-zero tail.

The metric is full-frame: camera moves and animated scenery count as energy
exactly like subject performance does. Compare clips sharing one locked (or
near-locked) camera setup — as the W5/W6 arms do — or a high number may only
mean the world moved.
"""
import subprocess
import sys

import numpy as np

W, H = 96, 160


def energy(path):
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-vf", f"scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True)
    frames = np.frombuffer(proc.stdout, dtype=np.uint8)
    frames = frames.reshape(-1, H, W).astype(np.float32)
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    return diffs


for path in sys.argv[1:]:
    d = energy(path)
    n = len(d)
    a, b = d[:n // 3], d[n // 3:2 * n // 3]
    tail = d[2 * n // 3:]
    name = path.rsplit("/", 1)[-1].replace("mmh3_", "").replace("_00001_.mp4", "")
    print(f"{name:28s} frames={n + 1:4d} mean={d.mean():5.2f} "
          f"| thirds {a.mean():5.2f} {b.mean():5.2f} {tail.mean():5.2f} "
          f"| tail min={tail.min():4.2f} p10={np.percentile(tail, 10):4.2f}")
