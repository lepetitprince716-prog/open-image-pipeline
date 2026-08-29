#!/usr/bin/env python3
"""foot_scan.py — toe fissure scanner with a two-evidence confirmation gate.

投产闸门（2026-08-22 rt50 收口落地）：
程序化梯度趾缝检测的报警 ≠ 缺陷确诊。本工具输出 SUSPECT 报警 + 原生分辨率
裁剪板（复核介质），必须第二证据（vision 原生分辨率读板 或 人工复核）一致
才可放行进修复队列。油皮/高光 LoRA 的镜面条纹是天然假峰源（#49 事件）。

用法：
  # 1) 扫描：梯度检测 → 打印 SUSPECT 峰 + 写裁剪板 + JSON 报告
  .venv/bin/python scripts/foot_scan.py scan IMG.png [--crops-out DIR] \\
      [--threshold 1.2] [--min-gap 8] [--band Y0,Y1]

  # 2) 复核介质：--crops-out 里每个峰一张近原生分辨率竖条裁剪，给 vision/人工读

  # 3) 闸门：合并检测峰 + 第二证据 verdicts，输出放行清单
  .venv/bin/python scripts/foot_scan.py gate IMG.png --peaks 461,478,491 \\
      --evidence ev.json [--crops-out DIR]

  # ev.json 格式：{"peaks":[{"x":461,"verdict":"defect","note":"vision: 双平行暗线"},
  #                          {"x":478,"verdict":"clean","note":"高光带，非缝"}]}

输出：SUSPECT（未确认）/ CONFIRMED（双证据一致，可进修复队列）/
     DISMISSED（第二证据否，纯净图 —「无病修复」拦截）。

Authoritative protocol: docs/t2i_prompt_validate_anatomy.md; runbook:
docs/runbooks/foot_fix_runbook.md（"Gate" 节）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_gray(path: str) -> tuple[np.ndarray, tuple[int, int]]:
    from PIL import Image
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        size = rgb.size
        gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    return gray, size


def scan_fissures(
    gray: np.ndarray,
    *,
    threshold_sigma: float = 1.2,
    min_gap: int = 8,
    band: tuple[int, int] | None = None,
) -> list[dict]:
    """水平差分列峰值检测（rt50 复刻：mean + 1.2σ）。

    diff[x] = mean_y |I(y,x+1) - I(y,x-1)|，趾缝是竖直暗线 → 沿 x 出现响应峰。
    band=None 用全高；band=(y0,y1) 限制在足部带（排除脚踝以上干扰）。
    """
    h, w = gray.shape
    y0, y1 = band if band is not None else (0, h)
    y0 = max(0, y0)
    y1 = min(h, y1)
    if y1 - y0 < 4:
        y0, y1 = 0, h

    # 中心差分（对称，缝两沿同响应）
    d = np.abs(gray[y0:y1, 2:] - gray[y0:y1, :-2])
    col = d.mean(axis=0)  # 每列(相对 x=1..w-2)平均差分强度
    # 对齐回原始 x 索引
    col = np.concatenate([[0.0], col, [0.0]])

    mean = col.mean()
    std = col.std()
    if std < 1e-6:
        return []
    thresh = mean + threshold_sigma * std

    # 局部极大 + 阈值 + 最小间距
    peaks: list[dict] = []
    last = -10**9
    for x in range(2, w - 2):
        if col[x] > thresh and col[x] >= col[x - 1] and col[x] >= col[x + 1]:
            if x - last < min_gap:
                # 同簇取更强峰
                if col[x] > peaks[-1]["strength"]:
                    peaks[-1].update(x=x, strength=float(col[x]),
                                     z=float((col[x] - mean) / std))
                continue
            peaks.append({
                "x": int(x), "strength": float(col[x]),
                "z": float((col[x] - mean) / std),
                "threshold": float(thresh), "mean": float(mean), "std": float(std),
                "band": [y0, y1],
            })
            last = x
    return peaks


def write_peak_crops(rgb_path: str, peaks: list[dict], out_dir: Path,
                     stem: str, width_px: int = 90) -> list[Path]:
    """每个峰一条竖条裁剪（近原生宽度，供 vision 原生分辨率读板）。"""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with Image.open(rgb_path) as im:
        rgb = im.convert("RGB")
        W, H = rgb.size
        for p in peaks:
            x = p["x"]
            x0 = max(0, x - width_px // 2)
            x1 = min(W, x + width_px // 2 + 1)
            crop = rgb.crop((x0, 0, x1, H))
            dest = out_dir / f"{stem}_peak_x{x:04d}.png"
            crop.save(dest)
            p["crop"] = str(dest)
            saved.append(dest)
    return saved


def _load_evidence(path: str) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    peaks = raw.get("peaks")
    if not isinstance(peaks, list):
        raise ValueError(f"--evidence {path}: missing peaks[]")
    return raw


def gate(scan_json: dict, evidence: dict) -> list[dict]:
    """双证据合流：检测峰 + 第二证据 verdict → 放行/拦截/待定。"""
    by_x = {p["x"]: p for p in scan_json.get("peaks", [])}
    ev = {p.get("x"): p.get("verdict", "pending") for p in evidence.get("peaks", [])}
    out: list[dict] = []
    for x, p in by_x.items():
        v = ev.get(x, "pending")
        p["verdict"] = v
        p["gate"] = (
            "CONFIRMED" if v == "defect" else
            "DISMISSED" if v == "clean" else
            "SUSPECT"
        )
        out.append(p)
    # 第二证据里出现、检测没报警的峰（vision 独立发现）
    for x, v in ev.items():
        if x not in by_x:
            out.append({"x": x, "verdict": v, "gate": "CONFIRMED" if v == "defect" else "SUSPECT",
                        "note": "second-evidence-only peak"})
    out.sort(key=lambda q: q["x"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="gradient fissure scan -> SUSPECT peaks + crops")
    p_scan.add_argument("image")
    p_scan.add_argument("--crops-out", default=None, metavar="DIR")
    p_scan.add_argument("--threshold", type=float, default=1.2, metavar="SIGMA")
    p_scan.add_argument("--min-gap", type=int, default=8)
    p_scan.add_argument("--band", default=None, metavar="Y0,Y1",
                        help="restrict detection to a vertical band (toe row)")
    p_scan.add_argument("--crop-width", type=int, default=90)

    p_gate = sub.add_parser("gate", help="merge scan peaks + second-evidence verdicts")
    p_gate.add_argument("image")
    p_gate.add_argument("--peaks", required=True,
                        help="comma-separated x positions from the scan")
    p_gate.add_argument("--evidence", required=True, metavar="JSON",
                        help='{"peaks":[{"x":461,"verdict":"defect|clean|pending",...}]}')
    p_gate.add_argument("--crops-out", default=None, metavar="DIR")
    p_gate.add_argument("--threshold", type=float, default=1.2)
    p_gate.add_argument("--min-gap", type=int, default=8)
    p_gate.add_argument("--band", default=None, metavar="Y0,Y1")

    args = ap.parse_args(argv)

    if args.cmd == "scan":
        gray, size = _load_gray(args.image)
        band: tuple[int, int] | None = None
        if args.band:
            b0, b1 = (int(v) for v in args.band.split(","))
            band = (b0, b1)
        peaks = scan_fissures(gray, threshold_sigma=args.threshold,
                              min_gap=args.min_gap, band=band)
        stem = Path(args.image).stem
        if args.crops_out and peaks:
            write_peak_crops(args.image, peaks, Path(args.crops_out),
                             stem, width_px=args.crop_width)
        report = {
            "image": args.image, "size": list(size),
            "threshold_sigma": args.threshold, "min_gap": args.min_gap,
            "band": list(band) if band else None,
            "peaks": peaks, "status": "SUSPECT",
            "note": "梯度报警≠缺陷确诊：需 vision 原生分辨率复核或双证据一致才进修复队列",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if peaks:
            print(f"\nSUSPECT peaks: {[p['x'] for p in peaks]} "
                  f"(z={[round(p['z'],1) for p in peaks]})")
            print("复核介质 ->", args.crops_out or "(无；用 --crops-out 生成裁剪板)")
            print("闸门：run gate 子命令合流第二证据")
        else:
            print("no peaks")
        return 0

    # gate
    gray, size = _load_gray(args.image)
    # gate 不复扫：检测峰来自 scan 子命令输出（--peaks），否则 band/阈值不一致会漏峰
    wanted = {int(v) for v in args.peaks.split(",") if v.strip()}
    scan_json: dict = {"peaks": [{"x": x, "strength": 0.0, "z": 0.0} for x in sorted(wanted)]}
    evidence = _load_evidence(args.evidence)
    merged = gate(scan_json, evidence)
    if args.crops_out and any(p["gate"] == "CONFIRMED" for p in merged):
        write_peak_crops(args.image, [p for p in merged if p["gate"] == "CONFIRMED"],
                         Path(args.crops_out), Path(args.image).stem)
    out = {
        "image": args.image, "size": list(size),
        "gate": merged,
        "summary": {
            "confirmed": [p["x"] for p in merged if p["gate"] == "CONFIRMED"],
            "dismissed": [p["x"] for p in merged if p["gate"] == "DISMISSED"],
            "suspect": [p["x"] for p in merged if p["gate"] == "SUSPECT"],
        },
        "note": "CONFIRMED 才可进修复队列；DISMISSED=第二证据否（无病修复拦截）；SUSPECT=待复核",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())