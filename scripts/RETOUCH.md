# Corrective retouch line — fix first, grade last

Companion doc for `retouch.py` (corrective toolbox) + `post_fx.py` (stylistic finish) + `fx_sweep.py`
(parameter-sweep contact sheets). Pure post-processing (PIL + numpy); `retouch.py`'s cv2 denoise
accelerant is optional, with a PIL fallback. No GPU, no ComfyUI.

## What each tool is for

| | `retouch.py` | `post_fx.py` |
|--|--|--|
| Job | **fix defects**: noise, colour cast, dull/flat contrast, blown/crushed levels, soft focus | **add a look**: colour grade, halation, vignette, grain |
| Frame | "this render has a problem" | "this render is clean, now make it art" |
| Neutral state | `Retouch()` / preset `none` = exact identity | `Grade("none")` / preset `none` = exact identity |
| Determinism | fully deterministic, no randomness anywhere in the chain | deterministic per `--seed` (only `grain` uses randomness) |

They are sequential, not interchangeable. Grading a defective image (crushed shadows, a colour cast,
noise) bakes the defect into the graded look — `post_fx.py`'s halation/vignette/grain will amplify
noise and a cast will fight the colour grade. Retouch first, always.

## Canonical order

```
retouch.py  (fix defects)   -->   post_fx.py  (add a look)
```

Never the reverse. If an image needs both, run `retouch.py` to a clean intermediate, then `post_fx.py`
on that intermediate — don't try to fold correction into a grading pass.

```bash
python3 retouch.py raw.png --preset fix --out raw_rt.png
python3 post_fx.py raw_rt.png --preset film --out raw_final.png
```

## Ops table (`retouch.py` fields)

Apply order inside `retouch_image`: denoise → auto_wb → wb_temp/tint → auto_levels → manual levels
(black/white/gamma) → curve → scurve → shadows → highlights → vibrance → saturation → clarity →
sharpen.

| param | range | fixes |
|-------|-------|-------|
| `denoise` | 0..1 | sensor/compression noise (cv2 fastNlMeansDenoisingColored, or PIL median-filter blend without cv2) |
| `auto_wb` | 0..1 | colour cast, automatically (gray-world gain per channel, luminance-preserving) |
| `wb_temp` | -1..1 | manual warm(+)/cool(-) cast correction |
| `wb_tint` | -1..1 | manual green(+)/magenta(-) cast correction |
| `auto_levels` | 0 (off)..~1.0 typical, PERCENT clip per side | dull/flat contrast, blown/crushed levels — percentile stretch. Skips itself on flat/near-flat inputs instead of dividing by a near-zero span (see Gotchas) |
| `per_channel` | bool | `auto_levels` per-channel (also corrects a cast) vs. on shared luminance |
| `black` / `white` | 0..1 / 0..1 | manual levels endpoints (clipped) |
| `gamma` | >0, 1=neutral | midtone brightness; **>1 brightens** (Photoshop convention) |
| `curve` | `"in:out,in:out,..."` 0..255 | custom tone curve through named control points; **not sweepable via `fx_sweep.py --sweep`** (see Gotchas) |
| `scurve` | -1..1 | quick S-curve contrast (shadows down, highlights up) |
| `shadows` | 0..1 | shadow lift (luminance-masked gamma brighten) |
| `highlights` | 0..1 | highlight recovery (luminance-masked gamma darken) |
| `vibrance` | -1..1 | saturation boost weighted toward already-low-saturation pixels |
| `saturation` | 0..~2, 1=neutral | flat overall saturation (same math as `post_fx`) |
| `clarity` | -1..1 | local midtone contrast (unsharp-mask on luminance, protected at the extremes) |
| `sharpen` / `sharpen_radius` | 0..~2 / px | soft focus (unsharp mask) |

### `retouch.py` presets

| preset | recipe | when |
|--------|--------|------|
| `none` | all neutral | A/B baseline, or start from scratch with explicit flags |
| `fix` (default) | `auto_wb 0.5, auto_levels 0.3, clarity 0.08, sharpen 0.25` | general-purpose "clean it up" |
| `crisp` | `clarity 0.20, sharpen 0.6, scurve 0.12` | punchy, snappy contrast |
| `soft` | `denoise 0.25, shadows 0.15, clarity -0.10` | noisy/harsh source, gentler result |

### `post_fx.py` presets

| preset | recipe | mood |
|--------|--------|------|
| `none` | identity | A/B baseline |
| `film` (default) | warm split-tone, halation 0.22, vignette 0.18, grain 0.020 | classic warm/cool film look |
| `teal-orange` | cool shadows / warm highlights, halation 0.20, vignette 0.20 | cinematic |
| `muted` | saturation 0.82, softer halation/vignette, grain 0.024 | desaturated editorial |
| `warm` | strongest warm shift, halation 0.26 | golden-hour |

Scalar `post_fx.py` controls are validated before rendering: `temp`/`tint` are `-1..1`,
`contrast`/`saturation` are `0..2`, `halation`/`vignette` are `0..1`, and `grain` is `0..0.15`.
Split-tone RGB offsets must contain exactly three finite components in `-1..1`; `seed` is a
non-negative integer.

## Sweep workflow (`fx_sweep.py`)

`fx_sweep.py` renders one source image through the cartesian product of `--sweep param=v1,v2,...` axes
(a `retouch.Retouch` / `post_fx.Grade` field, or the literal `preset`), tiles every variant plus the
untouched original into one labeled contact sheet, and prints the sheet path. `--set param=value` fixes
a field across every variant without adding it to the grid or the filenames. Default `--preset none` —
a sweep explores deltas from neutral unless you name a starting look with `--preset`. Each fresh output
directory also gets `manifest.json` with the input hash, tool/preset, fixed values, sweep axes, seed,
variant mapping, exact effective config, exact sheet order, and SHA-256 for every variant, the
re-encoded original reference, and the contact sheet.

### Example 1 — single-axis: dial in shadow lift

```bash
python3 fx_sweep.py portrait.png --sweep shadows=0.0,0.2,0.4,0.6,0.8 --cols 6
```

Five `retouch.py` variants (plus the original) at increasing shadow lift, one row. Good for finding the
smallest lift that opens up the dark half of a frame without going flat/gray.

### Example 2 — two-axis grid: clarity × sharpen (3×3)

```bash
python3 fx_sweep.py portrait.png --sweep clarity=0.0,0.15,0.3 --sweep sharpen=0.0,0.3,0.6 \
    --set denoise=0.2 --cols 3
```

9 variants form the first 3×3 grid (clarity down the rows, sharpen across the columns); the untouched
original follows as a reference tile after the contiguous grid. `denoise` is fixed at 0.2 for every
variant so noise doesn't confound the read. Pick the cell with the local-contrast/sharpness balance you
want, not the strongest corner.

### Example 3 — `post_fx.py` preset shootout

```bash
python3 fx_sweep.py portrait_rt.png --tool post_fx --sweep preset=none,film,teal-orange,muted,warm --cols 5
```

Runs `retouch.py`'s already-clean output (`portrait_rt.png`) through every `post_fx` preset side by
side — the fastest way to pick a finishing look by eye instead of running each preset one at a time.

## The loop

1. **Sweep** — pick one or two axes that matter for the shot, run `fx_sweep.py`, get `<stem>_sweep/sheet.png`.
2. **Eyeball the sheet** — read the labeled grid, pick the cell (or the row/column) that looks right.
3. **Apply the chosen params directly** — re-run the underlying tool once with those exact values (not the sweep tool, which is for exploration, not final output):
   ```bash
   python3 retouch.py portrait.png --preset none --shadows 0.4 --clarity 0.15 --sharpen 0.3 --out portrait_rt.png
   ```
4. **Optional `post_fx.py` finish** — if the shot also wants a look, not just a fix, grade the corrected output:
   ```bash
   python3 post_fx.py portrait_rt.png --preset film --out portrait_final.png
   ```

Repeat the loop with `--tool post_fx` (step 1) once the correction is locked, if you want to sweep the
finish too (Example 3).

## Gotchas

- **`curve` can't be a `--sweep` axis.** A curve spec is itself comma-separated
  (`"0:0,128:145,255:255"`), which collides with `--sweep`'s own comma-separated value list. Fix a
  curve with `--set curve="0:0,128:145,255:255"` instead — `--set` takes one raw value, never split.
- **`--set` and `--sweep` on the same param is an error, not a silent override.** Fixing and sweeping
  the same field at once is ambiguous, so `fx_sweep.py` refuses it up front rather than letting one
  silently win.
- **Numeric controls are validated before rendering.** Both tools reject `nan` / `inf` and
  out-of-contract strengths. Unordered black/white points and non-positive gamma/radius values also
  fail with a clear message instead of writing a corrupt image.
- **Retouch output is no-clobber by default.** The input path is never a valid output; replacing a
  different existing output requires explicit `--overwrite`.
- **`post_fx.py` uses the same no-clobber contract.** It also rejects the input itself, hardlink
  aliases, and broken output symlinks; replacing a different existing output requires `--overwrite`.
- **Every sweep uses a fresh output directory.** An existing `--out-dir` is rejected so reruns cannot
  overwrite an earlier sheet or mix stale variants into a new experiment. Repeated `--sweep` or
  `--set` names are also rejected instead of silently keeping the last value. Rendering happens in a
  sibling staging directory and publishes through the platform's atomic no-replace rename, so errors,
  interrupts, and concurrent directory creation leave no partial canonical run.
- **`auto_levels` is a no-op on flat/near-flat images, on purpose.** If the percentile spread it finds
  is negligible next to the image's real dynamic range (a uniformly-blown or uniformly-crushed frame,
  or a near-flat frame with only a few outlier pixels), it leaves the image unchanged instead of
  dividing by a near-zero span and crushing everything to black/white. Recover a genuinely blown-out
  photo with `black`/`white`/`gamma` or `curve` instead — `auto_levels` is for images that already have
  real (if dull) spread to stretch.
- **Grid size is capped (`--max-combos`, default 36), never silently truncated.** A sweep bigger than
  the cap raises with the combination count; narrow an axis or raise `--max-combos` explicitly.
