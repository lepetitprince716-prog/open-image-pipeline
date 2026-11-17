# Legacy Harem manifest snapshot

Metadata-only copies of the two preserved legacy Harem analysis directories, taken from
`outputs/_analysis/` on 2026-07-26. `tests/test_app_experiments.py` rebuilds both runs from
these files so the legacy adapters stay covered on a clean checkout; before this snapshot the
test read the working tree directly and found nothing in CI.

- `harem_complex_dimensions_2026-07-26/` — `harem-complex-prompt-benchmark-manifest-v1`
  (24 outputs / 4 dimensions / 3 arms / 2 seeds) plus its `benchmark_spec.json` and
  `scoring.json` support files.
- `harem_miaomiao15_vs_8step_2026-07-26/` — `loramake.harem_ab.v1` (4 runs × 2 seeds,
  no automatic scores).

The rendered PNGs stay out of git. The test regenerates a small placeholder for every asset
path and patches the manifest's `bytes` / `sha256` to match, so the adapters still verify the
digests they are handed. Support files are copied verbatim, so the `sha256` the manifest
records for them must stay valid — re-copy the manifest and its support files together if
this snapshot is ever refreshed.
