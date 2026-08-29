# Wildcard / negative tag lint — booru_wildcard_distill.py

Checked against booru_distill snapshot: 304 alias edges, 1542 tag cards.
Actionable: 4 alias hits, 1 weak-tag hits. (128 segments unknown to the local card snapshot — freeform phrases are fine, tags just steer harder; coverage is only 404 cards so absence is NOT a warning.)

| origin | segment | issue | detail |
|---|---|---|---|
| wildcards/camera.txt | `face focus` | WEAK | post_count=0 < 200 |
| wildcards/pose.txt | `hand on hip` | ALIAS | canonical: `hand on own hip` |
| wildcards/pose.txt | `looking over shoulder` | ALIAS | canonical: `looking back` |
| wildcards/pose.txt | `lying down` | ALIAS | canonical: `lying` |
| wildcards/pose.txt | `viewed from above` | ALIAS | canonical: `from above` |

Fixes are applied by hand (curated assets stay human-owned).