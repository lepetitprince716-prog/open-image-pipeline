#!/usr/bin/env python3
"""Wildcard prompt engine: {a|b|c} variant selection + __file__ wildcards for batch prompt exploration.

Syntax (the common A1111 / ComfyUI "dynamic prompts" dialect):
  {a|b|c}        pick one option at random
  {3::a|1::b}    weighted pick (weight before '::', default weight 1)
  {2$$a|b|c}     pick exactly 2 distinct options, joined with ', '
  {1-2$$a|b|c}   pick between 1 and 2 distinct options
  __name__       file wildcard: pick a random line from wildcards/name.txt
  __sub/name__   subdir file wildcard: wildcards/sub/name.txt

Options and wildcard-file lines may themselves contain {..} / __..__ (resolved recursively, with a
depth guard against cycles). A plain prompt with no wildcard syntax expands to itself unchanged, so
routing every prompt through expand() is safe. A '{...}' group that has no '|' and no '$$' directive
is left literal, so ordinary braces in a prompt survive.

CLI:
  python3 scripts/wildcard.py "1girl, {red|blue} dress, __pose__" --n 5 --seed 42
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

DEFAULT_WILDCARD_DIR = Path(__file__).resolve().parent.parent / "wildcards"
MAX_DEPTH = 50
_FILE_RE = re.compile(r"__([A-Za-z0-9_./-]+?)__")
_COUNT_RE = re.compile(r"\s*(\d+)(?:-(\d+))?\$\$(.*)$", re.DOTALL)
_WEIGHT_RE = re.compile(r"\s*(\d+(?:\.\d+)?)::(.*)$", re.DOTALL)


class WildcardError(ValueError):
    """Raised for missing/empty wildcard files, traversal escapes, or runaway recursion."""


class WildcardEngine:
    """Expands wildcard templates. File contents are cached for the engine's lifetime."""

    def __init__(self, wildcard_dir: str | Path | None = None) -> None:
        self.dir = Path(wildcard_dir) if wildcard_dir else DEFAULT_WILDCARD_DIR
        self._cache: dict[str, list[str]] = {}

    # --- file wildcards -------------------------------------------------------------------------
    def _load_file(self, name: str) -> list[str]:
        if name in self._cache:
            return self._cache[name]
        path = self.dir / (name + ".txt")
        try:
            path.resolve().relative_to(self.dir.resolve())
        except ValueError as exc:
            raise WildcardError(f"wildcard path escapes the wildcard dir: {name!r}") from exc
        if not path.is_file():
            raise WildcardError(f"wildcard file not found: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        opts = [
            s.split(" # ", 1)[0].strip()   # strip inline metadata markers ("text # motion:seated")
            for s in lines
            if s.strip() and not s.lstrip().startswith("#")
        ]
        if not opts:
            raise WildcardError(f"wildcard file is empty: {path}")
        self._cache[name] = opts
        return opts

    # --- expansion ------------------------------------------------------------------------------
    def expand(self, template: str, seed: int | None = None) -> str:
        """Resolve one random expansion of the template (deterministic for a given seed)."""
        return self._resolve(template, random.Random(seed), 0)

    def expand_many(self, template: str, n: int, seed: int | None = None) -> list[str]:
        """Resolve n expansions. With a seed the batch is reproducible and items vary independently."""
        driver = random.Random(seed)
        out: list[str] = []
        for _ in range(n):
            item_seed = driver.randrange(2**31) if seed is not None else None
            out.append(self.expand(template, item_seed))
        return out

    def _resolve(self, s: str, rng: random.Random, depth: int) -> str:
        if depth > MAX_DEPTH:
            raise WildcardError(f"wildcard recursion exceeded {MAX_DEPTH} (cycle?)")
        out: list[str] = []
        i, n = 0, len(s)
        while i < n:
            ch = s[i]
            if ch == "{":
                inner, j = self._match_brace(s, i)
                directive = self._parse_group(inner) if inner is not None else None
                if directive is None:  # unmatched, or not a wildcard group -> keep literal
                    out.append(ch)
                    i += 1
                    continue
                chosen = self._pick(directive, rng)
                out.append(directive["sep"].join(self._resolve(opt, rng, depth + 1) for opt in chosen))
                i = j
            elif ch == "_" and s.startswith("__", i):
                m = _FILE_RE.match(s, i)
                if m:
                    line = rng.choice(self._load_file(m.group(1)))
                    out.append(self._resolve(line, rng, depth + 1))
                    i = m.end()
                else:
                    out.append(ch)
                    i += 1
            else:
                out.append(ch)
                i += 1
        return "".join(out)

    # --- parsing helpers ------------------------------------------------------------------------
    @staticmethod
    def _match_brace(s: str, i: int) -> tuple[str | None, int]:
        """Return (inner, index_after_close) for the brace group at s[i], tracking nesting."""
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
        return None, i

    @staticmethod
    def _split_pipes(inner: str) -> list[str]:
        """Split on '|' only at brace-depth 0, so nested groups stay intact."""
        parts: list[str] = []
        buf: list[str] = []
        depth = 0
        for ch in inner:
            if ch == "{":
                depth += 1
                buf.append(ch)
            elif ch == "}":
                depth -= 1
                buf.append(ch)
            elif ch == "|" and depth == 0:
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        parts.append("".join(buf))
        return parts

    def _parse_group(self, inner: str) -> dict | None:
        """Parse a brace group's directive. Return None if it is not a wildcard group (keep literal)."""
        count = _COUNT_RE.match(inner)
        body = count.group(3) if count else inner
        raw_opts = self._split_pipes(body)
        if count is None and len(raw_opts) == 1:
            return None  # no '|' and no '$$' -> ordinary braces, not a wildcard
        lo = int(count.group(1)) if count else 1
        hi = int(count.group(2)) if count and count.group(2) else lo
        options: list[str] = []
        weights: list[float] = []
        for opt in raw_opts:
            wm = _WEIGHT_RE.match(opt)
            weights.append(float(wm.group(1)) if wm else 1.0)
            options.append(wm.group(2) if wm else opt)
        return {"options": options, "weights": weights, "lo": lo, "hi": hi, "sep": ", "}

    @staticmethod
    def _pick(directive: dict, rng: random.Random) -> list[str]:
        """Weighted sample without replacement of between lo and hi options."""
        options, weights = directive["options"], directive["weights"]
        k = max(0, min(rng.randint(directive["lo"], directive["hi"]), len(options)))
        pool = list(range(len(options)))
        chosen: list[str] = []
        for _ in range(k):
            total = sum(weights[p] for p in pool)
            target = rng.uniform(0, total)
            acc = 0.0
            for pi, p in enumerate(pool):
                acc += weights[p]
                if target <= acc:
                    chosen.append(options[p])
                    pool.pop(pi)
                    break
            else:  # float drift fallthrough
                chosen.append(options[pool[-1]])
                pool.pop()
        return chosen


def expand(template: str, seed: int | None = None, wildcard_dir: str | Path | None = None) -> str:
    """Module-level convenience: one expansion with a fresh engine (re-reads files each call)."""
    return WildcardEngine(wildcard_dir).expand(template, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand a wildcard prompt template")
    ap.add_argument("template", help="prompt template with {a|b} and __file__ wildcards")
    ap.add_argument("--n", type=int, default=1, help="how many expansions to print")
    ap.add_argument("--seed", type=int, default=None, help="base seed (reproducible batch)")
    ap.add_argument("--wildcard-dir", default=None, help="dir holding __name__.txt files (default: ./wildcards)")
    args = ap.parse_args()
    engine = WildcardEngine(args.wildcard_dir)
    for line in engine.expand_many(args.template, args.n, args.seed):
        print(line)


if __name__ == "__main__":
    main()
