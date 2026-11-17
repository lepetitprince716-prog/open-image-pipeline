#!/usr/bin/env python3
"""Privacy guard for a public repository, implemented with only the stdlib.

It looks for personal data -- LAN addresses, home paths, emails, credentials and
image-generation metadata -- in staged files, tracked files, git history, a commit
message or any path, and checks that the configured git identity is a noreply address.

A finding is reported as ``location: rule_id -- why``; the matched value is never
printed. Exit status: 0 clean, 1 findings, 2 usage or internal error. The guard fails
closed: a file it cannot read or decode is a finding, not a pass.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NamedTuple

EXIT_CLEAN, EXIT_FINDINGS, EXIT_ERROR = 0, 1, 2

PUBLIC_RULES = Path(__file__).resolve().with_name("privacy_rules.json")
PRIVATE_RULES_ENV = "AGENT_OVERSIGHT_PRIVATE_RULES"
REQUIRE_PRIVATE_ENV = "AGENT_OVERSIGHT_REQUIRE_PRIVATE"

DOCUMENT_LISTS = (
    "rules",
    "allow_literals",
    "allow_patterns",
    "allowed_emails",
    "allowed_binary_suffixes",
)
DOCUMENT_KEYS = frozenset({"version", "note", *DOCUMENT_LISTS})  # "note" is free text, ignored
RULE_REQUIRED = frozenset({"id", "pattern", "why"})
RULE_KEYS = RULE_REQUIRED | {"flags", "suffixes"}
FLAG_BITS = {"i": re.IGNORECASE}

SHORT_SHA = 12
OBJECT_ID = re.compile(rb"[0-9a-f]{40}|[0-9a-f]{64}")
IDENT_FIELDS = ("author", "committer", "tagger")
IDENT_EMAIL = re.compile(r"<([^<>]*)>$")

CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_TEXT_CHUNKS = (b"tEXt", b"zTXt", b"iTXt")
PNG_PIXEL_CHUNKS = (b"IDAT", b"fdAT")
GENERATION_KEYS = frozenset({"prompt", "workflow", "parameters", "comment", "description"})
MAX_INFLATED = 64 * 1024 * 1024

WHY_GENERATION = (
    "image carries generation metadata (prompt, workflow or parameters); "
    "strip it or keep the image out of the repository"
)
WHY_BINARY = (
    "binary file cannot be scanned; only image suffixes in allowed_binary_suffixes may be binary"
)
WHY_NOT_UTF8 = "file is not valid UTF-8, so it cannot be scanned"
WHY_UNREADABLE = "file could not be read, so it cannot be scanned"
WHY_BAD_IMAGE = "PNG chunks could not be parsed or decoded, so its metadata cannot be checked"
WHY_MISSING = "object is missing from the repository, so it cannot be scanned"
WHY_IDENTITY = (
    "email is not on the allowed_emails list; set your GitHub noreply address (Settings > Emails): "
    "git config user.email '<id>+<name>@users.noreply.github.com'"
)


class ScanError(Exception):
    """A usage or environment problem that stops the scan (exit status 2)."""


class Hit(NamedTuple):
    """One rule firing on one line of some content."""

    line: int
    rule_id: str
    why: str


class Finding(NamedTuple):
    """A hit with the place it was found; ``str()`` is the report line."""

    where: str
    line: int
    rule_id: str
    why: str

    def __str__(self) -> str:
        # A file name may hold a newline; escape it so one finding stays one line.
        where = CONTROL_CHARS.sub(lambda match: f"\\x{ord(match.group()):02x}", self.where)
        return f"{where}:{self.line}: {self.rule_id} -- {self.why}"


@dataclass(frozen=True)
class Rule:
    """A compiled deny rule, optionally limited to some file suffixes."""

    id: str
    regex: re.Pattern[str]
    why: str
    suffixes: tuple[str, ...]


@dataclass(frozen=True)
class RuleSet:
    """Everything the scanner needs from the merged rules documents."""

    rules: tuple[Rule, ...]
    allow: tuple[re.Pattern[str], ...]
    allowed_emails: tuple[re.Pattern[str], ...]
    binary_suffixes: frozenset[str]


# --- rules -------------------------------------------------------------------------


def read_document(path: Path) -> dict:
    """Read one rules file and check that its outer shape is right."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScanError(f"cannot read rules file {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ScanError(f'{path}: expected a JSON object with "version": 1')
    unknown = sorted(set(document) - DOCUMENT_KEYS)
    if unknown:
        raise ScanError(f"{path}: unknown key(s): {', '.join(unknown)}")
    for key in DOCUMENT_LISTS:
        if not isinstance(document.get(key, []), list):
            raise ScanError(f"{path}: {key} must be a list")
    return document


def compile_pattern(pattern: str, flags: int, where: str) -> re.Pattern[str]:
    """Compile one regex; an error names the entry, never the pattern text."""
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise ScanError(f"{where}: invalid pattern ({exc.msg})") from exc


def literal_pattern(literal: str) -> re.Pattern[str]:
    """Match a literal as a whole token: never glued to a word character at an alphanumeric edge."""
    head = r"(?<!\w)" if literal[0].isalnum() else ""
    tail = r"(?!\w)" if literal[-1].isalnum() else ""
    return re.compile(head + re.escape(literal) + tail)


def compile_rule(raw: object, where: str) -> Rule:
    """Validate and compile one entry of ``rules``."""
    if not isinstance(raw, dict) or not RULE_REQUIRED <= set(raw) <= RULE_KEYS:
        raise ScanError(f"{where}: needs id, pattern and why, and may add flags and suffixes")
    flags, suffixes = raw.get("flags", ""), raw.get("suffixes", [])
    required = (raw["id"], raw["pattern"], raw["why"], flags)
    if not all(isinstance(value, str) for value in required):
        raise ScanError(f"{where}: id, pattern, why and flags must be strings")
    if set(flags) - set(FLAG_BITS) or not isinstance(suffixes, list):
        raise ScanError(f"{where}: flags allows only i, suffixes must be a list")
    bits = 0
    for letter in flags:
        bits |= FLAG_BITS[letter]
    regex = compile_pattern(raw["pattern"], bits, where)
    return Rule(raw["id"], regex, raw["why"], tuple(str(item).lower() for item in suffixes))


def compile_rules(document: dict, origin: str) -> RuleSet:
    """Validate a rules document and compile every pattern in it."""
    for key in DOCUMENT_LISTS[1:]:
        if not all(isinstance(item, str) and item for item in document.get(key, [])):
            raise ScanError(f"{origin}: {key} must be a list of non-empty strings")
    rules = [
        compile_rule(raw, f"{origin}: rules entry {number}")
        for number, raw in enumerate(document.get("rules", []), start=1)
    ]
    allow = [literal_pattern(item) for item in document.get("allow_literals", [])]
    allow += [
        compile_pattern(item, 0, f"{origin}: allow_patterns entry {number}")
        for number, item in enumerate(document.get("allow_patterns", []), start=1)
    ]
    emails = [
        compile_pattern(item, 0, f"{origin}: allowed_emails entry {number}")
        for number, item in enumerate(document.get("allowed_emails", []), start=1)
    ]
    suffixes = frozenset(item.lower() for item in document.get("allowed_binary_suffixes", []))
    return RuleSet(tuple(rules), tuple(allow), tuple(emails), suffixes)


def merge_rulesets(first: RuleSet, second: RuleSet) -> RuleSet:
    """Concatenate two rule sets, as the rules files promise."""
    return RuleSet(
        first.rules + second.rules,
        first.allow + second.allow,
        first.allowed_emails + second.allowed_emails,
        first.binary_suffixes | second.binary_suffixes,
    )


def load_ruleset(path: Path) -> RuleSet:
    """Read, validate and compile one rules file."""
    return compile_rules(read_document(path), str(path))


def private_rules_path() -> Path:
    """Where the owner's private rules live (``AGENT_OVERSIGHT_PRIVATE_RULES`` overrides)."""
    override = os.environ.get(PRIVATE_RULES_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "agent-oversight" / "private_rules.json"


def load_rules(public_path: Path, use_private: bool) -> tuple[RuleSet, str]:
    """Load the public rules, merge the private file if present, and say what happened."""
    rules = load_ruleset(public_path)
    if not use_private:
        return rules, "private rules skipped (--no-private)"
    path = private_rules_path()
    if not path.is_file():
        if os.environ.get(REQUIRE_PRIVATE_ENV) == "1":
            raise ScanError(f"{REQUIRE_PRIVATE_ENV}=1 but there is no private rules file at {path}")
        return rules, "private rules not loaded (0 rules)"
    private = load_ruleset(path)
    return merge_rulesets(rules, private), f"private rules loaded ({len(private.rules)} rules)"


# --- matching ----------------------------------------------------------------------


def blank_allowed(line: str, rules: RuleSet) -> str:
    """Replace every allowed literal and allow-pattern match with spaces of equal length."""
    for pattern in rules.allow:
        line = pattern.sub(lambda match: " " * len(match.group()), line)
    return line


def scan_text(text: str, rules: RuleSet, suffix: str) -> list[Hit]:
    """Run every applicable rule over each line; a rule fires at most once per line."""
    applicable = [rule for rule in rules.rules if not rule.suffixes or suffix in rule.suffixes]
    hits = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        cleaned = blank_allowed(line, rules)
        fired = (rule for rule in applicable if rule.regex.search(cleaned))
        hits += [Hit(number, rule.id, rule.why) for rule in fired]
    return hits


def to_findings(where: str, hits: Iterable[Hit]) -> list[Finding]:
    """Attach a location to hits."""
    return [Finding(where, hit.line, hit.rule_id, hit.why) for hit in hits]


def scan_values(values: Iterable[str], rules: RuleSet, suffix: str) -> list[Hit]:
    """Scan decoded metadata strings; an image reports each rule once, on line 1."""
    found: dict[str, Hit] = {}
    for value in values:
        for hit in scan_text(value, rules, suffix):
            found.setdefault(hit.rule_id, Hit(1, hit.rule_id, hit.why))
    return list(found.values())


# --- images ------------------------------------------------------------------------


def inflate(raw: bytes) -> bytes:
    """Decompress a zlib stream, refusing truncated streams and decompression bombs."""
    inflater = zlib.decompressobj()
    out = inflater.decompress(raw, MAX_INFLATED + 1)
    if len(out) > MAX_INFLATED or not inflater.eof:
        raise ValueError("compressed text is truncated or too large")
    return out


def decode_text_chunk(kind: bytes, payload: bytes) -> list[str]:
    """Return ``[keyword, text, ...]`` of a tEXt, zTXt or iTXt chunk; ValueError if malformed."""
    key, sep, rest = payload.partition(b"\0")
    if not sep:
        raise ValueError("text chunk has no keyword separator")
    fields = [key.decode("latin-1")]
    if kind == b"tEXt":
        fields.append(rest.decode("latin-1"))
    elif kind == b"zTXt":
        if rest[:1] != b"\0":
            raise ValueError("unknown compression method")
        fields.append(inflate(rest[1:]).decode("latin-1"))
    else:
        if len(rest) < 2:
            raise ValueError("short iTXt chunk")
        compressed, method = rest[0], rest[1]
        language, sep_a, rest = rest[2:].partition(b"\0")
        translated, sep_b, text = rest.partition(b"\0")
        if not (sep_a and sep_b) or (compressed and method != 0):
            raise ValueError("malformed iTXt chunk")
        if compressed:
            text = inflate(text)
        fields += [language.decode("latin-1"), translated.decode("utf-8"), text.decode("utf-8")]
    return fields


def raw_views(data: bytes) -> list[str]:
    """Latin-1 text of raw bytes, plus a NUL-free view so UTF-16 strings read as ASCII."""
    views = [data.decode("latin-1")]
    if b"\0" in data:
        views.append(data.replace(b"\0", b"").decode("latin-1"))
    return views


def has_class_type(text: str) -> bool:
    """True if the text holds a ComfyUI node graph (``"class_type"``)."""
    return '"class_type"' in text


def looks_generated(text: str) -> bool:
    """True if raw image bytes carry a ComfyUI graph or an A1111-style parameters block."""
    a1111 = "Steps: " in text and "Sampler: " in text
    return has_class_type(text) or a1111 or "parameters" in text


def read_png(data: bytes) -> tuple[list[str], bool]:
    """Walk a PNG: decode text chunks, read other chunks (not pixels) raw.

    Returns the strings to scan and whether generation metadata was seen. Raises
    ValueError, struct.error or zlib.error when the file cannot be parsed.
    """
    values: list[str] = []
    generated = False
    leftover = bytearray()
    pos, ended = len(PNG_SIGNATURE), False
    while pos < len(data) and not ended:
        length, kind = struct.unpack_from(">I4s", data, pos)
        start = pos + 8
        end = start + length + 4  # payload plus CRC
        if end > len(data):
            raise ValueError("truncated chunk")
        if kind in PNG_TEXT_CHUNKS:
            fields = decode_text_chunk(kind, data[start : start + length])
            values += fields
            generated |= fields[0].lower() in GENERATION_KEYS or any(map(has_class_type, fields))
        elif kind not in PNG_PIXEL_CHUNKS:
            leftover += data[start : start + length]
        pos, ended = end, kind == b"IEND"
    if not ended:
        raise ValueError("no IEND chunk")
    leftover += data[pos:]
    views = raw_views(bytes(leftover))
    return values + views, generated or any(map(has_class_type, views))


def scan_image(data: bytes, rules: RuleSet, suffix: str) -> list[Hit]:
    """Scan an image with an allowed suffix; PNG text is decoded, other formats are read raw."""
    if data.startswith(PNG_SIGNATURE):
        try:
            values, generated = read_png(data)
        except (ValueError, struct.error, zlib.error):
            return [Hit(1, "undecodable", WHY_BAD_IMAGE)]
    else:
        values = raw_views(data)
        generated = any(map(looks_generated, values))
    hits = scan_values(values, rules, suffix)
    if generated:
        hits.append(Hit(1, "generation_metadata", WHY_GENERATION))
    return hits


def scan_bytes(name: str, data: bytes, rules: RuleSet) -> list[Hit]:
    """Classify a file's bytes (allowed image, binary or text) and scan them."""
    suffix = Path(name).suffix.lower()
    if suffix in rules.binary_suffixes:
        return scan_image(data, rules, suffix)
    binary = [Hit(1, "binary_not_allowed", WHY_BINARY)] if b"\0" in data else []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return binary + [Hit(1, "undecodable", WHY_NOT_UTF8)]
    return binary or scan_text(text, rules, suffix)


# --- git ---------------------------------------------------------------------------


def git(args: list[str], cwd: Path) -> bytes:
    """Run git with an argument list and return stdout; any failure is a ScanError."""
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False)
    except OSError as exc:
        raise ScanError(f"cannot run git: {exc}") from exc
    if done.returncode != 0:
        detail = done.stderr.decode("utf-8", "replace").strip()
        raise ScanError(f"git {args[0]} failed: {detail}")
    return done.stdout


def repo_root() -> Path:
    """The top-level directory of the repository containing the current directory."""
    return Path(os.fsdecode(git(["rev-parse", "--show-toplevel"], Path.cwd()).strip()))


def split_names(listing: bytes) -> Iterator[str]:
    """Decode a NUL-separated name list from git."""
    return (os.fsdecode(raw) for raw in listing.split(b"\0") if raw)


def read_worktree(path: Path) -> bytes:
    """A file's bytes; a symlink yields its target text, which is what git stores."""
    if path.is_symlink():
        return os.fsencode(os.readlink(path))
    return path.read_bytes()


def scan_worktree_file(where: str, path: Path, rules: RuleSet) -> list[Finding]:
    """Scan one working-tree file; an unreadable file is itself a finding."""
    try:
        data = read_worktree(path)
    except OSError:
        return [Finding(where, 1, "undecodable", WHY_UNREADABLE)]
    return to_findings(where, scan_bytes(path.name, data, rules))


def scan_staged(root: Path, rules: RuleSet) -> list[Finding]:
    """Scan the index version of every file about to be committed."""
    listing = git(["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"], root)
    findings = []
    for name in split_names(listing):
        try:
            data = git(["show", f":./{name}"], root)
        except ScanError:
            findings.append(Finding(name, 1, "undecodable", WHY_UNREADABLE))
        else:
            findings += to_findings(name, scan_bytes(name, data, rules))
    return findings


def scan_tracked(root: Path, rules: RuleSet) -> list[Finding]:
    """Scan every tracked file as it is in the working tree."""
    findings = []
    for name in split_names(git(["ls-files", "-z"], root)):
        findings += scan_worktree_file(name, root / name, rules)
    return findings


def check_ident(where: str, ident: str, rules: RuleSet) -> list[Finding]:
    """Check a ``Name <email> ...`` identity: strict email list first, then the normal rules."""
    ident = ident[: ident.rfind(">") + 1]
    match = IDENT_EMAIL.search(ident)
    findings = []
    if match is None or not any(p.fullmatch(match.group(1)) for p in rules.allowed_emails):
        findings.append(Finding(where, 1, "email", WHY_IDENTITY))
    return findings + to_findings(where, scan_text(ident, rules, ""))


def scan_identity(rules: RuleSet) -> list[Finding]:
    """Check the author and committer identities git would use for the next commit."""
    findings = []
    for var, role in (("GIT_AUTHOR_IDENT", "author"), ("GIT_COMMITTER_IDENT", "committer")):
        ident = git(["var", var], Path.cwd()).decode("utf-8", "replace").strip()
        findings += check_ident(f"identity {role}", ident, rules)
    return findings


def scan_commit(kind: str, oid: str, data: bytes, rules: RuleSet) -> list[Finding]:
    """Scan a commit or tag object: identities by the email list, the message by the rules."""
    head, _, body = data.decode("utf-8", "replace").partition("\n\n")
    where = f"{kind} {oid[:SHORT_SHA]}"
    findings = []
    for line in head.split("\n"):
        field, _, value = line.partition(" ")
        if field in IDENT_FIELDS:
            findings += check_ident(f"{where} {field}", value, rules)
    return findings + to_findings(f"{where} message", scan_text(body, rules, ""))


def read_batch(stream: BinaryIO) -> Iterator[tuple[str, str, bytes]]:
    """Yield ``(id, type, content)`` from ``git cat-file --batch``; type is "missing" if lost."""
    for header in iter(stream.readline, b""):
        oid, kind, *size = header.decode("ascii").split()
        if kind == "missing":
            yield oid, kind, b""
            continue
        data = stream.read(int(size[0]))
        if len(data) != int(size[0]):
            raise ScanError("git cat-file ended early")
        stream.read(1)  # the newline after the content
        yield oid, kind, data


def scan_history(root: Path, rules: RuleSet) -> list[Finding]:
    """Scan every blob, commit and tag reachable from any ref; no commits means clean."""
    if git(["rev-parse", "--is-shallow-repository"], root).strip() == b"true":
        raise ScanError("shallow clone: history is incomplete, run git fetch --unshallow")
    paths: dict[str, str] = {}
    for line in git(["rev-list", "--objects", "--all"], root).split(b"\n"):
        oid, _, path = line.partition(b" ")
        if OBJECT_ID.fullmatch(oid):
            paths[oid.decode("ascii")] = os.fsdecode(path)
    if not paths:
        return []
    findings: list[Finding] = []
    with tempfile.TemporaryFile() as feed:
        feed.write("".join(f"{oid}\n" for oid in paths).encode("ascii"))
        feed.seek(0)
        try:
            proc = subprocess.Popen(
                ["git", "cat-file", "--batch"],
                cwd=root,
                stdin=feed,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise ScanError(f"cannot run git: {exc}") from exc
        with proc:
            for oid, kind, data in read_batch(proc.stdout):
                short = oid[:SHORT_SHA]
                if kind == "blob":
                    name = paths[oid]
                    findings += to_findings(f"blob {short} {name}", scan_bytes(name, data, rules))
                elif kind in ("commit", "tag"):
                    findings += scan_commit(kind, oid, data, rules)
                elif kind == "missing":
                    findings.append(Finding(f"object {short}", 1, "undecodable", WHY_MISSING))
    if proc.returncode != 0:
        raise ScanError("git cat-file failed")
    return findings


# --- files and messages ------------------------------------------------------------


def scan_message(path: Path, rules: RuleSet) -> list[Finding]:
    """Scan a commit message file; lines starting with ``#`` are comments."""
    try:
        text = path.read_bytes().decode("utf-8", "replace")
    except OSError as exc:
        raise ScanError(f"cannot read message file {path}: {exc.strerror}") from exc
    kept = "\n".join("" if line.startswith("#") else line for line in text.split("\n"))
    return to_findings(path.as_posix(), scan_text(kept, rules, ""))


def walk_error(exc: OSError) -> None:
    """os.walk skips unlistable directories silently; make that an error instead."""
    raise ScanError(f"cannot list {exc.filename}: {exc.strerror}")


def iter_files(paths: list[Path]) -> Iterator[Path]:
    """Yield files under the given paths in a stable order; symlinks are not followed."""
    for top in paths:
        if top.is_dir() and not top.is_symlink():
            for folder, dirs, files in os.walk(top, onerror=walk_error):
                links = [d for d in dirs if os.path.islink(os.path.join(folder, d))]
                dirs[:] = sorted(d for d in dirs if d != ".git" and d not in links)
                for name in sorted(files + links):
                    if name != ".git":
                        yield Path(folder, name)
        elif top.exists() or top.is_symlink():
            yield top
        else:
            raise ScanError(f"no such path: {top}")


def scan_paths(paths: list[Path], rules: RuleSet) -> list[Finding]:
    """Scan files and directories given on the command line."""
    findings = []
    for path in iter_files(paths):
        findings += scan_worktree_file(path.as_posix(), path, rules)
    return findings


# --- command line ------------------------------------------------------------------


def unique(findings: Iterable[Finding]) -> list[Finding]:
    """Drop repeats of one rule at one place, keeping the first wording."""
    seen: set[tuple[str, int, str]] = set()
    kept = []
    for finding in findings:
        if finding[:3] not in seen:
            seen.add(finding[:3])
            kept.append(finding)
    return kept


def run_scans(args: argparse.Namespace, rules: RuleSet) -> list[Finding]:
    """Run every requested mode and collect the findings."""
    findings: list[Finding] = []
    if args.staged or args.tracked or args.history:
        root = repo_root()
        if args.staged:
            findings += scan_staged(root, rules)
        if args.tracked:
            findings += scan_tracked(root, rules)
        if args.history:
            findings += scan_history(root, rules)
    if args.identity:
        findings += scan_identity(rules)
    if args.message:
        findings += scan_message(args.message, rules)
    if args.paths:
        findings += scan_paths(args.paths, rules)
    return findings


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(
        prog="privacy_scan.py",
        description="Scan for personal data before it reaches a public repository. "
        "Matched values are never printed; a finding is 'location: rule_id -- why'.",
        epilog="exit status: 0 clean, 1 findings, 2 usage or internal error",
        allow_abbrev=False,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        metavar="PATH",
        help="files or directories to scan (recursive, skips .git)",
    )
    parser.add_argument(
        "--staged", action="store_true", help="scan staged content (the index)"
    )
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="scan every tracked file as it is in the working tree",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="scan every blob, commit and tag reachable from any ref",
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        help="require noreply author and committer emails in the git identity",
    )
    parser.add_argument(
        "--message",
        type=Path,
        metavar="FILE",
        help="scan a commit message file (lines starting with # are ignored)",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        metavar="FILE",
        default=PUBLIC_RULES,
        help="use this public rules file instead of privacy_rules.json",
    )
    parser.add_argument(
        "--no-private", action="store_true", help="do not load the private rules file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the scans, print findings; return the exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = [args.staged, args.tracked, args.history, args.identity, args.message, args.paths]
    if not any(modes):
        parser.error("nothing to scan: give a mode (see --help) or a PATH")
    sys.stdout.reconfigure(errors="backslashreplace")
    try:
        rules, note = load_rules(args.rules, use_private=not args.no_private)
        print(f"privacy_scan: {note}", file=sys.stderr)
        findings = unique(run_scans(args, rules))
    except ScanError as exc:
        print(f"privacy_scan: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # keep the exit-status contract: a crash is 2, never 1
        print(f"privacy_scan: internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    for finding in findings:
        print(finding)
    sys.stdout.flush()  # keep the summary after the findings when both streams share a terminal
    if findings:
        print(f"privacy_scan: {len(findings)} finding(s)", file=sys.stderr)
        return EXIT_FINDINGS
    print("privacy_scan: clean", file=sys.stderr)
    return EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
