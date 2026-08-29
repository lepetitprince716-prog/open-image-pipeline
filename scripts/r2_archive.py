"""Archive render outputs to a Cloudflare R2 bucket, under a hard free-tier quota.

Render outputs are PNG-heavy (~2.3 MB per image), so they are transcoded to WebP
before upload; measured on a sample of real outputs, q90 shrinks them ~10x.

Transcoded files keep their original extension and gain a `.webp` one on top
(`portrait.png` -> `portrait.png.webp`), so an image and a same-stem thumbnail never
collide in the staging tree. `build_jobs` asserts uniqueness anyway.

The quota gate is fail-closed: if the current bucket usage cannot be determined,
the upload is refused rather than attempted.

Uploads go through the AWS CLI against R2's S3-compatible endpoint, which gives
concurrent transfers and incremental sync. Credentials stay with the CLI and are
never read, logged, or echoed by this script; the pre-flight check only asks
whether some source exists — --profile, AWS_PROFILE, the access-key variables, or
a shared credentials file. Sources visible only to the CLI's own config
resolution (credential_process, role or web-identity entries in ~/.aws/config)
are not detected, so those setups need an explicit --profile.

Usage:
    python scripts/r2_archive.py status
    python scripts/r2_archive.py plan --src outputs/_gallery
    python scripts/r2_archive.py push --src outputs/_gallery --prefix gallery
    python scripts/r2_archive.py push --src outputs/_gallery --prefix gallery --apply
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ACCOUNT_ID = "e52b948ad550ac6418ca680ce6c26171"
DEFAULT_BUCKET = "loramake-archive"
STAGING_DIR = REPO_ROOT / "outputs" / "_r2_staging"

# Cloudflare quotes the R2 free tier in decimal GB, so the ceiling is computed the
# same way. Using GiB here would silently make the gate ~7% looser than advertised.
GB = 1000**3
FREE_TIER_BYTES = 10 * GB
# Refuse to cross 80% of the free tier. The remaining headroom absorbs the lag
# between an upload finishing and R2 usage reporting catching up.
QUOTA_CEILING_BYTES = int(FREE_TIER_BYTES * 0.8)

WEBP_QUALITY = 90
TRANSCODE_SUFFIXES = {".png", ".jpg", ".jpeg"}
COPY_SUFFIXES = {".webp", ".mp4", ".json", ".txt", ".yaml", ".yml"}


# Video goes into COPY_SUFFIXES, so it uploads at full size while stills shrink 3-10x.
# The quota gate below still catches a genuine breach, but it only fires at 8 GB; this
# threshold catches the shape of mistake that gets there in one command — a directory of
# video_finish.py masters (ProRes 422 LT, ~350 MB for 20s of 1080p) swept in by a --src
# that was meant to pick up probe clips a thousand times smaller.
LARGE_MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
LARGE_MEDIA_WARN_BYTES = 1_000_000_000


@dataclass
class StagedFile:
    source: Path
    staged: Path
    source_bytes: int
    staged_bytes: int


def human(num_bytes: float) -> str:
    # Decimal units throughout, to match how Cloudflare states the quota.
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1000 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1000
    return f"{num_bytes:.1f} GB"


def endpoint_url() -> str:
    account_id = os.environ.get("R2_ACCOUNT_ID", DEFAULT_ACCOUNT_ID)
    return f"https://{account_id}.r2.cloudflarestorage.com"


def bucket_name() -> str:
    return os.environ.get("R2_BUCKET", DEFAULT_BUCKET)


def aws_base_command(profile: str | None) -> list[str]:
    # R2 ignores the region but the AWS CLI refuses to sign without one.
    command = ["aws", "--endpoint-url", endpoint_url(), "--region", "auto"]
    if profile:
        command += ["--profile", profile]
    return command


def has_credentials(profile: str | None) -> bool:
    """Whether the AWS CLI has some credential source to work from.

    Checking only the two access-key variables would refuse to run for anyone
    configured the ordinary way, through AWS_PROFILE or the shared credentials
    file. This is a pre-flight courtesy, not an authorisation check — the CLI
    still rejects anything invalid.
    """
    if profile or os.environ.get("AWS_PROFILE"):
        return True
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True
    # An empty AWS_SHARED_CREDENTIALS_FILE is unset, not a path — Path("") is Path("."),
    # which would swallow the default-location fallback.
    override = os.environ.get("AWS_SHARED_CREDENTIALS_FILE")
    # expanduser because the CLI expands a "~/..." override; not doing so would read
    # it as a relative path and refuse a machine that is in fact configured.
    shared = Path(override).expanduser() if override else Path.home() / ".aws" / "credentials"
    return shared.is_file()


def bucket_usage(profile: str | None, prefix: str | None = None) -> tuple[int, int]:
    """Return (total_bytes, object_count) for the bucket, or for one prefix.

    Raises RuntimeError on any failure so callers can fail closed.
    """
    target = f"s3://{bucket_name()}"
    if prefix:
        target = f"{target}/{prefix}"
    command = aws_base_command(profile) + [
        "s3",
        "ls",
        target,
        "--recursive",
        "--summarize",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not reach R2 to read bucket usage: {exc}") from exc

    if result.returncode != 0:
        raise RuntimeError(f"aws s3 ls failed (exit {result.returncode}): {result.stderr.strip()}")

    total_bytes = None
    object_count = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Total Size:"):
            total_bytes = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("Total Objects:"):
            object_count = int(stripped.split(":", 1)[1].strip())

    if total_bytes is None or object_count is None:
        raise RuntimeError("aws s3 ls did not report a usage summary; refusing to guess")
    return total_bytes, object_count


def validate_prefix(prefix: str) -> str:
    """Reject any prefix that would place the staging root outside STAGING_DIR.

    `Path / prefix` silently discards the left side for an absolute prefix and
    walks upward for one containing `..`; prune_staging unlinks everything under
    whatever root it is handed, so this has to be checked before either runs.
    """
    if not prefix or prefix != prefix.strip():
        raise ValueError("prefix must be a non-empty string without surrounding whitespace")

    candidate = Path(prefix)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"prefix must be a relative path inside the staging tree: {prefix!r}")

    # No component may be a symlink, not just the last one. A link anywhere along the
    # path lets prune delete through it and sync upload through it, and resolve()
    # below cannot catch one pointing at another prefix inside the tree.
    walked = STAGING_DIR
    for part in candidate.parts:
        walked = walked / part
        if walked.is_symlink():
            raise ValueError(f"staging path {walked} is a symlink; refusing prefix {prefix!r}")

    staging_root = STAGING_DIR.resolve()
    root = (STAGING_DIR / prefix).resolve()
    if root == staging_root:
        # "." and "./." land here: inside the tree, but naming the whole of it, which
        # would make prune_staging sweep every prefix instead of one. ("a/." is fine —
        # pathlib folds it to "a".)
        raise ValueError(f"prefix must name a subdirectory, not the staging root: {prefix!r}")
    if not root.is_relative_to(staging_root):
        raise ValueError(f"prefix escapes the staging tree: {prefix!r} -> {root}")
    return prefix


def classify(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in TRANSCODE_SUFFIXES:
        return "transcode"
    if suffix in COPY_SUFFIXES:
        return "copy"
    return None


def stage_one(job: tuple[str, str, str]) -> tuple[str, str, int, int] | None:
    """Worker: transcode or copy a single file into the staging tree."""
    source_str, staged_str, action = job
    source = Path(source_str)
    staged = Path(staged_str)
    staged.parent.mkdir(parents=True, exist_ok=True)

    source_bytes = source.stat().st_size
    if staged.exists() and staged.stat().st_mtime >= source.stat().st_mtime:
        return source_str, staged_str, source_bytes, staged.stat().st_size

    if action == "transcode":
        try:
            with Image.open(source) as image:
                # WebP carries alpha; flattening to RGB would black out transparent art.
                transparent = image.mode in ("RGBA", "LA") or (
                    image.mode == "P" and "transparency" in image.info
                )
                mode = "RGBA" if transparent else "RGB"
                image.convert(mode).save(staged, format="WEBP", quality=WEBP_QUALITY, method=6)
        except OSError as exc:
            # Drop any older staged copy, otherwise sync would upload a stale
            # rendition of a source that no longer encodes.
            staged.unlink(missing_ok=True)
            print(f"  skip (unreadable): {source} — {exc}", file=sys.stderr)
            return None
    else:
        shutil.copy2(source, staged)

    return source_str, staged_str, source_bytes, staged.stat().st_size


def build_jobs(src: Path, prefix: str) -> list[tuple[str, str, str]]:
    validate_prefix(prefix)
    jobs: list[tuple[str, str, str]] = []
    claimed: dict[str, str] = {}
    for path in sorted(src.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        action = classify(path)
        if action is None:
            continue
        relative = path.relative_to(src)
        if action == "transcode":
            # Append rather than replace the suffix. Replacing it would map both
            # portrait.png and its portrait.webp thumbnail onto one staging path,
            # dropping one of them nondeterministically.
            relative = relative.with_name(relative.name + ".webp")
        staged = str(STAGING_DIR / prefix / relative)
        if staged in claimed:
            raise RuntimeError(
                f"staging path collision: {path} and {claimed[staged]} both map to {staged}"
            )
        claimed[staged] = str(path)
        jobs.append((str(path), staged, action))
    return jobs


def _stage_with(
    executor_class: type, workers: int, jobs: list[tuple[str, str, str]]
) -> list[StagedFile]:
    staged: list[StagedFile] = []
    with executor_class(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(stage_one, jobs, chunksize=8), start=1):
            if result is None:
                continue
            source_str, staged_str, source_bytes, staged_bytes = result
            staged.append(StagedFile(Path(source_str), Path(staged_str), source_bytes, staged_bytes))
            if index % 200 == 0:
                print(f"  staged {index}/{len(jobs)}")
    return staged


def prune_staging(prefix: str, keep: set[str]) -> int:
    """Drop staged files whose source no longer exists.

    Without this, a deleted source keeps being re-uploaded from its stale staging
    copy. Only the staging tree is touched; objects already in R2 are left alone,
    since sync runs without --delete.
    """
    # validate_prefix rejects a symlink at any component, so root cannot be one here.
    validate_prefix(prefix)
    staging_root = STAGING_DIR.resolve()
    root = STAGING_DIR / prefix
    if not root.is_dir():
        return 0

    removed = 0
    for path in root.rglob("*"):
        if str(path) in keep:
            continue
        if path.is_symlink():
            # Unlink the link itself, never follow it. This is the only safe way to
            # clear a symlink: is_file() misses one pointing at a directory, and the
            # rmdir sweep below would hit ENOTDIR on it.
            path.unlink()
            removed += 1
            continue
        if not path.is_file():
            continue
        # Second line of defence: never unlink outside the staging tree.
        if not path.resolve().is_relative_to(staging_root):
            print(f"  skip (outside staging tree): {path}", file=sys.stderr)
            continue
        path.unlink()
        removed += 1

    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()
    return removed


def stage_all(jobs: list[tuple[str, str, str]]) -> list[StagedFile]:
    """Encode in parallel, preferring processes for the CPU-bound WebP work.

    A sandbox may deny the semaphores a process pool needs — at construction on
    some platforms, only once work is dispatched on others — so both failure
    points fall back to threads.
    """
    workers = max(1, (os.cpu_count() or 4) - 1)
    try:
        return _stage_with(ProcessPoolExecutor, workers, jobs)
    except (PermissionError, OSError, BrokenProcessPool):
        print("  (process pool unavailable, using threads)")
    return _stage_with(ThreadPoolExecutor, workers, jobs)


def pending_upload_bytes(prefix: str, profile: str | None) -> int:
    """Bytes `aws s3 sync` would actually transfer, taken from its own dry run.

    Charging the whole staging tree against the quota instead would refuse a
    re-run of an already-uploaded prefix, even though --size-only would skip
    every object. Any parse or transport failure raises, so callers fail closed.
    """
    validate_prefix(prefix)
    command = aws_base_command(profile) + [
        "s3",
        "sync",
        str(STAGING_DIR / prefix),
        f"s3://{bucket_name()}/{prefix}",
        "--size-only",
        "--dryrun",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not dry-run the sync: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"sync --dryrun failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    marker = "(dryrun) upload:"
    total = 0
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith(marker):
            continue
        source = stripped[len(marker) :].split(" to ", 1)[0].strip()
        try:
            total += Path(source).stat().st_size
        except OSError as exc:
            raise RuntimeError(f"could not size a pending upload ({source}): {exc}") from exc
    return total


def check_large_media(staged: list[StagedFile], allow_large_media: bool) -> None:
    """Raise unless a video-heavy push has been opted into explicitly.

    Reports on staged size, which for video equals source size — these suffixes are
    copied, never transcoded.
    """
    videos = [item for item in staged if item.source.suffix.lower() in LARGE_MEDIA_SUFFIXES]
    total = sum(item.staged_bytes for item in videos)
    if total <= LARGE_MEDIA_WARN_BYTES:
        return

    print(f"\nlarge-media guard: {len(videos)} video file(s), {human(total)} uncompressed")
    for item in sorted(videos, key=lambda i: i.staged_bytes, reverse=True)[:10]:
        print(f"  {human(item.staged_bytes):>10}  {item.source}")
    if len(videos) > 10:
        print(f"  ... and {len(videos) - 10} more")

    if not allow_large_media:
        raise RuntimeError(
            f"upload refused: {human(total)} of video exceeds the "
            f"{human(LARGE_MEDIA_WARN_BYTES)} large-media threshold. "
            "Re-run with --allow-large-media if that is intended."
        )
    print("  guard: overridden by --allow-large-media")


def check_quota(prefix: str, staged_bytes: int, profile: str | None) -> int:
    """Return current bucket bytes, or raise if the upload would breach the ceiling."""
    current_bytes, object_count = bucket_usage(profile)
    pending_bytes = pending_upload_bytes(prefix, profile)
    projected = current_bytes + pending_bytes

    print(f"  bucket now:  {human(current_bytes)} across {object_count} objects")
    print(f"  staged:      {human(staged_bytes)}")
    print(f"  to transfer: {human(pending_bytes)} (rest already in R2)")
    print(f"  projected:   {human(projected)} of {human(QUOTA_CEILING_BYTES)} ceiling")

    if projected > QUOTA_CEILING_BYTES:
        raise RuntimeError(
            f"upload refused: projected {human(projected)} exceeds the "
            f"{human(QUOTA_CEILING_BYTES)} ceiling (80% of the {human(FREE_TIER_BYTES)} free tier)"
        )
    return current_bytes


def sync(prefix: str, profile: str | None, apply: bool) -> int:
    validate_prefix(prefix)
    local = STAGING_DIR / prefix
    command = aws_base_command(profile) + [
        "s3",
        "sync",
        str(local),
        f"s3://{bucket_name()}/{prefix}",
        "--size-only",
    ]
    if not apply:
        command.append("--dryrun")
    print(f"  $ {' '.join(command)}")
    try:
        return subprocess.run(command).returncode
    except OSError as exc:
        # Most often a missing `aws` binary. main() turns this into one clean line.
        raise RuntimeError(f"could not run the AWS CLI: {exc}") from exc


def cmd_status(args: argparse.Namespace) -> int:
    if not has_credentials(args.profile):
        print("No R2 credentials in the environment. See the setup note in this script's docstring.")
        return 2
    try:
        current_bytes, object_count = bucket_usage(args.profile)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    used_pct = 100 * current_bytes / FREE_TIER_BYTES
    print(f"bucket:   {bucket_name()}")
    print(f"objects:  {object_count}")
    print(f"used:     {human(current_bytes)} ({used_pct:.1f}% of the {human(FREE_TIER_BYTES)} free tier)")
    print(f"ceiling:  {human(QUOTA_CEILING_BYTES)}")
    print(f"headroom: {human(max(0, QUOTA_CEILING_BYTES - current_bytes))}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    src = (REPO_ROOT / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src)
    if not src.is_dir():
        print(f"error: {src} is not a directory", file=sys.stderr)
        return 1

    jobs = build_jobs(src, args.prefix)
    if not jobs:
        print(f"no archivable files under {src}")
        return 0

    source_bytes = sum(Path(job[0]).stat().st_size for job in jobs)
    transcode_count = sum(1 for job in jobs if job[2] == "transcode")

    print(f"source:     {src}")
    print(f"files:      {len(jobs)} ({transcode_count} to transcode, {len(jobs) - transcode_count} to copy)")
    print(f"source size:{human(source_bytes)}")
    print(f"estimate:   ~{human(source_bytes / 10)} after WebP q{WEBP_QUALITY} (measured ~10x on real outputs)")
    print("\nRun 'push' to stage and check the quota gate for an exact figure.")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    src = (REPO_ROOT / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src)
    if not src.is_dir():
        print(f"error: {src} is not a directory", file=sys.stderr)
        return 1

    jobs = build_jobs(src, args.prefix)
    if not jobs:
        print(f"no archivable files under {src}")
        return 0

    pruned = prune_staging(args.prefix, {job[1] for job in jobs})
    if pruned:
        # "entries", not "files": the count includes unlinked symlinks.
        print(f"pruned {pruned} staged entries with no live source")

    print(f"staging {len(jobs)} files into {STAGING_DIR / args.prefix} ...")
    staged = stage_all(jobs)
    staged_bytes = sum(item.staged_bytes for item in staged)
    source_bytes = sum(item.source_bytes for item in staged)
    ratio = source_bytes / staged_bytes if staged_bytes else 0
    print(f"  {len(staged)} files: {human(source_bytes)} -> {human(staged_bytes)} ({ratio:.1f}x smaller)")

    # Ahead of the credential check: a video-heavy --src is a mistake worth naming even
    # when the run could never have uploaded anything.
    try:
        check_large_media(staged, args.allow_large_media)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    if not has_credentials(args.profile):
        print("\nStaging complete, but no R2 credentials are set — stopping before upload.")
        print("See the setup note in this script's docstring, then re-run with --apply.")
        return 2

    print("\nquota gate:")
    try:
        check_quota(args.prefix, staged_bytes, args.profile)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print("  gate: PASS")

    print(f"\nsync ({'apply' if args.apply else 'dry run'}):")
    returncode = sync(args.prefix, args.profile, args.apply)
    if returncode != 0:
        print(f"\naws s3 sync exited {returncode}", file=sys.stderr)
        return returncode

    if not args.apply:
        print("\nDry run only. Re-run with --apply to upload.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="AWS CLI profile holding the R2 credentials")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="show bucket usage against the free tier")
    status.set_defaults(func=cmd_status)

    plan = subparsers.add_parser("plan", help="estimate an archive run without writing anything")
    plan.add_argument("--src", required=True, help="source directory, absolute or relative to the repo root")
    plan.add_argument("--prefix", default="archive", help="key prefix inside the bucket")
    plan.set_defaults(func=cmd_plan)

    push = subparsers.add_parser("push", help="stage, check the quota gate, then sync")
    push.add_argument("--src", required=True, help="source directory, absolute or relative to the repo root")
    push.add_argument("--prefix", default="archive", help="key prefix inside the bucket")
    push.add_argument("--apply", action="store_true", help="actually upload (default is a dry run)")
    push.add_argument(
        "--allow-large-media",
        action="store_true",
        help=f"permit a push carrying over {human(LARGE_MEDIA_WARN_BYTES)} of video",
    )
    push.set_defaults(func=cmd_push)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
