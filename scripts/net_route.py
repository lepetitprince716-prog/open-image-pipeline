#!/usr/bin/env python3
"""Probe and pick the live transport to each GPU box.

    python3 scripts/net_route.py status [--json] [--box boxa|boxb]
    python3 scripts/net_route.py pick boxa|boxb [--what route|ssh|http|proxy]
    python3 scripts/net_route.py env              # export lines for the live routes
    python3 scripts/net_route.py refresh          # force re-probe, rewrite the cache
    python3 scripts/net_route.py reroute [--dry-run]  # kill forwards stuck on a stale transport

`status` probes every route without touching the cache. `pick` reads the
cached answer (refreshing it when stale) and prints one field — `proxy` is the
argv an ssh ProxyCommand execs. `env` prints BOXA_HOST/BOXB_HOST export lines
pointing at each box's live ComfyUI endpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from loramake import net_routes  # noqa: E402


def cmd_status(args: argparse.Namespace) -> int:
    boxes = args.box or list(net_routes.ROUTES)
    report = {"boxes": [net_routes.probe_box(b) for b in boxes]}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        for box in report["boxes"]:
            print(box["box"])
            for r in box["routes"]:
                ssh = f"ssh {'OK' if r['ssh_ok'] else 'DOWN':<4} {r['ssh_rtt_ms']:>8.1f} ms"
                http = (f"http {'OK' if r['http_ok'] else 'DOWN':<4} "
                        f"{r['http_rtt_ms']:>8.1f} ms")
                ver = f"  ComfyUI {r['comfyui_version']}" if r["comfyui_version"] else ""
                print(f"  {r['route']:<4} {ssh}   {http}   {r['comfy_host']}{ver}")
    all_ssh = all(
        any(r["ssh_ok"] for r in box["routes"]) for box in report["boxes"])
    return 0 if all_ssh else 1


def cmd_pick(args: argparse.Namespace) -> int:
    what = args.what
    if what == "route":
        out = net_routes.pick_route(args.box, refresh=args.refresh)
    elif what == "ssh":
        out = net_routes.ssh_alias(args.box, refresh=args.refresh)
    elif what == "http":
        out = net_routes.comfy_host(args.box, refresh=args.refresh)
    else:
        out = net_routes.proxy_command_line(args.box, refresh=args.refresh)
    print(out)
    return 0


def cmd_env(_args: argparse.Namespace) -> int:
    for box in net_routes.ROUTES:
        print(f"export {box.upper()}_HOST={net_routes.comfy_host(box)}")
    return 0


def cmd_refresh(_args: argparse.Namespace) -> int:
    for box in net_routes.ROUTES:
        route = net_routes.pick_route(box, refresh=True)
        print(f"{box}: {route}")
    return 0


def cmd_reroute(args: argparse.Namespace) -> int:
    rc = 0
    for box in net_routes.ROUTES:
        rep = net_routes.reroute(box, dry_run=args.dry_run)
        verb = "would kill" if args.dry_run else "killed"
        acted = rep["stale"] if args.dry_run else rep["killed"]
        print(f"{box}: route={rep['route']} stale={rep['stale']} {verb}={acted}")
        for pid, why in rep["failed"]:
            print(f"  pid {pid}: {why}", file=sys.stderr)
            rc = 1
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("status", help="probe every route (never uses the cache)")
    st.add_argument("--json", action="store_true")
    st.add_argument("--box", choices=list(net_routes.ROUTES), action="append")
    st.set_defaults(func=cmd_status)

    pk = sub.add_parser("pick", help="print the cached/live route for one box")
    pk.add_argument("box", choices=list(net_routes.ROUTES))
    pk.add_argument("--what", choices=("route", "ssh", "http", "proxy"),
                    default="route")
    pk.add_argument("--refresh", action="store_true",
                    help="re-probe instead of reading the cache")
    pk.set_defaults(func=cmd_pick)

    en = sub.add_parser("env", help="print BOXA_HOST/BOXB_HOST export lines")
    en.set_defaults(func=cmd_env)

    rf = sub.add_parser("refresh", help="re-probe all boxes and rewrite the cache")
    rf.set_defaults(func=cmd_refresh)

    rr = sub.add_parser("reroute", help="kill ssh forwards riding a transport that is no longer the pick")
    rr.add_argument("--dry-run", action="store_true")
    rr.set_defaults(func=cmd_reroute)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
