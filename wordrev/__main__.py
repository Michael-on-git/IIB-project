# wordrev/__main__.py
"""
Command-line interface.

    python -m wordrev <netlist.json> [options]

See --help for the full option list.
"""
from __future__ import annotations

import argparse
import sys

from .pipeline import format_human, format_json, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m wordrev",
        description=(
            "Recover word-level structure from a Yosys JSON gate-level netlist."
        ),
    )
    p.add_argument("netlist", help="Path to Yosys JSON netlist")
    p.add_argument("--top", help="Module name (default: first module)")
    p.add_argument("--shape-k", type=int, default=3,
                   help="Shape hashing depth (default: 3)")
    p.add_argument("--min-width", type=int, default=2,
                   help="Minimum word width (default: 2)")
    p.add_argument("--no-propagate", action="store_true",
                   help="Skip word propagation")
    p.add_argument("--no-identify", action="store_true",
                   help="Skip function identification")
    p.add_argument("--max-inputs", type=int, default=12,
                   help="Identify: max free variables (default: 12)")
    p.add_argument("--max-gates", type=int, default=500,
                   help="Identify: max interior gates (default: 500)")
    p.add_argument("--json", action="store_true",
                   help="Machine-readable JSON output")
    p.add_argument("--stats", action="store_true",
                   help="Include timing statistics")
    p.add_argument("--method", choices=["wl", "unroll"], default="wl",
                   help="Shape hashing method (default: wl)")
    return p


def main(argv: list[str] | None = None) -> int:
    # Reconfigure stdout for UTF-8 with a safe fallback. On Windows, the
    # default console encoding (cp1252, cp437, etc.) cannot represent
    # many Unicode characters; errors="replace" turns an unrepresentable
    # character into '?' rather than crashing mid-print.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Some environments (rare) do not support reconfigure. Fall
            # through to default behaviour; the ASCII formatter output
            # would still work.
            pass

    args = build_parser().parse_args(argv)

    try:
        result = run_pipeline(
            args.netlist,
            top=args.top,
            shape_k=args.shape_k,
            min_width=args.min_width,
            propagate=not args.no_propagate,
            shape_method=args.method,
            identify=not args.no_identify,
            max_external_inputs=args.max_inputs,
            max_fragment_gates=args.max_gates,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(format_json(result))
    else:
        print(format_human(result, include_stats=args.stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())