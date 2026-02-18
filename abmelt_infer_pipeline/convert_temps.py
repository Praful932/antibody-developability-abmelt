#!/usr/bin/env python3
"""
Convert Tagg, Tm, and Tmon values between normalized (z-score) and denormalized (°C) scales.

Uses training-set statistics from utils.py:
  Tagg: mean=75.4°C, std=12.5
  Tm:   mean=64.9°C, std=9.02
  Tmon: mean=56.0°C, std=9.55

Usage:
  # Normalize raw °C values to z-scores
  python convert_temps.py --normalize --tagg 94.64 --tm 69.84 --tmon 54.07

  # Denormalize z-scores back to °C
  python convert_temps.py --denormalize --tagg 0.935 --tm 0.538 --tmon -0.2

  # Any subset works
  python convert_temps.py --normalize --tm 69.84
  python convert_temps.py --denormalize --tagg 1.433 --tmon -0.209
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from utils import normalize, renormalize, DEFAULT_STATS


def main():
    parser = argparse.ArgumentParser(
        description="Convert Tagg/Tm/Tmon between normalized (z-score) and denormalized (°C) scales.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument(
        "--normalize",
        action="store_true",
        help="Convert °C values to z-scores: (x - mean) / std"
    )
    direction.add_argument(
        "--denormalize",
        action="store_true",
        help="Convert z-scores back to °C: x * std + mean"
    )

    parser.add_argument("--tagg", type=float, default=None, metavar="VALUE",
                        help="Tagg value to convert")
    parser.add_argument("--tm", type=float, default=None, metavar="VALUE",
                        help="Tm value to convert")
    parser.add_argument("--tmon", type=float, default=None, metavar="VALUE",
                        help="Tmon value to convert")

    args = parser.parse_args()

    inputs = {"tagg": args.tagg, "tm": args.tm, "tmon": args.tmon}
    if all(v is None for v in inputs.values()):
        parser.error("Provide at least one of --tagg, --tm, --tmon")

    fn = normalize if args.normalize else renormalize
    in_unit = "°C" if args.normalize else "z-score"
    out_unit = "z-score" if args.normalize else "°C"

    print(f"\n{'Target':<8}  {'Input':>12}  {'Output':>12}  {'mean':>8}  {'std':>8}")
    print("-" * 56)
    for temp_type, value in inputs.items():
        if value is None:
            continue
        result = fn(np.array([value]), temp_type=temp_type)[0]
        stats = DEFAULT_STATS[temp_type]
        print(
            f"{temp_type.upper():<8}  "
            f"{value:>10.4f} {in_unit}  "
            f"{result:>10.4f} {out_unit}  "
            f"{stats['mean']:>8}  "
            f"{stats['std']:>8}"
        )
    print()


if __name__ == "__main__":
    main()
