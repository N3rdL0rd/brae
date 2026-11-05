#!/usr/bin/env python3
"""Print a scaling table from run_dgx.sh result directories.

Usage: collect_results.py RESULT_DIR [RESULT_DIR ...] > scaling.tsv
Only runs whose summary.env status is COMPLETE are included. A failed run
must be diagnosed from its retained rank logs, never silently dropped here.
"""
from __future__ import annotations

import csv
import pathlib
import shlex
import sys

FIELDS = ["ROSIE_CASE", "ROSIE_NP", "ROSIE_BACKEND", "ROSIE_GPU_MAP", "ROSIE_COND_GRAPH",
          "ROSIE_STEPS", "ROSIE_WALL_SECONDS", "ROSIE_MS_PER_STEP",
          "ROSIE_STARTUP_AND_OUTPUT_SECONDS", "ROSIE_ALPHA_MIN", "ROSIE_ALPHA_MAX",
          "ROSIE_DELTAT", "ROSIE_END_TIME", "ROSIE_SLOWEST_RANK_LOG"]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    count = 0
    for arg in sys.argv[1:]:
        directory = pathlib.Path(arg)
        values = {}
        for line in (directory / "summary.env").read_text().splitlines():
            if line.startswith("ROSIE_") and "=" in line:
                key, raw = line.split("=", 1)
                values[key] = shlex.split(raw)[0] if raw else ""
        if values.get("ROSIE_STATUS") != "COMPLETE":
            raise SystemExit(f"{directory}: no COMPLETE summary; do not use it as scaling evidence")
        writer.writerow({key: values.get(key, "") for key in FIELDS})
        count += 1
    if not count:
        raise SystemExit("no result directories")


if __name__ == "__main__":
    main()
