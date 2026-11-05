#!/usr/bin/env python3
"""Summarize brae_interFoam per-rank logs as one shell-sourceable record.

Usage: summarize_run.py META.env LOG [LOG ...]
META.env contains only simple shell assignments, written by run_dgx.sh. The
script rejects incomplete/rank-inconsistent runs instead of reporting partial
startup output as a timing measurement. Standard output is a `ROSIE_SUMMARY`
line plus ROSIE_* values suitable for eval in bash (all values are quoted).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shlex
import sys

STEP = re.compile(r"^\s*t\s*=\s*(\S+)\s+dt\s*=\s*(\S+)\s+.*?\[device\]")
PHASE_STEPS = re.compile(r"^\s*phase time over\s+(\d+)\s+steps\b")
PHASE_TOTAL = re.compile(r"^\s*ALL PHASES\s+([0-9.eE+-]+)\s+ms/step\s*$")
END = re.compile(r"^\s*End:\s*t\s*=\s*(\S+),\s*alpha in\s*\[\s*([^,]+),\s*([^\]]+)\]")


def finite_float(text: str, what: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"invalid {what}: {text!r}") from error
    if not (-float("inf") < value < float("inf")):
        raise ValueError(f"non-finite {what}: {text!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meta", type=pathlib.Path)
    parser.add_argument("logs", type=pathlib.Path, nargs="+")
    args = parser.parse_args()
    meta: dict[str, str] = {}
    for line in args.meta.read_text().splitlines():
        if re.match(r"^[A-Z_][A-Z0-9_]*=", line):
            key, value = line.split("=", 1)
            meta[key] = shlex.split(value)[0] if value else ""
    np_expected = int(meta["NP"])
    backend = meta["BACKEND"]
    case = meta["CASE"]
    wall = finite_float(meta["WALL_SECONDS"], "wall seconds")
    timeout = int(meta["TIMEOUT_SECONDS"])
    if len(args.logs) != np_expected:
        raise ValueError(f"found {len(args.logs)} rank logs, expected {np_expected}")

    # All ranks are part of the result. rank0 alone can print End after peers
    # failed, or be waiting on a peer in a collective; neither is a PASS.
    rank_reports = []
    for path in args.logs:
        steps = []
        phase_count = []
        phase_ms = []
        ends = []
        for line in path.read_text(errors="replace").splitlines():
            match = STEP.match(line)
            if match:
                steps.append((finite_float(match.group(1), "time"),
                              finite_float(match.group(2), "deltaT")))
            match = PHASE_STEPS.match(line)
            if match:
                phase_count.append(int(match.group(1)))
            match = PHASE_TOTAL.match(line)
            if match:
                phase_ms.append(finite_float(match.group(1), "ALL PHASES ms/step"))
            match = END.match(line)
            if match:
                ends.append((finite_float(match.group(1), "end time"),
                             finite_float(match.group(2), "alpha minimum"),
                             finite_float(match.group(3), "alpha maximum")))
        if not steps:
            raise ValueError(f"{path}: no device time-step lines")
        if not ends:
            raise ValueError(f"{path}: no End line (incomplete run)")
        if not phase_count or not phase_ms:
            raise ValueError(f"{path}: no BRAE_INTER_PHASE_TIME phase table")
        if len(set(phase_count)) != 1 or len(set(phase_ms)) != 1:
            raise ValueError(f"{path}: multiple inconsistent phase tables")
        if phase_count[0] != len(steps):
            raise ValueError(f"{path}: phase table covers {phase_count[0]} steps, saw {len(steps)}")
        rank_reports.append(dict(path=str(path), steps=steps, phase_ms=phase_ms[0], end=ends[-1]))

    steps_count = len(rank_reports[0]["steps"])
    if any(len(r["steps"]) != steps_count for r in rank_reports):
        raise ValueError("ranks disagree on number of steps")
    expected_steps = int(meta.get("EXPECTED_STEPS", "0"))
    if expected_steps and steps_count != expected_steps:
        raise ValueError(f"run completed {steps_count} steps, expected {expected_steps}")
    final_dt = rank_reports[0]["steps"][-1][1]
    if any(r["steps"][-1] != rank_reports[0]["steps"][-1] for r in rank_reports):
        raise ValueError("ranks disagree on final time or deltaT")
    final_end = rank_reports[0]["end"]
    if any(r["end"] != final_end for r in rank_reports):
        raise ValueError("ranks disagree on End time or alpha range")
    phase_values = [r["phase_ms"] for r in rank_reports]
    # Every rank runs the same number of steps. Report the slowest rank: it is
    # the collective iteration's critical path, not an unweighted average.
    slowest = max(range(len(rank_reports)), key=lambda i: phase_values[i])
    phase_ms = phase_values[slowest]
    startup_seconds = wall - phase_ms * steps_count / 1000.0
    values = {
        "CASE": case,
        "NP": str(np_expected),
        "BACKEND": backend,
        "GPU_MAP": meta.get("GPU_MAP", ""),
        "COND_GRAPH": meta.get("COND_GRAPH", ""),
        "STEPS": str(steps_count),
        "EXPECTED_STEPS": str(expected_steps),
        "WALL_SECONDS": f"{wall:.6f}",
        "MS_PER_STEP": f"{phase_ms:.3f}",
        "STARTUP_AND_OUTPUT_SECONDS": f"{startup_seconds:.6f}",
        "SLOWEST_RANK_LOG": rank_reports[slowest]["path"],
        "ALPHA_MIN": f"{final_end[1]:.12g}",
        "ALPHA_MAX": f"{final_end[2]:.12g}",
        "DELTAT": f"{final_dt:.17g}",
        "END_TIME": f"{final_end[0]:.17g}",
        "TIMEOUT_SECONDS": str(timeout),
        "STATUS": "COMPLETE",
    }
    for key, value in values.items():
        print(f"ROSIE_{key}={shlex.quote(value)}")
    print("ROSIE_SUMMARY=" + shlex.quote(
        f"case={case} np={np_expected} backend={backend} map={values['GPU_MAP'] or '-'} "
        f"condGraph={values['COND_GRAPH'] or '-'} steps={steps_count} wall_s={wall:.3f} "
        f"ms_per_step={phase_ms:.3f} startup_output_s={startup_seconds:.3f} "
        f"alpha=[{final_end[1]:.8g},{final_end[2]:.8g}] deltaT={final_dt:.8g} "
        f"endTime={final_end[0]:.8g} slowestLog={rank_reports[slowest]['path']}"))


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError) as error:
        print(f"SUMMARY ERROR: {error}", file=sys.stderr)
        sys.exit(1)
