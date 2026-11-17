"""Run the numbered modelling stages in order, stopping at the first failure.

Stages 00-04 prepare the data and are deliberately not run here; run them first
(see README).  This script only orchestrates the numbered stages 10 -> 95.

Usage:
    python code/run_all.py --dry-run
    python code/run_all.py
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")

STEPS = [
    "10_q1_model.py",
    "11_q1_extra.py",
    "20_q2_diag.py",
    "21_q2_opt.py",
    "22_q2_sim.py",
    "23_q2_saveplan.py",
    "24_cell_curve.py",
    "30_q3_incentive.py",
    "40_q4_efence.py",
    "50_q5_platform.py",
    "60_sensitivity.py",
    "90_figs.py",
    "95_numbers.py",
]


def resolve(name):
    return name if os.path.isabs(name) else os.path.join(CODE, name)


def run_steps(steps, python=sys.executable, cwd=ROOT):
    """Run each script; return the first non-zero status (0 if all pass)."""
    for i, name in enumerate(steps, 1):
        path = resolve(name)
        print("[%d/%d] %s" % (i, len(steps), path), flush=True)
        if not os.path.exists(path):
            print("FAILED: missing script %s; stopping" % path, flush=True)
            return 2
        status = subprocess.call([python, path], cwd=cwd)
        if status != 0:
            print("FAILED: %s exited %d; stopping" %
                  (os.path.basename(path), status), flush=True)
            return status
    print("all %d steps completed" % len(steps), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run the numbered modelling stages 10 -> 95 in order.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the order only, run nothing")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter to use (default: this Python)")
    parser.add_argument("--scripts", default=None,
                        help="comma-separated override of the step list "
                             "(testing only; absolute paths allowed)")
    args = parser.parse_args()

    steps = ([s.strip() for s in args.scripts.split(",") if s.strip()]
             if args.scripts else STEPS)
    if args.dry_run:
        print("dry run: data preparation 00-04 is not included")
        for i, name in enumerate(steps, 1):
            print("%2d. %s" % (i, resolve(name)))
        return 0
    return run_steps(steps, python=args.python)


if __name__ == "__main__":
    raise SystemExit(main())
