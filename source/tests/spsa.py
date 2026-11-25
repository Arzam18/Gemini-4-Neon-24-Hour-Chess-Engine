"""
Autonomous SPSA (Simultaneous Perturbation Stochastic Approximation) tuner
using fastchess on up to 14 cores.

Usage:
  python source/tests/spsa.py --engine source/build/engine_tune.exe --params source/tests/spsa_params.json --iters 200 --games 28 --tc 3+0.03
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time


SCORE_RE = re.compile(r"Games:\s*\d+,\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)")


def run_match(fastchess, engine, book, tc, concurrency, hash_mb, games, plus_opts, minus_opts, seed):
    cmd = [
        fastchess,
        "-engine", f"cmd={engine}", "name=plus", f"option.Hash={hash_mb}",
    ]
    for k, v in plus_opts.items():
        cmd.append(f"option.{k}={v}")

    cmd += [
        "-engine", f"cmd={engine}", "name=minus", f"option.Hash={hash_mb}",
    ]
    for k, v in minus_opts.items():
        cmd.append(f"option.{k}={v}")

    cmd += [
        "-each", f"tc={tc}", "timemargin=0", "proto=uci",
        "-rounds", str(games // 2),
        "-games", "2",
        "-repeat",
        "-openings", f"file={book}", "format=pgn", "order=random", f"plies=16",
        "-concurrency", str(concurrency),
        "-recover",
        "-srand", str(seed),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    matches = SCORE_RE.findall(out)
    if not matches:
        raise RuntimeError(f"Could not parse fastchess score.\nOutput tail:\n{out[-1200:]}")
    w, l, d = map(int, matches[-1])
    return w, l, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastchess", default=r"resources\fastchess\fastchess.exe")
    ap.add_argument("--book", default=r"resources\fastchess\UHO.pgn")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--params", required=True, help="JSON list of {name, val, min, max, c, r}")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games", type=int, default=28, help="Even number of games per iteration (multiple of 2*concurrency ideal)")
    ap.add_argument("--tc", default="3+0.03")
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=14)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--state", default=r"source\tests\spsa_state.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    with open(args.params, "r", encoding="utf-8") as f:
        params = json.load(f)

    # Current continuous theta
    theta = [float(p["val"]) for p in params]
    start_iter = 0

    if os.path.exists(args.state):
        with open(args.state, "r", encoding="utf-8") as f:
            st = json.load(f)
        if len(st.get("theta", [])) == len(params):
            theta = [float(x) for x in st["theta"]]
            start_iter = int(st.get("iter", 0))
            print(f"Resumed SPSA state from {args.state} at iter {start_iter}", flush=True)

    alpha = 0.602
    gamma = 0.101
    A = 0.1 * args.iters
    t0 = time.time()

    for k in range(start_iter, args.iters):
        ck_scale = 1.0 / ((k + 1) ** gamma)
        ak_scale = ((A + 1.0) ** alpha) / ((A + k + 1.0) ** alpha)

        delta = [1.0 if rng.random() < 0.5 else -1.0 for _ in params]
        plus_opts = {}
        minus_opts = {}
        eff_delta = []

        for i, p in enumerate(params):
            c_i = float(p["c"]) * ck_scale
            p_plus = int(round(max(p["min"], min(p["max"], theta[i] + c_i * delta[i]))))
            p_minus = int(round(max(p["min"], min(p["max"], theta[i] - c_i * delta[i]))))
            if p_plus == p_minus:
                step = 1 if delta[i] > 0 else -1
                p_plus = int(max(p["min"], min(p["max"], round(theta[i]) + step)))
                p_minus = int(max(p["min"], min(p["max"], round(theta[i]) - step)))
            plus_opts[p["name"]] = p_plus
            minus_opts[p["name"]] = p_minus
            eff_delta.append((p_plus - p_minus) / 2.0)

        w, l, d = run_match(
            args.fastchess, args.engine, args.book, args.tc,
            args.concurrency, args.hash, args.games,
            plus_opts, minus_opts, args.seed + k * 9973
        )
        n = w + l + d
        score_diff = (w - l) / float(max(1, n))  # in [-1, 1]

        for i, p in enumerate(params):
            if eff_delta[i] == 0.0:
                continue
            r_i = float(p["r"]) * ak_scale
            c_i = float(p["c"]) * ck_scale
            # Standard SPSA step scaled by parameter range c_i
            step = r_i * score_diff * (1.0 if eff_delta[i] > 0 else -1.0) * c_i
            theta[i] = max(float(p["min"]), min(float(p["max"]), theta[i] + step))

        # Save state
        with open(args.state, "w", encoding="utf-8") as f:
            json.dump({"iter": k + 1, "theta": theta}, f, indent=2)

        if (k + 1) % 5 == 0 or k == start_iter or k + 1 == args.iters:
            elapsed = time.time() - t0
            summary = ", ".join(f"{p['name']}={int(round(theta[i]))}" for i, p in enumerate(params))
            print(
                f"[Iter {k+1:3d}/{args.iters}] W={w:2d} L={l:2d} D={d:2d} (diff={score_diff:+.3f}) "
                f"t={elapsed:.0f}s | {summary}",
                flush=True,
            )

    print("\n=== FINAL TUNED PARAMETERS ===", flush=True)
    for i, p in enumerate(params):
        print(f"  {p['name']:24s}: {int(p['val']):6d} -> {int(round(theta[i])):6d}", flush=True)


if __name__ == "__main__":
    main()
