#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from statistics import median

import numpy as np

import firm3dpp
from firm3d.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from firm3d.util.gpu_utils import boozer_interpolant


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("n-points list is empty")
    return out


def sample_points(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 0.95, size=(n, 1))
    t = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    z = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    return np.ascontiguousarray(np.hstack((s, t, z)))


def cpu_eval_vacuum(field: InterpolatedBoozerField, stz: np.ndarray) -> np.ndarray:
    field.set_points(stz)
    modb = field.modB()
    modb_derivs = field.modB_derivs()
    g = field.G()
    iota = field.iota()
    return np.hstack((modb, modb_derivs, g, iota))


def gpu_eval_vacuum(
    quad_info: np.ndarray,
    srange: tuple[float, float, int],
    trange: tuple[float, float, int],
    zrange: tuple[float, float, int],
    stz: np.ndarray,
) -> np.ndarray:
    out = firm3dpp.test_gpu_interpolation(
        quad_info,
        srange,
        trange,
        zrange,
        stz.copy(),
        "boozer_vacuum",
        stz.shape[0],
    )
    return np.reshape(out, (stz.shape[0], -1))


def bench_one(fn, repeats: int) -> tuple[list[float], np.ndarray]:
    times_ms: list[float] = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)
    assert result is not None
    return times_ms, result


def summarize(times_ms: list[float]) -> tuple[float, float, float]:
    arr = np.array(times_ms, dtype=float)
    return float(np.min(arr)), float(median(times_ms)), float(np.max(arr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GPU interpolation against CPU baseline.")
    parser.add_argument(
        "--boozmn",
        default="examples/inputs/boozmn_aten_rescaled_low_res.nc",
        help="Path to boozmn file.",
    )
    parser.add_argument(
        "--n-metagrid",
        type=int,
        default=15,
        help="Interpolant metagrid resolution (ns=ntheta=nzeta=n-metagrid).",
    )
    parser.add_argument(
        "--n-points",
        type=str,
        default="10000,50000,100000",
        help="Comma-separated list of point counts to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Timed repeats per method.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per method.")
    parser.add_argument("--seed", type=int, default=1865, help="Base seed for random points.")
    args = parser.parse_args()

    n_points_list = parse_int_list(args.n_points)

    print("Setting up field/interpolant...")
    bri = BoozerRadialInterpolant(args.boozmn, 3, enforce_vacuum=True)
    field = InterpolatedBoozerField(
        bri,
        3,
        ns_interp=args.n_metagrid,
        ntheta_interp=args.n_metagrid,
        nzeta_interp=args.n_metagrid,
    )
    nfp = bri.nfp

    srange, trange, zrange, quad_info, _ = boozer_interpolant(
        field,
        nfp,
        args.n_metagrid,
        args.n_metagrid,
        args.n_metagrid,
        vacuum=True,
    )

    print("")
    print("Benchmarking boozer_vacuum interpolation")
    print(f"boozmn={args.boozmn}")
    print(f"n_metagrid={args.n_metagrid} repeats={args.repeats} warmup={args.warmup}")
    print("")
    print("{:>10}  {:>22}  {:>22}  {:>12}  {:>12}".format(
        "n_points", "cpu_ms(min/med/max)", "gpu_ms(min/med/max)", "speedup", "max_rel_err"
    ))

    for idx, npts in enumerate(n_points_list):
        stz = sample_points(npts, args.seed + idx)

        for _ in range(args.warmup):
            _ = cpu_eval_vacuum(field, stz)
            _ = gpu_eval_vacuum(quad_info, srange, trange, zrange, stz)

        cpu_times, cpu_out = bench_one(lambda: cpu_eval_vacuum(field, stz), args.repeats)
        gpu_times, gpu_out = bench_one(
            lambda: gpu_eval_vacuum(quad_info, srange, trange, zrange, stz), args.repeats
        )

        cpu_min, cpu_med, cpu_max = summarize(cpu_times)
        gpu_min, gpu_med, gpu_max = summarize(gpu_times)
        speedup = cpu_med / gpu_med if gpu_med > 0.0 else float("inf")

        rel_err = np.abs(cpu_out - gpu_out) / (np.abs(cpu_out) + 1.0)
        max_rel_err = float(np.max(rel_err))

        print(
            "{:10d}  {:7.2f}/{:7.2f}/{:7.2f}  {:7.2f}/{:7.2f}/{:7.2f}  {:12.3f}  {:12.3e}".format(
                npts,
                cpu_min,
                cpu_med,
                cpu_max,
                gpu_min,
                gpu_med,
                gpu_max,
                speedup,
                max_rel_err,
            )
        )


if __name__ == "__main__":
    main()
