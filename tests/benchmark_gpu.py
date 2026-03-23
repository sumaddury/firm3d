#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from statistics import median

import numpy as np

import firm3dpp
from firm3d.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from firm3d.util.gpu_utils import boozer_interpolant

HAS_GPU = hasattr(firm3dpp, "test_gpu_interpolation")
IS_CUDA = hasattr(firm3dpp, "boozer_gpu_tracing")
BACKEND = "CUDA" if IS_CUDA else "Metal"

N_FIELDS_VACUUM = 6


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if token:
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


def cpu_eval(field: InterpolatedBoozerField, stz: np.ndarray) -> np.ndarray:
    field.set_points(stz)
    return np.hstack((field.modB(), field.modB_derivs(), field.G(), field.iota()))


def gpu_eval(quad_info, srange, trange, zrange, stz: np.ndarray) -> np.ndarray:
    n = stz.shape[0]
    out = firm3dpp.test_gpu_interpolation(
        quad_info, srange, trange, zrange, stz.copy(), "boozer_vacuum", n
    )
    return np.reshape(out, (n, N_FIELDS_VACUUM))


def bench_one(fn, repeats):
    times_ms = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)
    assert result is not None
    return times_ms, result


def summarize(times_ms):
    arr = np.array(times_ms, dtype=float)
    return float(np.min(arr)), float(median(times_ms)), float(np.max(arr))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark test_gpu_interpolation (CUDA/Metal) vs CPU RegularGridInterpolant3D."
    )
    parser.add_argument(
        "--boozmn",
        default="examples/inputs/boozmn_aten_rescaled_low_res.nc",
        help="Path to boozmn NetCDF file.",
    )
    parser.add_argument("--n-metagrid", type=int, default=15,
                        help="Metagrid resolution (ns=ntheta=nzeta). Grid has 3*n+1 points per axis.")
    parser.add_argument("--n-points", type=str, default="10000,50000,100000",
                        help="Comma-separated evaluation point counts.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup",  type=int, default=2)
    parser.add_argument("--seed",    type=int, default=1865)
    args = parser.parse_args()

    if not HAS_GPU:
        print(
            "ERROR: firm3dpp.test_gpu_interpolation not found. "
            "Rebuild with CUDA or Apple Metal support.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_points_list = parse_int_list(args.n_points)

    print("Setting up field and interpolant...")
    bri = BoozerRadialInterpolant(args.boozmn, 3, enforce_vacuum=True)
    field = InterpolatedBoozerField(
        bri, 3,
        ns_interp=args.n_metagrid,
        ntheta_interp=args.n_metagrid,
        nzeta_interp=args.n_metagrid,
    )
    nfp = bri.nfp

    srange, trange, zrange, quad_info, _ = boozer_interpolant(
        field, nfp,
        args.n_metagrid, args.n_metagrid, args.n_metagrid,
        vacuum=True,
    )

    n_cells = args.n_metagrid
    print()
    print(f"Benchmarking: test_gpu_interpolation ({BACKEND})  vs  RegularGridInterpolant3D (CPU)")
    print(f"  boozmn     : {args.boozmn}")
    print(f"  n_metagrid : {n_cells}  =>  {3*n_cells+1}^3 grid points,  {n_cells}^3 cubic cells")
    print(f"  repeats    : {args.repeats}   warmup: {args.warmup}")
    print()
    print("{:>10}  {:>27}  {:>27}  {:>10}  {:>12}".format(
        "n_points", "cpu_ms (min/med/max)", "gpu_ms (min/med/max)", "speedup", "max_rel_err",
    ))

    for run_idx, npts in enumerate(n_points_list):
        stz = sample_points(npts, args.seed + run_idx)

        for _ in range(args.warmup):
            cpu_eval(field, stz)
            gpu_eval(quad_info, srange, trange, zrange, stz)

        cpu_times, cpu_out = bench_one(lambda: cpu_eval(field, stz), args.repeats)
        gpu_times, gpu_out = bench_one(
            lambda: gpu_eval(quad_info, srange, trange, zrange, stz),
            args.repeats,
        )

        cpu_min, cpu_med, cpu_max = summarize(cpu_times)
        gpu_min, gpu_med, gpu_max = summarize(gpu_times)
        speedup = cpu_med / gpu_med if gpu_med > 0.0 else float("inf")

        rel_err = np.abs(cpu_out - gpu_out) / (np.abs(cpu_out) + 1.0)
        max_rel_err = float(np.max(rel_err))

        print(
            "{:10d}  {:9.2f}/{:9.2f}/{:9.2f}  {:9.2f}/{:9.2f}/{:9.2f}  {:10.3f}  {:12.3e}".format(
                npts,
                cpu_min,   cpu_med,   cpu_max,
                gpu_min,   gpu_med,   gpu_max,
                speedup,
                max_rel_err,
            )
        )


if __name__ == "__main__":
    main()
