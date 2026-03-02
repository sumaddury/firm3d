import sys
import unittest
from pathlib import Path

import numpy as np

import firm3dpp
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.util.gpu_utils import boozer_interpolant


HAS_METAL_INTERPOLATION = sys.platform == "darwin" and hasattr(
    firm3dpp, "test_gpu_interpolation"
)


def _sample_test_points(n_test_pts: int) -> np.ndarray:
    np.random.seed(1865)
    s = np.random.uniform(low=0, high=0.95, size=(n_test_pts, 1))
    t = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    z = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    return np.hstack((s, t, z))


@unittest.skipUnless(
    HAS_METAL_INTERPOLATION,
    "Metal interpolation export not available on this platform/build.",
)
class TestMetalInterpolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = (Path(__file__).parent / ".." / "test_files").resolve()
        boozmn_filename = str((test_dir / "boozmn_LandremanPaul2021_QA_lowres.nc").resolve())

        bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=True)
        cls.field = InterpolatedBoozerField(
            bri,
            3,
            ns_interp=15,
            ntheta_interp=15,
            nzeta_interp=15,
        )
        cls.nfp = bri.nfp

        cls.srange, cls.trange, cls.zrange, cls.quad_info, _ = boozer_interpolant(
            cls.field, cls.nfp, ns=15, ntheta=15, nzeta=15, vacuum=True
        )

    def _compute_cpu_reference(self, stz: np.ndarray) -> np.ndarray:
        self.field.set_points(stz)
        modB = self.field.modB()
        modB_derivs = self.field.modB_derivs()
        G = self.field.G()
        iota = self.field.iota()
        return np.hstack((modB, modB_derivs, G, iota))

    def _run_and_compare(self, rhs: str, tol: float):
        n_test_pts = 5000
        stz = np.ascontiguousarray(_sample_test_points(n_test_pts))
        cpu = self._compute_cpu_reference(stz)
        try:
            metal = firm3dpp.test_gpu_interpolation(
                self.quad_info,
                self.srange,
                self.trange,
                self.zrange,
                stz.copy(),
                rhs,
                stz.shape[0],
            )
        except RuntimeError as err:
            if "Metal device not available" in str(err):
                self.skipTest("Metal device not available in this runtime")
            raise
        metal = np.reshape(metal, (stz.shape[0], -1))

        error = np.abs(cpu - metal) / (np.abs(cpu) + 1.0)
        max_err = error.max()
        self.assertTrue(
            np.isclose(metal, cpu, rtol=tol, atol=tol).all(),
            msg=f"Metal interpolation mismatch for rhs='{rhs}', max relative error={max_err}",
        )

    def test_boozer_vacuum_f32(self):
        self._run_and_compare(rhs="boozer_vacuum_f32", tol=1e-2)
