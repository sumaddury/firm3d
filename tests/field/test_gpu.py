# import time
import numpy as np

import firm3dpp
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.util.gpu_utils import boozer_interpolant, boozer_saw_interpolant
from firm3d.util.constants import (
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
    ALPHA_PARTICLE_CHARGE as CHARGE
)
from firm3d.field.tracing import (
    IterationStoppingCriterion,
    trace_particles_boozer,
    trace_particles_boozer_perturbed
)
from firm3d.saw.ae3d import AE3DEigenvector

import unittest

from firm3d.field.boozermagneticfield import ShearAlfvenWavesSuperposition

# Capability flags per API surface. This allows partial backends (e.g. Metal interpolation only).
HAS_GPU_INTERPOLATION = hasattr(firm3dpp, "test_gpu_interpolation")
HAS_GPU_DERIVATIVES = all(
    hasattr(firm3dpp, name)
    for name in [
        "test_derivatives_boozer",
        "test_derivatives_saw",
        "test_derivatives_saw_nok",
    ]
)
# Metal derivative flags (one per mode as they are added incrementally).
HAS_METAL_DERIV_BOOZER_VAC = hasattr(firm3dpp, "test_gpu_derivatives_boozer_vacuum")
HAS_METAL_DERIV_CARTESIAN  = hasattr(firm3dpp, "test_gpu_derivatives_cartesian")

# Metal timestep flags (one per mode as they are added incrementally).
HAS_METAL_TIMESTEP_BOOZER_VAC = hasattr(firm3dpp, "test_gpu_timestep_boozer_vacuum")

# Metal full tracing flags.
HAS_METAL_TRACING_BOOZER_VAC = hasattr(firm3dpp, "metal_boozer_vacuum_tracing")

HAS_GPU_TIMESTEP = all(
    hasattr(firm3dpp, name)
    for name in [
        "test_timestep_boozer",
        "test_timestep_saw",
        "test_timestep_saw_nok",
    ]
)

# CUDA backend provides the full GPU API. Metal provides interpolation + select derivatives.
HAS_CUDA_BACKEND = hasattr(firm3dpp, "boozer_gpu_tracing")
IS_METAL_BACKEND = HAS_GPU_INTERPOLATION and not HAS_CUDA_BACKEND

# Use looser tolerance for single-precision Metal interpolation.
INTERP_TOL = 1e-2 if IS_METAL_BACKEND else 1e-8


def get_field(boozmn_filename, n_metagrid_pts, vacuum):
    # start_time = time.time()
    bri = BoozerRadialInterpolant(boozmn_filename, 3, enforce_vacuum=vacuum)
    # print(f"Time to initialize BoozerRadialInterpolant: {time.time() - start_time} seconds")
    nfp = bri.nfp
    degree = 3
    # start_time = time.time()
    # print("bri field type:", bri.field_type)
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=n_metagrid_pts,
        ntheta_interp=n_metagrid_pts,
        nzeta_interp=n_metagrid_pts,
    )
    # print(f"Time to initialize InterpolatedBoozerField: {time.time() - start_time} seconds")
    # Even though bri isn't used further in this script, we need to return it,
    # or else it is garbage-collected, resulting in an error.
    return bri, field, nfp


### this function should be replaced by a function in the InterpolatedBoozerField field class
def construct_interpolant(field, nfp, saw_present=False):
    ns, ntheta, nzeta = 15, 15, 15

    if isinstance(field, ShearAlfvenWavesSuperposition):
        field = field.B0
        srange, trange, zrange, quad_info, maxJ = boozer_saw_interpolant(
            field, nfp, ns, ntheta, nzeta
        )
    else: # the field is an InterpolatedBoozerField (unperturbed)
        # print(field.field_type)
        if field.field_type == "vac":
            srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
                field, nfp, ns, ntheta, nzeta, vacuum=True
            )
        elif field.field_type == "": # implies finite beta
            srange, trange, zrange, quad_info, maxJ = boozer_interpolant(
                field, nfp, ns, ntheta, nzeta, vacuum=False
            )

    return srange, trange, zrange, quad_info, maxJ


def sample_test_points(n_test_pts):
    np.random.seed(1865)
    # generate test points
    s = np.random.uniform(low=0, high=0.95, size=(n_test_pts, 1))
    t = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    z = np.random.uniform(low=0, high=2 * np.pi, size=(n_test_pts, 1))
    stz = np.hstack((s, t, z))
    return stz

def run_interpolant_check(field, nfp, stz, saw_present=False, tol=1e-8):
    srange, trange, zrange, quad_info, maxJ = construct_interpolant(field, nfp, saw_present=saw_present)
    
    # evaluate interpolants
    if isinstance(field, ShearAlfvenWavesSuperposition):
        field = field.B0
        field.set_points(stz)
        modB = field.modB()
        modB_derivs = field.modB_derivs()
        G = field.G()
        dGds = field.dGds()
        I = field.I()
        dIds = field.dIds()
        iota = field.iota()
        diotads = field.diotads()
        cpu_interpolation = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, diotads))

        ## evaluate GPU interpolant
        stz = np.ascontiguousarray(stz)
        gpu_interpolation = firm3dpp.test_gpu_interpolation(
            quad_info, srange, trange, zrange, stz.copy(), "boozer_saw_vacuum", stz.shape[0]
        )
    else:
        if field.field_type == "vac": 
            # print("saw not present")
            # evaluate CPU interpolant
            field.set_points(stz)
            modB = field.modB()
            modB_derivs = field.modB_derivs()
            G = field.G()
            iota = field.iota()
            cpu_interpolation = np.hstack((modB, modB_derivs, G, iota))

            ## evaluate GPU interpolant
            stz = np.ascontiguousarray(stz)
            gpu_interpolation = firm3dpp.test_gpu_interpolation(
                quad_info, srange, trange, zrange, stz.copy(), "boozer_vacuum", stz.shape[0]
            )
        elif field.field_type == "": # implies finite beta
            # evaluate CPU interpolant
            field.set_points(stz)
            modB = field.modB()
            modB_derivs = field.modB_derivs()
            G = field.G()
            dGds = field.dGds()
            I = field.I()
            dIds = field.dIds()
            iota = field.iota()
            K = field.K()
            K_derivs = field.K_derivs()
            cpu_interpolation = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, K, K_derivs))

            # evaluate GPU interpolant
            stz = np.ascontiguousarray(stz)
            gpu_interpolation = firm3dpp.test_gpu_interpolation(
                quad_info, srange, trange, zrange, stz.copy(), "boozer", stz.shape[0]
            )



    gpu_interpolation = np.reshape(gpu_interpolation, (stz.shape[0], -1))

    # compute error
    error_is_small = np.isclose(gpu_interpolation, cpu_interpolation, rtol=tol, atol=tol).all()
    error = np.abs(cpu_interpolation - gpu_interpolation) / (np.abs(cpu_interpolation) +1)


    if (error.max() > tol):
        print("tolerance not satisfied in interpolant")
        row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
        print("stz:", stz[row_idx, :])
        print("cpu:", cpu_interpolation[row_idx, :])
        print("gpu:", gpu_interpolation[row_idx, :])
        print("error:", error[row_idx, :])

    return error_is_small

def run_derivatives_check(field, nfp, stz, vpar, vtotal, psi0, time=None, saw_present=False, saw_filename=None, tol=1e-8):

    srange, trange, zrange, quad_info, maxJ = construct_interpolant(field, nfp)
    ## evaluate derivatives
    if isinstance(field, ShearAlfvenWavesSuperposition):
        assert time is not None, "time array must be provided when testing derivatives with SAW"
        assert saw_filename is not None, "saw filename must be provided when testing derivatives with SAW"
        # evaluate CPU derivatives
        # print("computing simsopt derivatives")
        cpu_derivs = np.empty((stz.shape[0], 4))

        if field.B0.field_type == "vac":
            for i in range(stz.shape[0]):
                cpu_derivs[i, :] = firm3dpp.simsopt_derivs_saw(
                    field, stz[i, :], MASS, CHARGE, vtotal, vpar[i], time[i], "vacuum_saw"
                )
        elif field.B0.field_type == "nok": # NoK tracing
            for i in range(stz.shape[0]):
                cpu_derivs[i, :] = firm3dpp.simsopt_derivs_saw(
                    field, stz[i, :], MASS, CHARGE, vtotal, vpar[i], time[i], "nok_saw"
                )
        else:
            ValueError("Field type not recognized for SAW derivatives")

        saw_nharmonics = 5
        ## load saw data as arrays
        saw_data = np.load(saw_filename, allow_pickle=True)
        saw_data = saw_data[()]
        saw_omega = field.get_wave(0).omega
        # print("omega=", saw_omega)
        s = field.get_wave(0).phihat.get_s_basis()
        # print("s", s)
        # print(s[40:])
        saw_srange = (s[0], s[-1], len(s))

        saw_m = [field.get_wave(i).Phim for i in range(saw_nharmonics)]
        saw_n = [field.get_wave(i).Phin for i in range(saw_nharmonics)]
        saw_phihats = np.ascontiguousarray(
            np.column_stack(
                [
                    np.array([field.get_wave(i).phihat(s_val) for s_val in s])
                    for i in range(saw_nharmonics)
                ]
            )
        )
        ## evaluate GPU interpolant
        stz = np.ascontiguousarray(stz)
        vpar = np.ascontiguousarray(vpar)
        # print("calculating new derivatives")
        # start_time = time.time()

        if field.B0.field_type == "vac":
            gpu_derivs = firm3dpp.test_derivatives_saw(
                quad_info,
                srange,
                trange,
                zrange,
                saw_omega,
                saw_srange,
                saw_m,
                saw_n,
                saw_phihats,
                saw_nharmonics,
                stz,
                vpar,
                time,
                vtotal,
                MASS,
                CHARGE,
                psi0,
                stz.shape[0],
            )
        elif field.B0.field_type == "nok":
            gpu_derivs = firm3dpp.test_derivatives_saw_nok(
                quad_info,
                srange,
                trange,
                zrange,
                saw_omega,
                saw_srange,
                saw_m,
                saw_n,
                saw_phihats,
                saw_nharmonics,
                stz,
                vpar,
                time,
                vtotal,
                MASS,
                CHARGE,
                psi0,
                stz.shape[0],
            )
        else:
            ValueError("Field type not recognized for SAW derivatives")
    else:
        if field.field_type == "vac":
            # evaluate CPU derivatives
            # print("computing simsopt derivatives")
            cpu_derivs = np.empty((stz.shape[0], 4))
            # start_time = time.time()
            for i in range(stz.shape[0]):
                cpu_derivs[i, :] = firm3dpp.simsopt_derivs_boozer(
                    field, stz[i, :], MASS, CHARGE, vtotal, vpar[i], vacuum=True
                )
            # print(f"Time to compute simsopt derivatives: {time.time() - start_time} seconds")

            ## evaluate GPU interpolant
            stz = np.ascontiguousarray(stz)
            vpar = np.ascontiguousarray(vpar)
            if HAS_METAL_DERIV_BOOZER_VAC and not HAS_GPU_DERIVATIVES:
                time_dummy = np.zeros(stz.shape[0])
                gpu_derivs = firm3dpp.test_gpu_derivatives_boozer_vacuum(
                    quad_info,
                    srange,
                    trange,
                    zrange,
                    stz.copy(),
                    vpar,
                    time_dummy,
                    vtotal,
                    MASS,
                    CHARGE,
                    psi0,
                    stz.shape[0],
                )
            else:
                gpu_derivs = firm3dpp.test_derivatives_boozer(
                    quad_info,
                    srange,
                    trange,
                    zrange,
                    stz.copy(),
                    vpar,
                    vtotal,
                    MASS,
                    CHARGE,
                    psi0,
                    stz.shape[0],
                    vacuum=True,
                )
        elif field.field_type == "": # implies finite beta
            # evaluate CPU derivatives
            cpu_derivs = np.empty((stz.shape[0], 4))
            for i in range(stz.shape[0]):
                cpu_derivs[i, :] = firm3dpp.simsopt_derivs_boozer(
                    field, stz[i, :], MASS, CHARGE, vtotal, vpar[i], vacuum=False
                )

            stz = np.ascontiguousarray(stz)
            vpar = np.ascontiguousarray(vpar)
            gpu_derivs = firm3dpp.test_derivatives_boozer(
                quad_info, srange, trange, zrange,
                stz.copy(), vpar, vtotal, MASS, CHARGE, psi0,
                stz.shape[0], vacuum=False,
            )
    gpu_derivs = np.reshape(gpu_derivs, (stz.shape[0], 4))

    error_is_small = np.isclose(gpu_derivs, cpu_derivs, rtol=tol, atol=tol).all()
    error = np.abs(cpu_derivs - gpu_derivs) / (np.abs(cpu_derivs) + 1)

    if not error_is_small:
        row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
        print("stz:", stz[row_idx, :])
        print("cpu:", cpu_derivs[row_idx, :])
        print("gpu:", gpu_derivs[row_idx, :])
        print("rel error:", error[row_idx, :])

    return error_is_small

def run_timestep_check(field, nfp, stz, vpar, vtotal, psi0, time=None, saw_filename=None, tol=1e-8):

    srange, trange, zrange, quad_info, maxJ = construct_interpolant(field, nfp)

    if isinstance(field, ShearAlfvenWavesSuperposition):
        assert saw_filename is not None, "saw filename must be provided when testing timesteps with SAW"
        # evaluate CPU timestep
        field.B0.set_points(stz)
        mu_init = (vtotal**2 - vpar**2) / (2 * field.B0.modB()[:, 0])

        gc_tys, gc_zeta_hits = trace_particles_boozer_perturbed(
            field,
            stz,
            vpar,
            mu_init,
            tmax=1e-2,
            mass=MASS,
            charge=CHARGE,
            tol=1e-9,
            stopping_criteria=[IterationStoppingCriterion(0)],
            forget_exact_path=True,
        )

        saw_nharmonics = 5
        ## load saw data as arrays
        saw_data = np.load(saw_filename, allow_pickle=True)
        saw_data = saw_data[()]
        saw_omega = field.get_wave(0).omega
        # print("omega=", saw_omega)
        s = field.get_wave(0).phihat.get_s_basis()
        # print("s", s)
        # print(s[40:])
        saw_srange = (s[0], s[-1], len(s))

        saw_m = [field.get_wave(i).Phim for i in range(saw_nharmonics)]
        saw_n = [field.get_wave(i).Phin for i in range(saw_nharmonics)]
        saw_phihats = np.ascontiguousarray(
            np.column_stack(
                [
                    np.array([field.get_wave(i).phihat(s_val) for s_val in s])
                    for i in range(saw_nharmonics)
                ]
            )
        )
        stz = np.ascontiguousarray(stz)
        psi0 = field.B0.psi0

        if field.B0.field_type == "vac":
            last_time = firm3dpp.test_timestep_saw(
                quad_pts=quad_info,
                srange=srange,
                trange=trange,
                zrange=zrange,
                saw_omega=saw_omega,
                saw_srange=saw_srange,
                saw_m=saw_m,
                saw_n=saw_n,
                saw_phihats=saw_phihats,
                saw_nharmonics=saw_nharmonics,
                stz_init=stz,
                m=MASS,
                q=CHARGE,
                vtotal=np.sqrt(2 * ENERGY / MASS),
                vtang=vpar,
                time=time,
                tol=1e-9,
                psi0=psi0,
                nparticles=stz.shape[0],
            )
        elif field.B0.field_type == "nok":
            last_time = firm3dpp.test_timestep_saw_nok(
                quad_pts=quad_info,
                srange=srange,
                trange=trange,
                zrange=zrange,
                saw_omega=saw_omega,
                saw_srange=saw_srange,
                saw_m=saw_m,
                saw_n=saw_n,
                saw_phihats=saw_phihats,
                saw_nharmonics=saw_nharmonics,
                stz_init=stz,
                m=MASS,
                q=CHARGE,
                vtotal=np.sqrt(2 * ENERGY / MASS),
                vtang=vpar,
                time=time,
                tol=1e-9,
                psi0=psi0,
                nparticles=stz.shape[0],
            )
        last_time = np.reshape(last_time, (stz.shape[0], 5))
    else:
        if field.field_type == "vac":
            gc_tys, gc_zeta_hits = trace_particles_boozer(
                field,
                stz,
                vpar,
                tmax=1e-2,
                mass=MASS,
                charge=CHARGE,
                Ekin=ENERGY,
                tol=1e-9,
                stopping_criteria=[IterationStoppingCriterion(0)],
                forget_exact_path=True,
            )

            stz = np.ascontiguousarray(stz)
            psi0 = field.psi0
            if HAS_METAL_TIMESTEP_BOOZER_VAC and not HAS_GPU_TIMESTEP:
                last_time = firm3dpp.test_gpu_timestep_boozer_vacuum(
                    quad_pts=quad_info,
                    x1_range=srange,
                    x2_range=trange,
                    x3_range=zrange,
                    loc=stz.copy(),
                    vpar=vpar,
                    v_total=np.sqrt(2 * ENERGY / MASS),
                    m=MASS,
                    q=CHARGE,
                    psi0=psi0,
                    tol=1e-9,
                    n_points=stz.shape[0],
                )
            else:
                last_time = firm3dpp.test_timestep_boozer(
                    quad_pts=quad_info,
                    srange=srange,
                    trange=trange,
                    zrange=zrange,
                    stz_init=stz,
                    m=MASS,
                    q=CHARGE,
                    vtotal=np.sqrt(2 * ENERGY / MASS),
                    vtang=vpar,
                    tol=1e-9,
                    psi0=psi0,
                    nparticles=stz.shape[0],
                    vacuum=True,
                )
            last_time = np.reshape(last_time, (stz.shape[0], 5))
        elif field.field_type == "": # implies finite beta
            gc_tys, gc_zeta_hits = trace_particles_boozer(
                field,
                stz,
                vpar,
                tmax=1e-2,
                mass=MASS,
                charge=CHARGE,
                Ekin=ENERGY,
                tol=1e-9,
                stopping_criteria=[IterationStoppingCriterion(0)],
                forget_exact_path=True,
            )

            stz = np.ascontiguousarray(stz)
            psi0 = field.psi0
            last_time = firm3dpp.test_timestep_boozer(
                quad_pts=quad_info,
                srange=srange,
                trange=trange,
                zrange=zrange,
                stz_init=stz,
                m=MASS,
                q=CHARGE,
                vtotal=np.sqrt(2 * ENERGY / MASS),
                vtang=vpar,
                tol=1e-9,
                psi0=psi0,
                nparticles=stz.shape[0],
                vacuum=False,
            )
            last_time = np.reshape(last_time, (stz.shape[0], 5))

    # map to pseudo-cylindrical coordinates
    cpu_positions = np.array([x[-1] for x in gc_tys])
    cpu_positions = np.array(
        [
            [x[0], x[1] * np.cos(x[2]), x[1] * np.sin(x[2]), x[3], x[4]]
            for x in cpu_positions
        ]
    )
    gpu_final_positions = np.array(
        [
            [x[0], x[1] * np.cos(x[2]), x[1] * np.sin(x[2]), x[3], x[4]]
            for x in last_time
        ]
    )
    error_is_small = np.isclose(gpu_final_positions, cpu_positions, rtol=tol, atol=tol).all()
    error = np.abs(cpu_positions - gpu_final_positions) / (np.abs(cpu_positions) + 1)

    if (error.max() > tol):
        row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
        print("stz:", stz[row_idx, :])
        print("cpu:", cpu_positions[row_idx, :])
        print("gpu:", gpu_final_positions[row_idx, :])
        print("error:", error[row_idx, :])

    return error_is_small

def run_tracing_check(field, nfp, stz, vpar, vtotal, psi0, tmax=1e-7, tol=1e-2):
    srange, trange, zrange, quad_info, _ = construct_interpolant(field, nfp)

    gc_tys, _ = trace_particles_boozer(
        field,
        stz,
        vpar,
        tmax=tmax,
        mass=MASS,
        charge=CHARGE,
        Ekin=ENERGY,
        tol=1e-9,
        stopping_criteria=[],
        dt_save=tmax / 10,
    )

    stz = np.ascontiguousarray(stz)
    gpu_out = firm3dpp.metal_boozer_vacuum_tracing(
        quad_pts=quad_info,
        x1_range=srange,
        x2_range=trange,
        x3_range=zrange,
        loc=stz.copy(),
        vpar=vpar,
        v_total=vtotal,
        m=MASS,
        q=CHARGE,
        psi0=psi0,
        tmax=tmax,
        tol=1e-9,
        n_points=stz.shape[0],
    )
    gpu_out = np.reshape(gpu_out, (stz.shape[0], 5))

    # Map [t, s, theta, zeta, v_par] -> [t, s*cos(theta), s*sin(theta), zeta, v_par]
    # so the angular ambiguity in theta doesn't affect the comparison.
    cpu_final = np.array([x[-1] for x in gc_tys])
    cpu_pos = np.array([[x[0], x[1]*np.cos(x[2]), x[1]*np.sin(x[2]), x[3], x[4]] for x in cpu_final])
    gpu_pos = np.array([[x[0], x[1]*np.cos(x[2]), x[1]*np.sin(x[2]), x[3], x[4]] for x in gpu_out])

    error = np.abs(cpu_pos - gpu_pos) / (np.abs(cpu_pos) + 1)
    error_is_small = error.max() <= tol

    if not error_is_small:
        row_idx = np.unravel_index(np.argmax(error), error.shape)[0]
        print("stz:", stz[row_idx, :])
        print("cpu:", cpu_pos[row_idx, :])
        print("gpu:", gpu_pos[row_idx, :])
        print("error:", error[row_idx, :])

    return error_is_small

@unittest.skipUnless(
    HAS_GPU_INTERPOLATION, "No GPU interpolation backend (neither CUDA nor Metal) available"
)
class TestGPUTracing(unittest.TestCase):
    def test_boozer_vacuum(self):
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = True
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        n_test_pts = 10000
        stz = sample_test_points(n_test_pts)

        tol = INTERP_TOL

        ### test interpolant
        is_small = run_interpolant_check(field, nfp, stz, tol=tol)
        self.assertTrue(is_small)

        ### test derivatives (CUDA full backend or Metal boozer-vacuum path)
        if HAS_GPU_DERIVATIVES or HAS_METAL_DERIV_BOOZER_VAC:
            VELOCITY = np.sqrt(2 * ENERGY / MASS)
            vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
            deriv_tol = INTERP_TOL if IS_METAL_BACKEND else 1e-8
            is_small = run_derivatives_check(field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol=deriv_tol)
            self.assertTrue(is_small)

        if not ((HAS_GPU_DERIVATIVES and HAS_GPU_TIMESTEP) or HAS_METAL_TIMESTEP_BOOZER_VAC):
            return

        ### test timesteps
        timestep_tol = INTERP_TOL if IS_METAL_BACKEND else 1e-8
        is_small = run_timestep_check(field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol=timestep_tol)
        self.assertTrue(is_small)

        ### test full tracing loop (Metal only for now; use fewer points to keep runtime reasonable)
        if HAS_METAL_TRACING_BOOZER_VAC and IS_METAL_BACKEND:
            n_trace_pts = 100
            stz_trace = sample_test_points(n_trace_pts)
            vpar_trace = np.random.uniform(-VELOCITY, VELOCITY, (n_trace_pts,))
            is_small = run_tracing_check(field, nfp, stz_trace, vpar_trace, VELOCITY, field.psi0)
            self.assertTrue(is_small)

    def test_boozer_finite_beta(self):
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = False
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        n_test_pts = 10000
        stz = sample_test_points(n_test_pts)

        tol = INTERP_TOL

        ### test interpolant
        is_small = run_interpolant_check(field, nfp, stz, tol=tol)
        self.assertTrue(is_small)

        if not (HAS_GPU_DERIVATIVES and HAS_GPU_TIMESTEP):
            return

        ### test derivatives
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        is_small = run_derivatives_check(field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol=1e-8)
        self.assertTrue(is_small)

        ### test timesteps
        is_small = run_timestep_check(field, nfp, stz, vpar_init, VELOCITY, field.psi0, tol=1e-8)
        self.assertTrue(is_small)

    def test_boozer_vacuum_saw(self):
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = True
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        ### set up SAW
        saw_filename = "./examples/tracing_with_AE/ae.npy"
        saw = ShearAlfvenWavesSuperposition.from_ae3d(
            eigenvector=AE3DEigenvector.load_from_numpy(
                filename=saw_filename,
            ),
            B0=field,
            max_dB_normal_by_B0=5e-3,
            minor_radius_meters=1.7,
        )

        n_test_pts = 10000
        stz = sample_test_points(n_test_pts)
        tol = INTERP_TOL

        ### test interpolant
        is_small = run_interpolant_check(saw, nfp, stz, saw_present=True, tol=tol)
        self.assertTrue(is_small)

        if not (HAS_GPU_DERIVATIVES and HAS_GPU_TIMESTEP):
            return

        ## test derivatives
        tol = 1e-8
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        time = np.random.uniform(low=0, high=1e-3, size=(n_test_pts,))
        is_small = run_derivatives_check(saw, nfp, stz, vpar_init, VELOCITY, field.psi0, time=time, saw_present=True, saw_filename=saw_filename, tol=tol)
        self.assertTrue(is_small)

        ### test timesteps
        is_small = run_timestep_check(saw, nfp, stz, vpar_init, VELOCITY, field.psi0, time=time, saw_filename=saw_filename, tol=tol)
        self.assertTrue(is_small)
        
    def test_boozer_nok_saw(self):
        n_metagrid_pts = 15

        ### Vacuum case
        boozmn_filename = "examples/inputs/boozmn_aten_rescaled_low_res.nc"
        vacuum = True
        bri, field, nfp = get_field(boozmn_filename, n_metagrid_pts, vacuum)

        ### set up SAW
        saw_filename = "./examples/tracing_with_AE/ae.npy"
        saw = ShearAlfvenWavesSuperposition.from_ae3d(
            eigenvector=AE3DEigenvector.load_from_numpy(
                filename=saw_filename,
            ),
            B0=field,
            max_dB_normal_by_B0=5e-3,
            minor_radius_meters=1.7,
        )

        n_test_pts = 10000
        stz = sample_test_points(n_test_pts)
        tol = INTERP_TOL

        ### test interpolant
        is_small = run_interpolant_check(saw, nfp, stz, saw_present=True, tol=tol)
        self.assertTrue(is_small)

        if not (HAS_GPU_DERIVATIVES and HAS_GPU_TIMESTEP):
            return

        ## test derivatives
        tol = 1e-8
        VELOCITY = np.sqrt(2 * ENERGY / MASS)
        vpar_init = np.random.uniform(-VELOCITY, VELOCITY, (n_test_pts,))
        time = np.random.uniform(low=0, high=1e-3, size=(n_test_pts,))
        is_small = run_derivatives_check(saw, nfp, stz, vpar_init, VELOCITY, field.psi0, time=time, saw_present=True, saw_filename=saw_filename, tol=tol)
        self.assertTrue(is_small)

        ### test timesteps
        is_small = run_timestep_check(saw, nfp, stz, vpar_init, VELOCITY, field.psi0, time=time, saw_filename=saw_filename, tol=tol)
        self.assertTrue(is_small)


if __name__ == "__main__":
    print("Running GPU tracing tests...")
    unittest.main()
