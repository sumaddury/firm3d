#include <metal_stdlib>
using namespace metal;

// python -m pip install -e . --no-build-isolation
// python -m pytest tests/field/test_gpu.py -v

enum {
    F3D_RHS_CARTESIAN_VACUUM = 0,
    F3D_RHS_BOOZER_VACUUM = 1,
    F3D_RHS_BOOZER_SAW_VACUUM = 2,
    F3D_RHS_BOOZER = 3
};

struct InterpolationTestConstants {
    int n_x2;
    int n_x3;
    int n_x23;
    int n_fields;
    int n_points;
    int rhs_mode;

    int x1_count;
    int x2_count;
    int x3_count;

    float x1_start;
    float x2_start;
    float x2_period;
    float x3_start;
    float x3_period;

    float x1_step;
    float x2_step;
    float x3_step;
};

struct DerivativeConstants {
    int x1_count;
    int x2_count;
    int x3_count;

    int n_x2;
    int n_x3;
    int n_x23;
    int n_points;

    float x1_start;
    float x2_start;
    float x2_period;
    float x3_start;
    float x3_period;

    float x1_step;
    float x2_step;
    float x3_step;

    float mass;
    float charge;
    float psi0;
    float v_total;

    float tol;   // adaptive step tolerance (used by timestep kernel)
    float tmax;  // maximum timestep cap (used by timestep kernel)
};

constant float f3d_dp5_wgts[7][7] = {
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f / 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {3.0f / 40.0f, 9.0f / 40.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {44.0f / 45.0f, -56.0f / 15.0f, 32.0f / 9.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {19372.0f / 6561.0f, -25360.0f / 2187.0f, 64448.0f / 6561.0f, -212.0f / 729.0f, 0.0f, 0.0f, 0.0f},
    {9017.0f / 3168.0f, -355.0f / 33.0f, 46732.0f / 5247.0f, 49.0f / 176.0f, -5103.0f / 18656.0f, 0.0f, 0.0f},
    {35.0f / 384.0f, 0.0f, 500.0f / 1113.0f, 125.0f / 192.0f, -2187.0f / 6784.0f, 11.0f / 84.0f, 0.0f}
};

constant float f3d_dp5_t_wgts[7] = {
    0.0f, 1.0f / 5.0f, 3.0f / 10.0f, 4.0f / 5.0f, 8.0f / 9.0f, 1.0f, 1.0f
};

inline float f3d_shape_fn(float x, int i) {
    switch (i) {
        case 0:
            return (1.0f - x) * (2.0f - x) * (3.0f - x) / 6.0f;
        case 1:
            return x * (2.0f - x) * (3.0f - x) / 2.0f;
        case 2:
            return x * (x - 1.0f) * (3.0f - x) / 2.0f;
        case 3:
            return x * (x - 1.0f) * (x - 2.0f) / 6.0f;
        default:
            return 0.0f;
    }
}

inline float f3d_positive_mod(float x, float period) {
    float out = fmod(x, period);
    if (out < 0.0f) {
        out += period;
    }
    return out;
}

template <int kRhsMode>
inline void f3d_map_to_grid(
    thread float3& interp_pt,
    thread float* x_temp,
    thread bool& symmetry_exploited,
    constant DerivativeConstants& c
) {
    const float two_pi = 6.28318530717958647692f;

    float x = x_temp[1];
    float y = x_temp[2];
    float z = x_temp[3];

    float r = sqrt(x * x + y * y);
    float phi = atan2(y, x);
    phi = f3d_positive_mod(phi, c.x2_period);

    symmetry_exploited = z < 0.0f;
    if (symmetry_exploited) {
        z = -z;
        phi = f3d_positive_mod(two_pi - phi, c.x2_period);
    }

    interp_pt = float3(r, phi, z);
}

template <>
inline void f3d_map_to_grid<F3D_RHS_BOOZER_VACUUM>(
    thread float3& interp_pt,
    thread float* x_temp,
    thread bool& symmetry_exploited,
    constant DerivativeConstants& c
) {
    const float pi = 3.14159265358979323846f;
    const float two_pi = 6.28318530717958647692f;

    float x1 = x_temp[1];
    float x2 = x_temp[2];
    float zeta = x_temp[3];

    float s = sqrt(x1 * x1 + x2 * x2);
    float theta = atan2(x2, x1);
    float t = f3d_positive_mod(theta, two_pi);
    zeta = f3d_positive_mod(zeta, c.x3_period);

    symmetry_exploited = t > pi;
    if (symmetry_exploited) {
        zeta = c.x3_period - zeta;
        t = two_pi - t;
    }

    interp_pt = float3(s, t, zeta);
}

template <int kRhsMode>
inline void f3d_build_state(
    thread float* x_temp,
    int deriv_id,
    thread bool* symmetry_exploited,
    thread int* index_i,
    thread int* index_j,
    thread int* index_k,
    thread float* x1_shape,
    thread float* x2_shape,
    thread float* x3_shape,
    thread float* state,
    thread float* derivs,
    thread float* t,
    thread float* dt,
    constant DerivativeConstants& c
) {
    x_temp[0] = t[0] + f3d_dp5_t_wgts[deriv_id] * dt[0];
    for (int i = 0; i < 4; ++i) {
        x_temp[i + 1] = state[i];
    }

    for (int j = 0; j < deriv_id; ++j) {
        for (int i = 0; i < 4; ++i) {
            x_temp[i + 1] += dt[0] * f3d_dp5_wgts[deriv_id][j] * derivs[6 * j + i];
        }
    }

    float3 interp_pt;
    bool sym = false;
    f3d_map_to_grid<kRhsMode>(interp_pt, x_temp, sym, c);
    symmetry_exploited[0] = sym;

    float x1 = interp_pt.x;
    float x2 = interp_pt.y;
    float x3 = interp_pt.z;

    int i = 3 * (int((x1 - c.x1_start) / c.x1_step) / 3);
    int j = 3 * (int((x2 - c.x2_start) / c.x2_step) / 3);
    int k = 3 * (int((x3 - c.x3_start) / c.x3_step) / 3);

    i = min(i, c.x1_count - 4);
    j = min(j, c.x2_count - 4);
    k = min(k, c.x3_count - 4);
    i = max(i, 0);
    j = max(j, 0);
    k = max(k, 0);

    float x1_rel = (x1 - float(i) * c.x1_step - c.x1_start) / c.x1_step;
    float x2_rel = (x2 - float(j) * c.x2_step - c.x2_start) / c.x2_step;
    float x3_rel = (x3 - float(k) * c.x3_step - c.x3_start) / c.x3_step;

    for (int ii = 0; ii < 4; ++ii) {
        x1_shape[ii] = f3d_shape_fn(x1_rel, ii);
        x2_shape[ii] = f3d_shape_fn(x2_rel, ii);
        x3_shape[ii] = f3d_shape_fn(x3_rel, ii);
    }

    index_i[0] = i / 3;
    index_j[0] = j / 3;
    index_k[0] = k / 3;
}

inline void f3d_interpolate(
    device const float* quad_pts,
    thread float* interp,
    int ci, int cj, int ck,
    int n_x23, int n_x3, int n_fields,
    thread const float* x1_shape,
    thread const float* x2_shape,
    thread const float* x3_shape
) {
    float w[64];
    thread float* wp = w;
    for (int ii = 0; ii < 4; ++ii) {
        float s1 = x1_shape[ii];
        for (int jj = 0; jj < 4; ++jj) {
            float s12 = s1 * x2_shape[jj];
            *wp++ = s12 * x3_shape[0];
            *wp++ = s12 * x3_shape[1];
            *wp++ = s12 * x3_shape[2];
            *wp++ = s12 * x3_shape[3];
        }
    }

    for (int zz = 0; zz < n_fields; ++zz) {
        interp[zz] = 0.0f;
    }

    device const float* dp = quad_pts + n_fields * 64 * (ci * n_x23 + cj * n_x3 + ck);
    for (int idx = 0; idx < 64; ++idx) {
        float wt = w[idx];
        for (int zz = 0; zz < n_fields; ++zz) {
            interp[zz] += dp[zz] * wt;
        }
        dp += n_fields;
    }
}

template <int kRhsMode>
inline void f3d_calc_derivs(
    device const float* quad_pts,
    thread float* derivs,
    int deriv_id,
    thread float* x_temp,
    thread bool* symmetry_exploited,
    thread int* index_i,
    thread int* index_j,
    thread int* index_k,
    thread float* x1_shape,
    thread float* x2_shape,
    thread float* x3_shape,
    thread float* mu,
    int nparticles_blk,
    constant DerivativeConstants& c
) {}

// Guiding-center vacuum Boozer RHS.
// State layout in x_temp: [t, x1, x2, zeta, v_par]
//   where x1 = sqrt(s)*cos(theta), x2 = sqrt(s)*sin(theta).
// Quad-pts field layout: [modB, dmodBds, dmodBdtheta, dmodBdzeta, G, iota] (6 fields).
// Derivs layout: derivs[6*deriv_id + i], i in 0..5.
//   0: dx1/dt   1: dx2/dt   2: dzeta/dt   3: dv_par/dt
//   4: modB (for mu)   5: G (diagnostic)
template <>
inline void f3d_calc_derivs<F3D_RHS_BOOZER_VACUUM>(
    device const float* quad_pts,
    thread float* derivs,
    int deriv_id,
    thread float* x_temp,
    thread bool* symmetry_exploited,
    thread int* index_i,
    thread int* index_j,
    thread int* index_k,
    thread float* x1_shape,
    thread float* x2_shape,
    thread float* x3_shape,
    thread float* mu,
    int nparticles_blk,
    constant DerivativeConstants& c
) {
    float interp[6];
    f3d_interpolate(quad_pts, interp,
                    index_i[0], index_j[0], index_k[0],
                    c.n_x23, c.n_x3, 6,
                    x1_shape, x2_shape, x3_shape);

    float modB = interp[0];
    float dmodBds = interp[1];
    float dmodBdtheta = interp[2];
    float dmodBdzeta = interp[3];
    float G = interp[4];
    float iota = interp[5];

    if (symmetry_exploited[0]) {
        dmodBdtheta *= -1.0f;
        dmodBdzeta  *= -1.0f;
    }

    float x1 = x_temp[1];
    float x2 = x_temp[2];
    float s = sqrt(x1 * x1 + x2 * x2);
    float theta = atan2(x2, x1);
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    float v_par = x_temp[4];
    float mu_val = mu[0];

    float modB_over_G = modB / G;
    float fak1 = c.mass * v_par * v_par / modB + c.mass * mu_val;
    float sdot = -dmodBdtheta * fak1 / (c.charge * c.psi0);
    float tdot = dmodBds * fak1 / (c.charge * c.psi0) + iota * v_par * modB_over_G;

    derivs[6 * deriv_id + 0] = sdot * cos_theta - s * sin_theta * tdot;
    derivs[6 * deriv_id + 1] = sdot * sin_theta + s * cos_theta * tdot;
    derivs[6 * deriv_id + 2] = v_par * modB_over_G;
    derivs[6 * deriv_id + 3] = -(iota * dmodBdtheta + dmodBdzeta) * mu_val * modB_over_G;
    derivs[6 * deriv_id + 4] = modB;
    derivs[6 * deriv_id + 5] = G;
}

// Guiding-center vacuum Cartesian RHS.
// State layout in x_temp: [t, x, y, z, v_par]
// Quad-pts field layout: [Br, Bphi, Bz, GradAbsB_r, GradAbsB_phi, GradAbsB_z, boundary_dist] (7 fields).
// Derivs layout: derivs[6*deriv_id + i], i in 0..5.
//   0: dx/dt   1: dy/dt   2: dz/dt   3: dv_par/dt
//   4: AbsB (for mu)   5: boundary dist fn
template <>
inline void f3d_calc_derivs<F3D_RHS_CARTESIAN_VACUUM>(
    device const float* quad_pts,
    thread float* derivs,
    int deriv_id,
    thread float* x_temp,
    thread bool* symmetry_exploited,
    thread int* index_i,
    thread int* index_j,
    thread int* index_k,
    thread float* x1_shape,
    thread float* x2_shape,
    thread float* x3_shape,
    thread float* mu,
    int nparticles_blk,
    constant DerivativeConstants& c
) {
    float interp[7];
    f3d_interpolate(quad_pts, interp,
                    index_i[0], index_j[0], index_k[0],
                    c.n_x23, c.n_x3, 7,
                    x1_shape, x2_shape, x3_shape);

    float B_r = interp[0];
    float B_phi = interp[1];
    float B_z = interp[2];
    float GradAbsB_r = interp[3];
    float GradAbsB_phi = interp[4];
    float GradAbsB_z = interp[5];

    if (symmetry_exploited[0]) {
        B_r *= -1.0f;
        GradAbsB_phi *= -1.0f;
        GradAbsB_z *= -1.0f;
    }

    float x = x_temp[1];
    float y = x_temp[2];
    float v_par = x_temp[4];
    float phi = atan2(y, x);
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    float B_x = cos_phi * B_r   - sin_phi * B_phi;
    float B_y = sin_phi * B_r   + cos_phi * B_phi;
    float GradAbsB_x = cos_phi * GradAbsB_r - sin_phi * GradAbsB_phi;
    float GradAbsB_y = sin_phi * GradAbsB_r + cos_phi * GradAbsB_phi;

    float AbsB = sqrt(B_x * B_x + B_y * B_y + B_z * B_z);
    float mu_val = mu[0];
    float v_perp2 = 2.0f * mu_val * AbsB;
    float AbsB2 = AbsB * AbsB;
    float fak1 = v_par / AbsB;
    float fak2 = (c.mass / (c.charge * AbsB2 * AbsB)) * (0.5f * v_perp2 + v_par * v_par);

    derivs[6 * deriv_id + 0] = fak1 * B_x + fak2 * (B_y * GradAbsB_z - B_z * GradAbsB_y);
    derivs[6 * deriv_id + 1] = fak1 * B_y + fak2 * (B_z * GradAbsB_x - B_x * GradAbsB_z);
    derivs[6 * deriv_id + 2] = fak1 * B_z + fak2 * (B_x * GradAbsB_y - B_y * GradAbsB_x);
    derivs[6 * deriv_id + 3] = -mu_val * (B_x * GradAbsB_x + B_y * GradAbsB_y + B_z * GradAbsB_z) / AbsB;
    derivs[6 * deriv_id + 4] = AbsB;
    derivs[6 * deriv_id + 5] = interp[6];
}

template <int kRhsMode>
inline void f3d_test_derivs_impl(
    device const float* quad_pts,
    device const float* loc,
    device const float* vpar_buf,
    device const float* time_buf,
    device float*       out,
    constant DerivativeConstants& c,
    uint gid
) {
    int p = int(gid);
    if (p >= c.n_points) return;

    float c1 = loc[3 * p + 0];
    float ang = loc[3 * p + 1];
    float c3 = loc[3 * p + 2];
    float vpar_val = vpar_buf[p];

    float state[4] = { c1 * cos(ang), c1 * sin(ang), c3, vpar_val };

    float x_temp[5];
    float derivs[42];  // 7 stages * 6 fields
    float x1_shape[4], x2_shape[4], x3_shape[4];
    bool  symmetry_exploited[1];
    int   index_i[1], index_j[1], index_k[1];
    float mu[1];
    float t[1], dt[1];

    t[0] = 0.0f; dt[0] = 0.0f;
    f3d_build_state<kRhsMode>(x_temp, 0, symmetry_exploited,
                               index_i, index_j, index_k,
                               x1_shape, x2_shape, x3_shape,
                               state, derivs, t, dt, c);
    mu[0] = -1.0f;
    f3d_calc_derivs<kRhsMode>(quad_pts, derivs, 0, x_temp, symmetry_exploited,
                               index_i, index_j, index_k,
                               x1_shape, x2_shape, x3_shape, mu, 0, c);

    float modB    = derivs[4];
    float v_perp2 = c.v_total * c.v_total - vpar_val * vpar_val;
    mu[0] = v_perp2 / (2.0f * modB);

    t[0] = time_buf[p]; dt[0] = 0.0f;
    f3d_build_state<kRhsMode>(x_temp, 0, symmetry_exploited,
                               index_i, index_j, index_k,
                               x1_shape, x2_shape, x3_shape,
                               state, derivs, t, dt, c);
    f3d_calc_derivs<kRhsMode>(quad_pts, derivs, 0, x_temp, symmetry_exploited,
                               index_i, index_j, index_k,
                               x1_shape, x2_shape, x3_shape, mu, 0, c);

    out[4 * p + 0] = derivs[0];
    out[4 * p + 1] = derivs[1];
    out[4 * p + 2] = derivs[2];
    out[4 * p + 3] = derivs[3];
}

kernel void test_gpu_derivs_boozer_vacuum_kernel(
    device const float* quad_pts [[buffer(0)]],
    device const float* loc      [[buffer(1)]],
    device const float* vpar_buf [[buffer(2)]],
    device const float* time_buf [[buffer(3)]],
    device float*       out      [[buffer(4)]],
    constant DerivativeConstants& c [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    f3d_test_derivs_impl<F3D_RHS_BOOZER_VACUUM>(quad_pts, loc, vpar_buf, time_buf, out, c, gid);
}

kernel void test_gpu_derivs_cartesian_kernel(
    device const float* quad_pts [[buffer(0)]],
    device const float* loc      [[buffer(1)]],
    device const float* vpar_buf [[buffer(2)]],
    device const float* time_buf [[buffer(3)]],
    device float*       out      [[buffer(4)]],
    constant DerivativeConstants& c [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    f3d_test_derivs_impl<F3D_RHS_CARTESIAN_VACUUM>(quad_pts, loc, vpar_buf, time_buf, out, c, gid);
}

// One adaptive Dormand-Prince 5(4) step per particle, Boozer-vacuum mode.
// Input loc: (n_points, 3) = [s, theta, zeta] (polar form).
// Input vpar_buf: (n_points,) parallel velocity.
// Output: (n_points, 5) = [t, x1, x2, zeta, v_par] after one accepted step.
//   The wrapper converts (x1, x2) back to (s, theta) before returning to Python.
kernel void test_gpu_timestep_boozer_vacuum_kernel(
    device const float* quad_pts [[buffer(0)]],
    device const float* loc      [[buffer(1)]],
    device const float* vpar_buf [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant DerivativeConstants& c [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int p = int(gid);
    if (p >= c.n_points) return;

    float c1      = loc[3 * p + 0];  // s
    float ang     = loc[3 * p + 1];  // theta
    float c3      = loc[3 * p + 2];  // zeta
    float vpar_val = vpar_buf[p];

    // Convert (s, theta) -> (x1, x2) for internal state representation.
    float state[4] = { c1 * cos(ang), c1 * sin(ang), c3, vpar_val };

    float x_temp[5];
    float derivs[42];  // 7 stages * 6 fields
    float x1_shape[4], x2_shape[4], x3_shape[4];
    bool  sym[1]  = { false };
    int   ii[1], jj[1], kk[1];
    float mu[1];
    float t[1]  = { 0.0f };
    float dt[1] = { 0.0f };

    // --- Setup: compute mu and initial dt from a dummy stage-0 eval ---
    f3d_build_state<F3D_RHS_BOOZER_VACUUM>(x_temp, 0, sym, ii, jj, kk,
                                           x1_shape, x2_shape, x3_shape,
                                           state, derivs, t, dt, c);
    mu[0] = -1.0f;
    f3d_calc_derivs<F3D_RHS_BOOZER_VACUUM>(quad_pts, derivs, 0, x_temp, sym, ii, jj, kk,
                                            x1_shape, x2_shape, x3_shape, mu, 0, c);

    float modB = derivs[4];
    float G    = derivs[5];
    mu[0] = (c.v_total * c.v_total - vpar_val * vpar_val) / (2.0f * modB);

    // Cap max step at one quarter of a transit (same heuristic as CUDA).
    const float pi = 3.14159265358979f;
    float dtmax = min((G / modB) * (0.5f * pi) / c.v_total, c.tmax);
    dt[0] = 1e-3f * dtmax;

    // --- DP5 error coefficients (difference between 5th- and 4th-order weights) ---
    const float bhat1 =  71.0f / 57600.0f;
    const float bhat3 = -71.0f / 16695.0f;
    const float bhat4 =  71.0f / 1920.0f;
    const float bhat5 = -17253.0f / 339200.0f;
    const float bhat6 =  22.0f / 525.0f;
    const float bhat7 =  -1.0f / 40.0f;

    // --- Adaptive loop: keep trying until one step is accepted ---
    for (int iter = 0; iter < 1000 && t[0] == 0.0f; ++iter) {

        // Compute all 7 DP5 stage derivatives.
        for (int k = 0; k < 7; ++k) {
            f3d_build_state<F3D_RHS_BOOZER_VACUUM>(x_temp, k, sym, ii, jj, kk,
                                                   x1_shape, x2_shape, x3_shape,
                                                   state, derivs, t, dt, c);
            f3d_calc_derivs<F3D_RHS_BOOZER_VACUUM>(quad_pts, derivs, k, x_temp, sym, ii, jj, kk,
                                                    x1_shape, x2_shape, x3_shape, mu, 0, c);
        }

        // Compute scaled max error across the 4 state components.
        float max_err = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float err = dt[0] * (bhat1 * derivs[6*0 + i]
                               + bhat3 * derivs[6*2 + i]
                               + bhat4 * derivs[6*3 + i]
                               + bhat5 * derivs[6*4 + i]
                               + bhat6 * derivs[6*5 + i]
                               + bhat7 * derivs[6*6 + i]);
            // Scale by tolerance + magnitude of state + predicted change.
            err = fabs(err) / (c.tol + c.tol * (fabs(state[i]) + dt[0] * fabs(derivs[i])));
            max_err = max(max_err, err);
        }

        // Compute new step size (same formula as CUDA adjust_time).
        float exponent = 0.0f;
        if (max_err > 1.0f) exponent = -1.0f / 3.0f;
        if (max_err < 0.5f) exponent = -1.0f / 5.0f;
        float dt_new = dt[0] * 0.9f * pow(max_err, exponent);
        dt_new = clamp(dt_new, 0.2f * dt[0], 5.0f * dt[0]);

        if (max_err <= 1.0f) {
            // Step accepted: do not change dt_new if error is moderate.
            if (max_err > 0.5f) dt_new = dt[0];
            t[0]  += dt[0];
            dt[0]  = min(dt_new, dtmax - t[0]);
            for (int i = 0; i < 4; ++i) state[i] = x_temp[i + 1];
        } else {
            // Step rejected: shrink dt and retry.
            dt[0] = dt_new;
        }
    }

    // Write [t, x1, x2, zeta, v_par]; wrapper converts (x1,x2) -> (s,theta).
    out[5 * p + 0] = t[0];
    out[5 * p + 1] = state[0];
    out[5 * p + 2] = state[1];
    out[5 * p + 3] = state[2];
    out[5 * p + 4] = state[3];
}

kernel void test_gpu_interpolation_kernel(
    device const float* quad_pts [[buffer(0)]],
    device const float* loc [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant InterpolationTestConstants& c [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= c.n_points) {
        return;
    }

    int p = int(gid);
    float x = loc[3 * p + 0];
    float y = loc[3 * p + 1];
    float z = loc[3 * p + 2];

    float interp_x1 = 0.0f;
    float interp_x2 = 0.0f;
    float interp_x3 = 0.0f;
    bool symmetry_exploited = false;
    const float two_pi = 6.28318530717958647692f;

    if (c.rhs_mode == F3D_RHS_CARTESIAN_VACUUM) {
        float r = sqrt(x * x + y * y);
        float phi = atan2(y, x);
        phi = f3d_positive_mod(phi, c.x2_period);

        symmetry_exploited = z < 0.0f;
        if (symmetry_exploited) {
            z = -z;
            phi = f3d_positive_mod(two_pi - phi, c.x2_period);
        }
        interp_x1 = r;
        interp_x2 = phi;
        interp_x3 = z;
    } else {
        float s = sqrt(x * x + y * y);
        float theta = atan2(y, x);
        float t = f3d_positive_mod(theta, two_pi);
        float zeta = f3d_positive_mod(z, c.x3_period);

        symmetry_exploited = t > 3.14159265358979323846f;
        if (symmetry_exploited) {
            zeta = c.x3_period - zeta;
            t = two_pi - t;
        }
        interp_x1 = s;
        interp_x2 = t;
        interp_x3 = zeta;
    }

    int i = 3 * (int((interp_x1 - c.x1_start) / c.x1_step) / 3);
    int j = 3 * (int((interp_x2 - c.x2_start) / c.x2_step) / 3);
    int k = 3 * (int((interp_x3 - c.x3_start) / c.x3_step) / 3);

    i = min(i, c.x1_count - 4);
    j = min(j, c.x2_count - 4);
    k = min(k, c.x3_count - 4);
    i = max(i, 0);
    j = max(j, 0);
    k = max(k, 0);

    int cell_i = i / 3;
    int cell_j = j / 3;
    int cell_k = k / 3;

    float x1_rel = (interp_x1 - float(i) * c.x1_step - c.x1_start) / c.x1_step;
    float x2_rel = (interp_x2 - float(j) * c.x2_step - c.x2_start) / c.x2_step;
    float x3_rel = (interp_x3 - float(k) * c.x3_step - c.x3_start) / c.x3_step;

    float s1[4];
    float s2[4];
    float s3[4];
    for (int ii = 0; ii < 4; ++ii) {
        s1[ii] = f3d_shape_fn(x1_rel, ii);
        s2[ii] = f3d_shape_fn(x2_rel, ii);
        s3[ii] = f3d_shape_fn(x3_rel, ii);
    }

    float interp[12];
    f3d_interpolate(quad_pts, interp,
                    cell_i, cell_j, cell_k,
                    c.n_x23, c.n_x3, c.n_fields,
                    s1, s2, s3);
    for (int zz = 0; zz < c.n_fields; ++zz) {
        out[p * c.n_fields + zz] = interp[zz];
    }

    if (symmetry_exploited) {
        if (c.rhs_mode == F3D_RHS_CARTESIAN_VACUUM) {
            out[p * c.n_fields + 0] *= -1.0f;
            out[p * c.n_fields + 4] *= -1.0f;
            out[p * c.n_fields + 5] *= -1.0f;
        } else if (c.rhs_mode == F3D_RHS_BOOZER_VACUUM || c.rhs_mode == F3D_RHS_BOOZER_SAW_VACUUM) {
            out[p * c.n_fields + 2] *= -1.0f;
            out[p * c.n_fields + 3] *= -1.0f;
        } else if (c.rhs_mode == F3D_RHS_BOOZER) {
            out[p * c.n_fields + 2] *= -1.0f;
            out[p * c.n_fields + 3] *= -1.0f;
            if (c.n_fields >= 12) {
                out[p * c.n_fields + 9] *= -1.0f;
            }
        }
    }
}
