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
    float v_par = x_temp[4];
    float mu_val = mu[0];

    float fak1 = c.mass * v_par * v_par / modB + c.mass * mu_val;
    float sdot = -dmodBdtheta * fak1 / (c.charge * c.psi0);
    float tdot = dmodBds * fak1 / (c.charge * c.psi0) + iota * v_par * modB / G;

    derivs[6 * deriv_id + 0] = sdot * cos(theta) - s * sin(theta) * tdot;
    derivs[6 * deriv_id + 1] = sdot * sin(theta) + s * cos(theta) * tdot;
    derivs[6 * deriv_id + 2] = v_par * modB / G;
    derivs[6 * deriv_id + 3] = -(iota * dmodBdtheta + dmodBdzeta) * mu_val * modB / G;
    derivs[6 * deriv_id + 4] = modB;
    derivs[6 * deriv_id + 5] = G;
}

// One thread per particle. Runs the two-phase setup (dummy call to get modB → compute mu,
// then real call at the given time) and writes the first 4 derivatives to out.
// Only the F3D_RHS_BOOZER_VACUUM physics are wired in; other modes would need
// additional specialisations of f3d_build_state / f3d_calc_derivs.
kernel void test_gpu_derivs_kernel(
    device const float* quad_pts [[buffer(0)]],
    device const float* loc      [[buffer(1)]],  // (s, theta, zeta) per particle, 3 floats each
    device const float* vpar_buf [[buffer(2)]],
    device const float* time_buf [[buffer(3)]],
    device float*       out      [[buffer(4)]],  // 4 floats per particle: dx1/dt, dx2/dt, dzeta/dt, dvpar/dt
    constant DerivativeConstants& c [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    int p = int(gid);
    if (p >= c.n_points) return;

    float s_val    = loc[3 * p + 0];
    float theta    = loc[3 * p + 1];
    float zeta     = loc[3 * p + 2];
    float vpar_val = vpar_buf[p];

    // State encoding: (x1, x2, zeta, vpar) where x1 = s*cos(theta), x2 = s*sin(theta).
    float state[4] = {
        s_val * cos(theta),
        s_val * sin(theta),
        zeta,
        vpar_val
    };

    float x_temp[5];
    float derivs[42];   // 7 stages * 6 fields
    float x1_shape[4], x2_shape[4], x3_shape[4];
    bool  symmetry_exploited[1];
    int   index_i[1], index_j[1], index_k[1];
    float mu[1];
    float t[1], dt[1];

    t[0]  = 0.0f;
    dt[0] = 0.0f;
    f3d_build_state<F3D_RHS_BOOZER_VACUUM>(
        x_temp, 0, symmetry_exploited,
        index_i, index_j, index_k,
        x1_shape, x2_shape, x3_shape,
        state, derivs, t, dt, c);

    mu[0] = -1.0f;
    f3d_calc_derivs<F3D_RHS_BOOZER_VACUUM>(
        quad_pts, derivs, 0,
        x_temp, symmetry_exploited,
        index_i, index_j, index_k,
        x1_shape, x2_shape, x3_shape,
        mu, 0, c);

    float modB    = derivs[4];
    float v_perp2 = c.v_total * c.v_total - vpar_val * vpar_val;
    mu[0] = v_perp2 / (2.0f * modB);

    t[0]  = time_buf[p];
    dt[0] = 0.0f;
    f3d_build_state<F3D_RHS_BOOZER_VACUUM>(
        x_temp, 0, symmetry_exploited,
        index_i, index_j, index_k,
        x1_shape, x2_shape, x3_shape,
        state, derivs, t, dt, c);

    f3d_calc_derivs<F3D_RHS_BOOZER_VACUUM>(
        quad_pts, derivs, 0,
        x_temp, symmetry_exploited,
        index_i, index_j, index_k,
        x1_shape, x2_shape, x3_shape,
        mu, 0, c);

    out[4 * p + 0] = derivs[0];
    out[4 * p + 1] = derivs[1];
    out[4 * p + 2] = derivs[2];
    out[4 * p + 3] = derivs[3];
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
