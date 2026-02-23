#include <metal_stdlib>
using namespace metal;

enum {
    F3D_RHS_CARTESIAN_VACUUM = 0,
    F3D_RHS_BOOZER_VACUUM = 1,
    F3D_RHS_BOOZER_SAW_VACUUM = 2,
    F3D_RHS_BOOZER = 3
};

struct TracingConstants {
    int n_x2;
    int n_x3;
    int n_x23;
    int n_fields;
    int n_points;
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

    double x1_start;
    double x2_start;
    double x2_period;
    double x3_start;
    double x3_period;

    double x1_step;
    double x2_step;
    double x3_step;
};

inline double f3d_shape_fn(double x, int i) {
    switch (i) {
        case 0:
            return (1.0 - x) * (2.0 - x) * (3.0 - x) / 6.0;
        case 1:
            return x * (2.0 - x) * (3.0 - x) / 2.0;
        case 2:
            return x * (x - 1.0) * (3.0 - x) / 2.0;
        case 3:
            return x * (x - 1.0) * (x - 2.0) / 6.0;
        default:
            return 0.0;
    }
}

inline double f3d_positive_mod(double x, double period) {
    double out = fmod(x, period);
    if (out < 0.0) {
        out += period;
    }
    return out;
}

kernel void apple_interpolate_kernel(
    device const double* data [[buffer(0)]],
    device const int* index_i [[buffer(1)]],
    device const int* index_j [[buffer(2)]],
    device const int* index_k [[buffer(3)]],
    device const double* x1_shape [[buffer(4)]],
    device const double* x2_shape [[buffer(5)]],
    device const double* x3_shape [[buffer(6)]],
    device double* out [[buffer(7)]],
    constant TracingConstants& c [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    int total = c.n_points * c.n_fields;
    if (int(gid) >= total) {
        return;
    }

    int particle_id = int(gid) / c.n_fields;
    int zz = int(gid) % c.n_fields;
    int i = index_i[particle_id];
    int j = index_j[particle_id];
    int k = index_k[particle_id];

    double local_val = 0.0;
    for (int ii = 0; ii < 4; ++ii) {
        for (int jj = 0; jj < 4; ++jj) {
            for (int kk = 0; kk < 4; ++kk) {
                int row_idx = 64 * (i * c.n_x23 + j * c.n_x3 + k) + 16 * ii + 4 * jj + kk;
                double shape_val =
                    x1_shape[ii * c.n_points + particle_id] *
                    x2_shape[jj * c.n_points + particle_id] *
                    x3_shape[kk * c.n_points + particle_id];
                local_val += data[c.n_fields * row_idx + zz] * shape_val;
            }
        }
    }
    out[particle_id * c.n_fields + zz] = local_val;
}

kernel void test_gpu_interpolation_kernel(
    device const double* quad_pts [[buffer(0)]],
    device const double* loc [[buffer(1)]],
    device double* out [[buffer(2)]],
    constant InterpolationTestConstants& c [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (int(gid) >= c.n_points) {
        return;
    }

    int p = int(gid);
    double x = loc[3 * p + 0];
    double y = loc[3 * p + 1];
    double z = loc[3 * p + 2];

    double interp_x1 = 0.0;
    double interp_x2 = 0.0;
    double interp_x3 = 0.0;
    bool symmetry_exploited = false;

    if (c.rhs_mode == F3D_RHS_CARTESIAN_VACUUM) {
        // Cartesian mode: loc = (x, y, z), map to (r, phi, z) with symmetry.
        double r = sqrt(x * x + y * y);
        double phi = atan2(y, x);
        phi = f3d_positive_mod(phi, c.x2_period);

        symmetry_exploited = z < 0.0;
        if (symmetry_exploited) {
            z = -z;
            phi = f3d_positive_mod(2.0 * M_PI - phi, c.x2_period);
        }
        interp_x1 = r;
        interp_x2 = phi;
        interp_x3 = z;
    } else {
        // Boozer-like modes: loc = (x1, x2, zeta), recover (s, theta, zeta).
        double s = sqrt(x * x + y * y);
        double theta = atan2(y, x);
        double t = f3d_positive_mod(theta, 2.0 * M_PI);
        double zeta = f3d_positive_mod(z, c.x3_period);

        symmetry_exploited = t > M_PI;
        if (symmetry_exploited) {
            zeta = c.x3_period - zeta;
            t = 2.0 * M_PI - t;
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

    double x1_rel = (interp_x1 - double(i) * c.x1_step - c.x1_start) / c.x1_step;
    double x2_rel = (interp_x2 - double(j) * c.x2_step - c.x2_start) / c.x2_step;
    double x3_rel = (interp_x3 - double(k) * c.x3_step - c.x3_start) / c.x3_step;

    double s1[4];
    double s2[4];
    double s3[4];
    for (int ii = 0; ii < 4; ++ii) {
        s1[ii] = f3d_shape_fn(x1_rel, ii);
        s2[ii] = f3d_shape_fn(x2_rel, ii);
        s3[ii] = f3d_shape_fn(x3_rel, ii);
    }

    for (int zz = 0; zz < c.n_fields; ++zz) {
        double local_val = 0.0;
        for (int ii = 0; ii < 4; ++ii) {
            for (int jj = 0; jj < 4; ++jj) {
                for (int kk = 0; kk < 4; ++kk) {
                    int row_idx = 64 * (cell_i * c.n_x23 + cell_j * c.n_x3 + cell_k) + 16 * ii + 4 * jj + kk;
                    local_val += quad_pts[c.n_fields * row_idx + zz] * s1[ii] * s2[jj] * s3[kk];
                }
            }
        }
        out[p * c.n_fields + zz] = local_val;
    }

    // Match CUDA account_for_symmetry_rhs behavior.
    if (symmetry_exploited) {
        if (c.rhs_mode == F3D_RHS_CARTESIAN_VACUUM) {
            out[p * c.n_fields + 0] *= -1.0;
            out[p * c.n_fields + 4] *= -1.0;
            out[p * c.n_fields + 5] *= -1.0;
        } else if (c.rhs_mode == F3D_RHS_BOOZER_VACUUM || c.rhs_mode == F3D_RHS_BOOZER_SAW_VACUUM) {
            out[p * c.n_fields + 2] *= -1.0;
            out[p * c.n_fields + 3] *= -1.0;
        } else if (c.rhs_mode == F3D_RHS_BOOZER) {
            out[p * c.n_fields + 2] *= -1.0;
            out[p * c.n_fields + 3] *= -1.0;
            if (c.n_fields >= 12) {
                out[p * c.n_fields + 9] *= -1.0;
            }
        }
    }
}
