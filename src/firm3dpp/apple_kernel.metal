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

    float x1_start;
    float x2_start;
    float x2_period;
    float x3_start;
    float x3_period;

    float x1_step;
    float x2_step;
    float x3_step;
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

kernel void apple_interpolate_kernel(
    device const float* data [[buffer(0)]],
    device const int* index_i [[buffer(1)]],
    device const int* index_j [[buffer(2)]],
    device const int* index_k [[buffer(3)]],
    device const float* x1_shape [[buffer(4)]],
    device const float* x2_shape [[buffer(5)]],
    device const float* x3_shape [[buffer(6)]],
    device float* out [[buffer(7)]],
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

    float local_val = 0.0f;
    for (int ii = 0; ii < 4; ++ii) {
        for (int jj = 0; jj < 4; ++jj) {
            for (int kk = 0; kk < 4; ++kk) {
                int row_idx = 64 * (i * c.n_x23 + j * c.n_x3 + k) + 16 * ii + 4 * jj + kk;
                float shape_val =
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

    for (int zz = 0; zz < c.n_fields; ++zz) {
        float local_val = 0.0f;
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
