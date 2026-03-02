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

#define F3D_DEFINE_SHAPE_AND_MOD(SUFFIX, REAL_T)                                              \
inline REAL_T f3d_shape_fn_##SUFFIX(REAL_T x, int i) {                                        \
    switch (i) {                                                                               \
        case 0:                                                                                \
            return (REAL_T(1.0) - x) * (REAL_T(2.0) - x) * (REAL_T(3.0) - x) / REAL_T(6.0);  \
        case 1:                                                                                \
            return x * (REAL_T(2.0) - x) * (REAL_T(3.0) - x) / REAL_T(2.0);                   \
        case 2:                                                                                \
            return x * (x - REAL_T(1.0)) * (REAL_T(3.0) - x) / REAL_T(2.0);                   \
        case 3:                                                                                \
            return x * (x - REAL_T(1.0)) * (x - REAL_T(2.0)) / REAL_T(6.0);                   \
        default:                                                                               \
            return REAL_T(0.0);                                                                \
    }                                                                                          \
}                                                                                              \
                                                                                               \
inline REAL_T f3d_positive_mod_##SUFFIX(REAL_T x, REAL_T period) {                            \
    REAL_T out = fmod(x, period);                                                              \
    if (out < REAL_T(0.0)) {                                                                   \
        out += period;                                                                         \
    }                                                                                          \
    return out;                                                                                \
}

#define F3D_DEFINE_APPLE_INTERPOLATE_KERNEL(KERNEL_NAME, SUFFIX, REAL_T)                      \
kernel void KERNEL_NAME(                                                                       \
    device const REAL_T* data [[buffer(0)]],                                                   \
    device const int* index_i [[buffer(1)]],                                                   \
    device const int* index_j [[buffer(2)]],                                                   \
    device const int* index_k [[buffer(3)]],                                                   \
    device const REAL_T* x1_shape [[buffer(4)]],                                               \
    device const REAL_T* x2_shape [[buffer(5)]],                                               \
    device const REAL_T* x3_shape [[buffer(6)]],                                               \
    device REAL_T* out [[buffer(7)]],                                                          \
    constant TracingConstants& c [[buffer(8)]],                                                \
    uint gid [[thread_position_in_grid]]                                                       \
) {                                                                                            \
    int total = c.n_points * c.n_fields;                                                       \
    if (int(gid) >= total) {                                                                   \
        return;                                                                                \
    }                                                                                          \
                                                                                               \
    int particle_id = int(gid) / c.n_fields;                                                   \
    int zz = int(gid) % c.n_fields;                                                            \
    int i = index_i[particle_id];                                                              \
    int j = index_j[particle_id];                                                              \
    int k = index_k[particle_id];                                                              \
                                                                                               \
    REAL_T local_val = REAL_T(0.0);                                                            \
    for (int ii = 0; ii < 4; ++ii) {                                                           \
        for (int jj = 0; jj < 4; ++jj) {                                                       \
            for (int kk = 0; kk < 4; ++kk) {                                                   \
                int row_idx = 64 * (i * c.n_x23 + j * c.n_x3 + k) + 16 * ii + 4 * jj + kk;   \
                REAL_T shape_val =                                                             \
                    x1_shape[ii * c.n_points + particle_id] *                                  \
                    x2_shape[jj * c.n_points + particle_id] *                                  \
                    x3_shape[kk * c.n_points + particle_id];                                   \
                local_val += data[c.n_fields * row_idx + zz] * shape_val;                     \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
    out[particle_id * c.n_fields + zz] = local_val;                                            \
}

#define F3D_DEFINE_TEST_INTERPOLATION_KERNEL(KERNEL_NAME, SUFFIX, REAL_T)                     \
kernel void KERNEL_NAME(                                                                       \
    device const REAL_T* quad_pts [[buffer(0)]],                                               \
    device const REAL_T* loc [[buffer(1)]],                                                    \
    device REAL_T* out [[buffer(2)]],                                                          \
    constant InterpolationTestConstants& c [[buffer(3)]],                                      \
    uint gid [[thread_position_in_grid]]                                                       \
) {                                                                                            \
    if (int(gid) >= c.n_points) {                                                              \
        return;                                                                                \
    }                                                                                          \
                                                                                               \
    int p = int(gid);                                                                          \
    REAL_T x = loc[3 * p + 0];                                                                 \
    REAL_T y = loc[3 * p + 1];                                                                 \
    REAL_T z = loc[3 * p + 2];                                                                 \
                                                                                               \
    REAL_T interp_x1 = REAL_T(0.0);                                                            \
    REAL_T interp_x2 = REAL_T(0.0);                                                            \
    REAL_T interp_x3 = REAL_T(0.0);                                                            \
    bool symmetry_exploited = false;                                                           \
    const REAL_T two_pi = REAL_T(6.28318530717958647692);                                     \
                                                                                               \
    if (c.rhs_mode == F3D_RHS_CARTESIAN_VACUUM) {                                              \
        REAL_T r = sqrt(x * x + y * y);                                                        \
        REAL_T phi = atan2(y, x);                                                              \
        phi = f3d_positive_mod_##SUFFIX(phi, REAL_T(c.x2_period));                            \
                                                                                               \
        symmetry_exploited = z < REAL_T(0.0);                                                  \
        if (symmetry_exploited) {                                                              \
            z = -z;                                                                            \
            phi = f3d_positive_mod_##SUFFIX(two_pi - phi, REAL_T(c.x2_period));               \
        }                                                                                      \
        interp_x1 = r;                                                                         \
        interp_x2 = phi;                                                                       \
        interp_x3 = z;                                                                         \
    } else {                                                                                   \
        REAL_T s = sqrt(x * x + y * y);                                                        \
        REAL_T theta = atan2(y, x);                                                            \
        REAL_T t = f3d_positive_mod_##SUFFIX(theta, two_pi);                                  \
        REAL_T zeta = f3d_positive_mod_##SUFFIX(z, REAL_T(c.x3_period));                      \
                                                                                               \
        symmetry_exploited = t > REAL_T(3.14159265358979323846);                              \
        if (symmetry_exploited) {                                                              \
            zeta = REAL_T(c.x3_period) - zeta;                                                 \
            t = two_pi - t;                                                                    \
        }                                                                                      \
        interp_x1 = s;                                                                         \
        interp_x2 = t;                                                                         \
        interp_x3 = zeta;                                                                      \
    }                                                                                          \
                                                                                               \
    int i = 3 * (int((interp_x1 - REAL_T(c.x1_start)) / REAL_T(c.x1_step)) / 3);             \
    int j = 3 * (int((interp_x2 - REAL_T(c.x2_start)) / REAL_T(c.x2_step)) / 3);             \
    int k = 3 * (int((interp_x3 - REAL_T(c.x3_start)) / REAL_T(c.x3_step)) / 3);             \
                                                                                               \
    i = min(i, c.x1_count - 4);                                                                \
    j = min(j, c.x2_count - 4);                                                                \
    k = min(k, c.x3_count - 4);                                                                \
    i = max(i, 0);                                                                             \
    j = max(j, 0);                                                                             \
    k = max(k, 0);                                                                             \
                                                                                               \
    int cell_i = i / 3;                                                                        \
    int cell_j = j / 3;                                                                        \
    int cell_k = k / 3;                                                                        \
                                                                                               \
    REAL_T x1_rel = (interp_x1 - REAL_T(i) * REAL_T(c.x1_step) - REAL_T(c.x1_start)) / REAL_T(c.x1_step); \
    REAL_T x2_rel = (interp_x2 - REAL_T(j) * REAL_T(c.x2_step) - REAL_T(c.x2_start)) / REAL_T(c.x2_step); \
    REAL_T x3_rel = (interp_x3 - REAL_T(k) * REAL_T(c.x3_step) - REAL_T(c.x3_start)) / REAL_T(c.x3_step); \
                                                                                               \
    REAL_T s1[4];                                                                              \
    REAL_T s2[4];                                                                              \
    REAL_T s3[4];                                                                              \
    for (int ii = 0; ii < 4; ++ii) {                                                           \
        s1[ii] = f3d_shape_fn_##SUFFIX(x1_rel, ii);                                            \
        s2[ii] = f3d_shape_fn_##SUFFIX(x2_rel, ii);                                            \
        s3[ii] = f3d_shape_fn_##SUFFIX(x3_rel, ii);                                            \
    }                                                                                          \
                                                                                               \
    for (int zz = 0; zz < c.n_fields; ++zz) {                                                  \
        REAL_T local_val = REAL_T(0.0);                                                        \
        for (int ii = 0; ii < 4; ++ii) {                                                       \
            for (int jj = 0; jj < 4; ++jj) {                                                   \
                for (int kk = 0; kk < 4; ++kk) {                                               \
                    int row_idx = 64 * (cell_i * c.n_x23 + cell_j * c.n_x3 + cell_k) + 16 * ii + 4 * jj + kk; \
                    local_val += quad_pts[c.n_fields * row_idx + zz] * s1[ii] * s2[jj] * s3[kk]; \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
        out[p * c.n_fields + zz] = local_val;                                                  \
    }                                                                                          \
                                                                                               \
    if (symmetry_exploited) {                                                                  \
        if (c.rhs_mode == F3D_RHS_CARTESIAN_VACUUM) {                                          \
            out[p * c.n_fields + 0] *= REAL_T(-1.0);                                           \
            out[p * c.n_fields + 4] *= REAL_T(-1.0);                                           \
            out[p * c.n_fields + 5] *= REAL_T(-1.0);                                           \
        } else if (c.rhs_mode == F3D_RHS_BOOZER_VACUUM || c.rhs_mode == F3D_RHS_BOOZER_SAW_VACUUM) { \
            out[p * c.n_fields + 2] *= REAL_T(-1.0);                                           \
            out[p * c.n_fields + 3] *= REAL_T(-1.0);                                           \
        } else if (c.rhs_mode == F3D_RHS_BOOZER) {                                             \
            out[p * c.n_fields + 2] *= REAL_T(-1.0);                                           \
            out[p * c.n_fields + 3] *= REAL_T(-1.0);                                           \
            if (c.n_fields >= 12) {                                                            \
                out[p * c.n_fields + 9] *= REAL_T(-1.0);                                       \
            }                                                                                  \
        }                                                                                      \
    }                                                                                          \
}

F3D_DEFINE_SHAPE_AND_MOD(f32, float)
F3D_DEFINE_APPLE_INTERPOLATE_KERNEL(apple_interpolate_kernel, f32, float)
F3D_DEFINE_TEST_INTERPOLATION_KERNEL(test_gpu_interpolation_kernel, f32, float)

F3D_DEFINE_APPLE_INTERPOLATE_KERNEL(apple_interpolate_kernel_f32, f32, float)
F3D_DEFINE_TEST_INTERPOLATION_KERNEL(test_gpu_interpolation_kernel_f32, f32, float)
