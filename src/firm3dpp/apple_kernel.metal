#include <metal_stdlib>
using namespace metal;

constant uint THREADS_PER_BLOCK = 64;
constant uint PARTICLES_PER_BLOCK = 8;

struct TracingConstants {
    int n_x2;
    int n_x3;
    int n_x23;
};

template <uint n>
inline void interpolate(
    threadgroup double* out,
    device const double* data,
    threadgroup const int* index_i,
    threadgroup const int* index_j,
    threadgroup const int* index_k,
    threadgroup const double* x1_shape,
    threadgroup const double* x2_shape,
    threadgroup const double* x3_shape,
    int nparticles_blk,
    constant TracingConstants& c,
    uint tid
) {
    for (int idx = int(tid); idx < nparticles_blk * int(n); idx += int(THREADS_PER_BLOCK)) {
        int zz = idx % int(n);
        int particle_id = idx / int(n);
        int i = index_i[particle_id];
        int j = index_j[particle_id];
        int k = index_k[particle_id];

        double local_val = 0.0;
        for (int ii = 0; ii < 4; ++ii) {
            for (int jj = 0; jj < 4; ++jj) {
                for (int kk = 0; kk < 4; ++kk) {
                    int row_idx = 64 * (i * c.n_x23 + j * c.n_x3 + k) + 16 * ii + 4 * jj + kk;
                    double shape_val =
                        x1_shape[ii * int(PARTICLES_PER_BLOCK) + particle_id] *
                        x2_shape[jj * int(PARTICLES_PER_BLOCK) + particle_id] *
                        x3_shape[kk * int(PARTICLES_PER_BLOCK) + particle_id];
                    local_val += data[int(n) * row_idx + zz] * shape_val;
                }
            }
        }
        out[int(PARTICLES_PER_BLOCK) * zz + particle_id] = local_val;
    }
}

kernel void apple_interpolate_stub(
    constant TracingConstants& c [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    (void)c;
    (void)tid;
}
