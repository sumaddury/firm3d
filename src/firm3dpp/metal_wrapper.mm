#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

enum {
    F3D_RHS_CARTESIAN_VACUUM = 0,
    F3D_RHS_BOOZER_VACUUM    = 1,
    F3D_RHS_BOOZER_SAW_VACUUM = 2,
    F3D_RHS_BOOZER           = 3
};

struct InterpolationTestConstants {
    int n_x2, n_x3, n_x23, n_fields, n_points, rhs_mode;
    int x1_count, x2_count, x3_count;
    float x1_start, x2_start, x2_period, x3_start, x3_period;
    float x1_step, x2_step, x3_step;
};

struct DerivativeConstants {
    int x1_count, x2_count, x3_count;
    int n_x2, n_x3, n_x23, n_points;
    float x1_start, x2_start, x2_period, x3_start, x3_period;
    float x1_step, x2_step, x3_step;
    float mass, charge, psi0, v_total;
    float tol;   // adaptive step tolerance (used by timestep kernel, 0 otherwise)
    float tmax;  // maximum timestep cap   (used by timestep kernel, 0 otherwise)
};

// ---------------------------------------------------------------------------
// Metal context: device, command queue, and one pipeline per kernel function.
// ---------------------------------------------------------------------------

struct MetalContext {
    id<MTLDevice>              device                         = nil;
    id<MTLCommandQueue>        queue                          = nil;
    id<MTLComputePipelineState> pipeline_interp               = nil;
    id<MTLComputePipelineState> pipeline_derivs_boozer_vac    = nil;
    id<MTLComputePipelineState> pipeline_derivs_cartesian     = nil;
    id<MTLComputePipelineState> pipeline_timestep_boozer_vac  = nil;
    id<MTLComputePipelineState> pipeline_tracing_boozer_vac   = nil;
};

static std::string load_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) throw std::runtime_error("Failed to open file: " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static std::string find_kernel_source_path() {
    if (const char* env = std::getenv("FIRM3DPP_METAL_KERNEL_PATH"))
        return std::string(env);
    for (const auto& p : {"apple_kernel.metal", "src/firm3dpp/apple_kernel.metal"}) {
        if (std::ifstream(p).good()) return p;
    }
    throw std::runtime_error(
        "Could not find apple_kernel.metal. Set FIRM3DPP_METAL_KERNEL_PATH.");
}

static id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device, id<MTLLibrary> lib, NSString* name)
{
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (fn == nil)
        throw std::runtime_error("Kernel not found in Metal source: " +
                                 std::string([name UTF8String]));
    NSError* err = nil;
    id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:fn error:&err];
    if (ps == nil) {
        std::string msg = err ? std::string([[err localizedDescription] UTF8String]) : "unknown";
        throw std::runtime_error("Failed to create Metal pipeline for " +
                                 std::string([name UTF8String]) + ": " + msg);
    }
    return ps;
}

static MetalContext& metal_context() {
    static MetalContext ctx = []() {
        MetalContext local;
        @autoreleasepool {
            local.device = MTLCreateSystemDefaultDevice();
            if (local.device == nil) throw std::runtime_error("Metal device not available");

            local.queue = [local.device newCommandQueue];
            if (local.queue == nil) throw std::runtime_error("Failed to create Metal command queue");

            const std::string src = load_file(find_kernel_source_path());
            NSString* ns_src = [NSString stringWithUTF8String:src.c_str()];
            NSError* lib_err = nil;
            id<MTLLibrary> lib = [local.device newLibraryWithSource:ns_src options:nil error:&lib_err];
            if (lib == nil) {
                std::string msg = lib_err
                    ? std::string([[lib_err localizedDescription] UTF8String]) : "unknown";
                throw std::runtime_error("Failed to compile Metal source: " + msg);
            }

            local.pipeline_interp              = make_pipeline(local.device, lib, @"test_gpu_interpolation_kernel");
            local.pipeline_derivs_boozer_vac   = make_pipeline(local.device, lib, @"test_gpu_derivs_boozer_vacuum_kernel");
            local.pipeline_derivs_cartesian    = make_pipeline(local.device, lib, @"test_gpu_derivs_cartesian_kernel");
            local.pipeline_timestep_boozer_vac = make_pipeline(local.device, lib, @"test_gpu_timestep_boozer_vacuum_kernel");
            local.pipeline_tracing_boozer_vac  = make_pipeline(local.device, lib, @"boozer_vacuum_tracing_kernel");
        }
        return local;
    }();
    return ctx;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Build a DerivativeConstants struct from the three range arrays passed from Python.
// Range arrays have layout [start, end_or_period, n_points].
static DerivativeConstants make_deriv_constants(
    const double* x1, const double* x2, const double* x3,
    double mass, double charge, double psi0, double v_total,
    int n_points)
{
    const int x1c = static_cast<int>(x1[2]);
    const int x2c = static_cast<int>(x2[2]);
    const int x3c = static_cast<int>(x3[2]);
    const double x1s = (x1[1] - x1[0]) / (x1[2] - 1.0);
    const double x2s = (x2[1] - x2[0]) / (x2[2] - 1.0);
    const double x3s = (x3[1] - x3[0]) / (x3[2] - 1.0);
    const int n_x2  = (x2c - 1) / 3;
    const int n_x3  = (x3c - 1) / 3;

    DerivativeConstants dc{};
    dc.x1_count  = x1c;    dc.x2_count  = x2c;    dc.x3_count  = x3c;
    dc.n_x2      = n_x2;   dc.n_x3      = n_x3;   dc.n_x23     = n_x2 * n_x3;
    dc.n_points  = n_points;
    dc.x1_start  = static_cast<float>(x1[0]);
    dc.x2_start  = static_cast<float>(x2[0]);   dc.x2_period = static_cast<float>(x2[1]);
    dc.x3_start  = static_cast<float>(x3[0]);   dc.x3_period = static_cast<float>(x3[1]);
    dc.x1_step   = static_cast<float>(x1s);
    dc.x2_step   = static_cast<float>(x2s);
    dc.x3_step   = static_cast<float>(x3s);
    dc.mass      = static_cast<float>(mass);
    dc.charge    = static_cast<float>(charge);
    dc.psi0      = static_cast<float>(psi0);
    dc.v_total   = static_cast<float>(v_total);
    dc.tol       = 0.0f;
    dc.tmax      = 0.0f;
    return dc;
}

// Build a DerivativeConstants for one-timestep tests. tol drives the DP5 error
// controller; tmax caps the maximum step size (mirrors CUDA's hardcoded 1e-2).
static DerivativeConstants make_timestep_constants(
    const double* x1, const double* x2, const double* x3,
    double mass, double charge, double psi0, double v_total,
    double tol, int n_points)
{
    DerivativeConstants dc = make_deriv_constants(x1, x2, x3, mass, charge, psi0, v_total, n_points);
    // Floor the tolerance to something achievable in f32 so the step controller
    // converges. 1e-5 is well above f32 machine epsilon (~1.2e-7) but tight
    // enough to produce accurate single-step results.
    dc.tol  = static_cast<float>(std::max(tol, 1e-5));
    dc.tmax = 1e-2f;
    return dc;
}

// Dispatch a derivatives kernel and return results as a flat f64 array of length 4*n_points.
// All inputs must already be in f32.
static py::array_t<double> metal_dispatch_derivs(
    id<MTLComputePipelineState> pipeline,
    const std::vector<float>& quad_f,
    const std::vector<float>& loc_f,
    const std::vector<float>& vpar_f,
    const std::vector<float>& time_f,
    const DerivativeConstants& dc)
{
    auto& ctx = metal_context();
    const size_t out_count = 4 * static_cast<size_t>(dc.n_points);

    @autoreleasepool {
        auto make_buf = [&](const void* data, size_t bytes) {
            return [ctx.device newBufferWithBytes:data length:bytes
                                         options:MTLResourceStorageModeShared];
        };

        id<MTLBuffer> quad_m = make_buf(quad_f.data(), quad_f.size() * sizeof(float));
        id<MTLBuffer> loc_m  = make_buf(loc_f.data(),  loc_f.size()  * sizeof(float));
        id<MTLBuffer> vpar_m = make_buf(vpar_f.data(), vpar_f.size() * sizeof(float));
        id<MTLBuffer> time_m = make_buf(time_f.data(), time_f.size() * sizeof(float));
        id<MTLBuffer> out_m  = [ctx.device newBufferWithLength:out_count * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> dc_m   = make_buf(&dc, sizeof(dc));

        id<MTLCommandBuffer>         cb  = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:quad_m offset:0 atIndex:0];
        [enc setBuffer:loc_m  offset:0 atIndex:1];
        [enc setBuffer:vpar_m offset:0 atIndex:2];
        [enc setBuffer:time_m offset:0 atIndex:3];
        [enc setBuffer:out_m  offset:0 atIndex:4];
        [enc setBuffer:dc_m   offset:0 atIndex:5];

        const NSUInteger total = static_cast<NSUInteger>(dc.n_points);
        const NSUInteger tpg   = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            std::string err = cb.error
                ? std::string([[cb.error localizedDescription] UTF8String]) : "unknown";
            throw std::runtime_error("Metal derivatives dispatch failed: " + err);
        }

        py::array_t<double> result(out_count);
        const float*  src = static_cast<const float*>([out_m contents]);
        double*       dst = static_cast<double*>(result.request().ptr);
        for (size_t i = 0; i < out_count; ++i) dst[i] = static_cast<double>(src[i]);
        return result;
    }
}

// Convert a Python f64 array to a std::vector<float>.
static std::vector<float> to_f32(const double* ptr, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(ptr[i]);
    return out;
}

// ---------------------------------------------------------------------------
// RHS-specific helpers
// ---------------------------------------------------------------------------

struct RhsSpec { int mode; int n_fields; };

static bool has_suffix(const std::string& s, const std::string& sfx) {
    return s.size() >= sfx.size() &&
           s.compare(s.size() - sfx.size(), sfx.size(), sfx) == 0;
}

static RhsSpec parse_rhs(std::string rhs) {
    if (has_suffix(rhs, "_f32")) rhs.erase(rhs.size() - 4);
    else if (has_suffix(rhs, "_f64"))
        throw std::invalid_argument("Metal f64 mode not supported. Use *_f32.");
    if (rhs == "cartesian_vacuum")  return {F3D_RHS_CARTESIAN_VACUUM,  7};
    if (rhs == "boozer_vacuum")     return {F3D_RHS_BOOZER_VACUUM,     6};
    if (rhs == "boozer_saw_vacuum") return {F3D_RHS_BOOZER_SAW_VACUUM, 10};
    if (rhs == "boozer")            return {F3D_RHS_BOOZER,            12};
    throw std::invalid_argument("Unsupported rhs: " + rhs);
}

// For the interpolation test, Python passes loc in (coord1, angle, coord3) form.
// The GPU kernel expects (x, y, coord3) = (coord1*cos(angle), coord1*sin(angle), coord3).
static void preprocess_loc(std::vector<double>& loc, int n_points) {
    for (int i = 0; i < n_points; ++i) {
        const double c1  = loc[3 * i + 0];
        const double ang = loc[3 * i + 1];
        loc[3 * i + 0] = c1 * std::cos(ang);
        loc[3 * i + 1] = c1 * std::sin(ang);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

extern "C" py::array_t<double> test_gpu_interpolation(
    py::array_t<double> quad_pts,
    py::array_t<double> x1_range,
    py::array_t<double> x2_range,
    py::array_t<double> x3_range,
    py::array_t<double> loc,
    std::string rhs,
    int n_points)
{
    const RhsSpec spec  = parse_rhs(rhs);
    const double* x1    = static_cast<double*>(x1_range.request().ptr);
    const double* x2    = static_cast<double*>(x2_range.request().ptr);
    const double* x3    = static_cast<double*>(x3_range.request().ptr);
    const double* qptr  = static_cast<double*>(quad_pts.request().ptr);
    const double* lptr  = static_cast<double*>(loc.request().ptr);

    const int x1c = static_cast<int>(x1[2]);
    const int x2c = static_cast<int>(x2[2]);
    const int x3c = static_cast<int>(x3[2]);
    const double x1s = (x1[1] - x1[0]) / (x1[2] - 1.0);
    const double x2s = (x2[1] - x2[0]) / (x2[2] - 1.0);
    const double x3s = (x3[1] - x3[0]) / (x3[2] - 1.0);
    const int n_x2  = (x2c - 1) / 3;
    const int n_x3  = (x3c - 1) / 3;

    InterpolationTestConstants c{};
    c.n_x2     = n_x2;  c.n_x3   = n_x3;  c.n_x23    = n_x2 * n_x3;
    c.n_fields = spec.n_fields;  c.n_points = n_points;  c.rhs_mode = spec.mode;
    c.x1_count = x1c;   c.x2_count = x2c;  c.x3_count  = x3c;
    c.x1_start = static_cast<float>(x1[0]);
    c.x2_start = static_cast<float>(x2[0]);  c.x2_period = static_cast<float>(x2[1]);
    c.x3_start = static_cast<float>(x3[0]);  c.x3_period = static_cast<float>(x3[1]);
    c.x1_step  = static_cast<float>(x1s);
    c.x2_step  = static_cast<float>(x2s);
    c.x3_step  = static_cast<float>(x3s);

    // Convert loc to (x, y, z) in-place on a local copy.
    std::vector<double> loc_host(lptr, lptr + loc.size());
    preprocess_loc(loc_host, n_points);

    const std::vector<float> quad_f = to_f32(qptr, quad_pts.size());
    const std::vector<float> loc_f  = to_f32(loc_host.data(), loc_host.size());

    auto& ctx = metal_context();
    const size_t out_count = static_cast<size_t>(spec.n_fields) * static_cast<size_t>(n_points);

    @autoreleasepool {
        auto make_buf = [&](const void* data, size_t bytes) {
            return [ctx.device newBufferWithBytes:data length:bytes
                                         options:MTLResourceStorageModeShared];
        };

        id<MTLBuffer> quad_m = make_buf(quad_f.data(), quad_f.size() * sizeof(float));
        id<MTLBuffer> loc_m  = make_buf(loc_f.data(),  loc_f.size()  * sizeof(float));
        id<MTLBuffer> out_m  = [ctx.device newBufferWithLength:out_count * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> c_m    = make_buf(&c, sizeof(c));

        id<MTLCommandBuffer>         cb  = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx.pipeline_interp];
        [enc setBuffer:quad_m offset:0 atIndex:0];
        [enc setBuffer:loc_m  offset:0 atIndex:1];
        [enc setBuffer:out_m  offset:0 atIndex:2];
        [enc setBuffer:c_m    offset:0 atIndex:3];

        const NSUInteger total = static_cast<NSUInteger>(n_points);
        const NSUInteger tpg   = std::min<NSUInteger>(ctx.pipeline_interp.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            std::string err = cb.error
                ? std::string([[cb.error localizedDescription] UTF8String]) : "unknown";
            throw std::runtime_error("Metal interpolation dispatch failed: " + err);
        }

        py::array_t<double> result(out_count);
        const float*  src = static_cast<const float*>([out_m contents]);
        double*       dst = static_cast<double*>(result.request().ptr);
        for (size_t i = 0; i < out_count; ++i) dst[i] = static_cast<double>(src[i]);
        return result;
    }
}

extern "C" py::array_t<double> test_gpu_derivatives_boozer_vacuum(
    py::array_t<double> quad_pts,
    py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range,
    py::array_t<double> loc, py::array_t<double> vpar, py::array_t<double> time,
    double v_total, double m, double q, double psi0, int n_points)
{
    const double* x1   = static_cast<double*>(x1_range.request().ptr);
    const double* x2   = static_cast<double*>(x2_range.request().ptr);
    const double* x3   = static_cast<double*>(x3_range.request().ptr);
    const double* qptr = static_cast<double*>(quad_pts.request().ptr);
    const double* lptr = static_cast<double*>(loc.request().ptr);
    const double* vptr = static_cast<double*>(vpar.request().ptr);
    const double* tptr = static_cast<double*>(time.request().ptr);

    const DerivativeConstants dc = make_deriv_constants(x1, x2, x3, m, q, psi0, v_total, n_points);
    return metal_dispatch_derivs(
        metal_context().pipeline_derivs_boozer_vac,
        to_f32(qptr, quad_pts.size()),
        to_f32(lptr, loc.size()),
        to_f32(vptr, n_points),
        to_f32(tptr, n_points),
        dc);
}

// Run one adaptive DP5 step per particle (Boozer-vacuum mode).
// loc: (n_points, 3) = [s, theta, zeta].
// vpar: (n_points,).
// Returns flat array of 5*n_points doubles: [t, s, theta, zeta, v_par] per particle.
extern "C" py::array_t<double> test_gpu_timestep_boozer_vacuum(
    py::array_t<double> quad_pts,
    py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range,
    py::array_t<double> loc, py::array_t<double> vpar,
    double v_total, double m, double q, double psi0,
    double tol, int n_points)
{
    const double* x1   = static_cast<double*>(x1_range.request().ptr);
    const double* x2   = static_cast<double*>(x2_range.request().ptr);
    const double* x3   = static_cast<double*>(x3_range.request().ptr);
    const double* qptr = static_cast<double*>(quad_pts.request().ptr);
    const double* lptr = static_cast<double*>(loc.request().ptr);
    const double* vptr = static_cast<double*>(vpar.request().ptr);

    const DerivativeConstants dc = make_timestep_constants(x1, x2, x3, m, q, psi0, v_total, tol, n_points);

    const std::vector<float> quad_f = to_f32(qptr, quad_pts.size());
    const std::vector<float> loc_f  = to_f32(lptr, loc.size());
    const std::vector<float> vpar_f = to_f32(vptr, n_points);

    // Output: 5 floats per particle = [t, x1, x2, zeta, v_par].
    const size_t out_count = 5 * static_cast<size_t>(n_points);
    auto& ctx = metal_context();

    @autoreleasepool {
        auto make_buf = [&](const void* data, size_t bytes) {
            return [ctx.device newBufferWithBytes:data length:bytes
                                         options:MTLResourceStorageModeShared];
        };

        id<MTLBuffer> quad_m = make_buf(quad_f.data(), quad_f.size() * sizeof(float));
        id<MTLBuffer> loc_m  = make_buf(loc_f.data(),  loc_f.size()  * sizeof(float));
        id<MTLBuffer> vpar_m = make_buf(vpar_f.data(), vpar_f.size() * sizeof(float));
        id<MTLBuffer> out_m  = [ctx.device newBufferWithLength:out_count * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> dc_m   = make_buf(&dc, sizeof(dc));

        id<MTLCommandBuffer>         cb  = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx.pipeline_timestep_boozer_vac];
        [enc setBuffer:quad_m offset:0 atIndex:0];
        [enc setBuffer:loc_m  offset:0 atIndex:1];
        [enc setBuffer:vpar_m offset:0 atIndex:2];
        [enc setBuffer:out_m  offset:0 atIndex:3];
        [enc setBuffer:dc_m   offset:0 atIndex:4];

        const NSUInteger total = static_cast<NSUInteger>(n_points);
        const NSUInteger tpg   = std::min<NSUInteger>(
            ctx.pipeline_timestep_boozer_vac.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            std::string err = cb.error
                ? std::string([[cb.error localizedDescription] UTF8String]) : "unknown";
            throw std::runtime_error("Metal timestep dispatch failed: " + err);
        }

        // Convert f32 output to f64 and transform (x1, x2) -> (s, theta).
        py::array_t<double> result(out_count);
        const float* src = static_cast<const float*>([out_m contents]);
        double*      dst = static_cast<double*>(result.request().ptr);
        for (int i = 0; i < n_points; ++i) {
            dst[5 * i + 0] = static_cast<double>(src[5 * i + 0]);  // t
            double x1v     = static_cast<double>(src[5 * i + 1]);
            double x2v     = static_cast<double>(src[5 * i + 2]);
            dst[5 * i + 1] = std::sqrt(x1v * x1v + x2v * x2v);    // s
            dst[5 * i + 2] = std::atan2(x2v, x1v);                 // theta
            dst[5 * i + 3] = static_cast<double>(src[5 * i + 3]);  // zeta
            dst[5 * i + 4] = static_cast<double>(src[5 * i + 4]);  // v_par
        }
        return result;
    }
}

// Full Boozer-vacuum tracing loop: integrate each particle from t=0 to tmax.
// loc: (n_points, 3) = [s, theta, zeta].
// vpar: (n_points,).
// Returns flat array of 5*n_points doubles: [t, s, theta, zeta, v_par] per particle.
extern "C" py::array_t<double> metal_boozer_vacuum_tracing(
    py::array_t<double> quad_pts,
    py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range,
    py::array_t<double> loc, py::array_t<double> vpar,
    double v_total, double m, double q, double psi0,
    double tmax, double tol, int n_points)
{
    const double* x1   = static_cast<double*>(x1_range.request().ptr);
    const double* x2   = static_cast<double*>(x2_range.request().ptr);
    const double* x3   = static_cast<double*>(x3_range.request().ptr);
    const double* qptr = static_cast<double*>(quad_pts.request().ptr);
    const double* lptr = static_cast<double*>(loc.request().ptr);
    const double* vptr = static_cast<double*>(vpar.request().ptr);

    // Build constants: tol floored to f32-achievable level, tmax passed through.
    DerivativeConstants dc = make_deriv_constants(x1, x2, x3, m, q, psi0, v_total, n_points);
    dc.tol  = static_cast<float>(std::max(tol, 1e-5));
    dc.tmax = static_cast<float>(tmax);

    const std::vector<float> quad_f = to_f32(qptr, quad_pts.size());
    const std::vector<float> loc_f  = to_f32(lptr, loc.size());
    const std::vector<float> vpar_f = to_f32(vptr, n_points);

    const size_t out_count = 5 * static_cast<size_t>(n_points);
    auto& ctx = metal_context();

    @autoreleasepool {
        auto make_buf = [&](const void* data, size_t bytes) {
            return [ctx.device newBufferWithBytes:data length:bytes
                                         options:MTLResourceStorageModeShared];
        };

        id<MTLBuffer> quad_m = make_buf(quad_f.data(), quad_f.size() * sizeof(float));
        id<MTLBuffer> loc_m  = make_buf(loc_f.data(),  loc_f.size()  * sizeof(float));
        id<MTLBuffer> vpar_m = make_buf(vpar_f.data(), vpar_f.size() * sizeof(float));
        id<MTLBuffer> out_m  = [ctx.device newBufferWithLength:out_count * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> dc_m   = make_buf(&dc, sizeof(dc));

        id<MTLCommandBuffer>         cb  = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx.pipeline_tracing_boozer_vac];
        [enc setBuffer:quad_m offset:0 atIndex:0];
        [enc setBuffer:loc_m  offset:0 atIndex:1];
        [enc setBuffer:vpar_m offset:0 atIndex:2];
        [enc setBuffer:out_m  offset:0 atIndex:3];
        [enc setBuffer:dc_m   offset:0 atIndex:4];

        const NSUInteger total = static_cast<NSUInteger>(n_points);
        const NSUInteger tpg   = std::min<NSUInteger>(
            ctx.pipeline_tracing_boozer_vac.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            std::string err = cb.error
                ? std::string([[cb.error localizedDescription] UTF8String]) : "unknown";
            throw std::runtime_error("Metal tracing dispatch failed: " + err);
        }

        py::array_t<double> result(out_count);
        const float* src = static_cast<const float*>([out_m contents]);
        double*      dst = static_cast<double*>(result.request().ptr);
        for (int i = 0; i < n_points; ++i) {
            dst[5 * i + 0] = static_cast<double>(src[5 * i + 0]);  // t
            double x1v     = static_cast<double>(src[5 * i + 1]);
            double x2v     = static_cast<double>(src[5 * i + 2]);
            dst[5 * i + 1] = std::sqrt(x1v * x1v + x2v * x2v);    // s
            dst[5 * i + 2] = std::atan2(x2v, x1v);                 // theta
            dst[5 * i + 3] = static_cast<double>(src[5 * i + 3]);  // zeta
            dst[5 * i + 4] = static_cast<double>(src[5 * i + 4]);  // v_par
        }
        return result;
    }
}

extern "C" py::array_t<double> test_gpu_derivatives_cartesian(
    py::array_t<double> quad_pts,
    py::array_t<double> x1_range, py::array_t<double> x2_range, py::array_t<double> x3_range,
    py::array_t<double> loc, py::array_t<double> vpar, py::array_t<double> time,
    double v_total, double m, double q, int n_points)
{
    const double* x1   = static_cast<double*>(x1_range.request().ptr);
    const double* x2   = static_cast<double*>(x2_range.request().ptr);
    const double* x3   = static_cast<double*>(x3_range.request().ptr);
    const double* qptr = static_cast<double*>(quad_pts.request().ptr);
    const double* lptr = static_cast<double*>(loc.request().ptr);
    const double* vptr = static_cast<double*>(vpar.request().ptr);
    const double* tptr = static_cast<double*>(time.request().ptr);

    // psi0 is not used in the Cartesian RHS; pass 0.0.
    const DerivativeConstants dc = make_deriv_constants(x1, x2, x3, m, q, 0.0, v_total, n_points);
    return metal_dispatch_derivs(
        metal_context().pipeline_derivs_cartesian,
        to_f32(qptr, quad_pts.size()),
        to_f32(lptr, loc.size()),
        to_f32(vptr, n_points),
        to_f32(tptr, n_points),
        dc);
}

