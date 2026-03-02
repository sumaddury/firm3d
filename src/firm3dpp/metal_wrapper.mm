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

struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline_f64 = nil;
    id<MTLComputePipelineState> pipeline_f32 = nil;
};

std::string load_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string find_kernel_source_path() {
    if (const char* env = std::getenv("FIRM3DPP_METAL_KERNEL_PATH")) {
        return std::string(env);
    }

    const std::vector<std::string> candidates = {
        "apple_kernel.metal",
        "src/firm3dpp/apple_kernel.metal"
    };
    for (const auto& p : candidates) {
        std::ifstream in(p);
        if (in.good()) {
            return p;
        }
    }

    throw std::runtime_error(
        "Could not find apple_kernel.metal. Set FIRM3DPP_METAL_KERNEL_PATH to the full path."
    );
}

id<MTLComputePipelineState> create_pipeline_state(id<MTLDevice> device, id<MTLLibrary> lib, NSString* kernel_name) {
    id<MTLFunction> fn = [lib newFunctionWithName:kernel_name];
    if (fn == nil) {
        throw std::runtime_error(
            std::string("Kernel ") + std::string([kernel_name UTF8String]) + " not found in Metal source."
        );
    }

    NSError* pso_error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&pso_error];
    if (pipeline == nil) {
        std::string err = pso_error ? std::string([[pso_error localizedDescription] UTF8String]) : "unknown";
        throw std::runtime_error(
            std::string("Failed to create Metal pipeline for ")
            + std::string([kernel_name UTF8String]) + ": " + err
        );
    }
    return pipeline;
}

MetalContext& metal_context() {
    static MetalContext ctx = []() {
        MetalContext local;
        @autoreleasepool {
            local.device = MTLCreateSystemDefaultDevice();
            if (local.device == nil) {
                throw std::runtime_error("Metal device not available");
            }

            local.queue = [local.device newCommandQueue];
            if (local.queue == nil) {
                throw std::runtime_error("Failed to create Metal command queue");
            }

            const std::string kernel_path = find_kernel_source_path();
            const std::string source = load_file(kernel_path);
            NSString* ns_source = [NSString stringWithUTF8String:source.c_str()];

            NSError* lib_error = nil;
            id<MTLLibrary> lib = [local.device newLibraryWithSource:ns_source options:nil error:&lib_error];
            if (lib == nil) {
                std::string err = lib_error ? std::string([[lib_error localizedDescription] UTF8String]) : "unknown";
                throw std::runtime_error("Failed to compile Metal source " + kernel_path + ": " + err);
            }

            local.pipeline_f64 = create_pipeline_state(local.device, lib, @"test_gpu_interpolation_kernel");
            local.pipeline_f32 = create_pipeline_state(local.device, lib, @"test_gpu_interpolation_kernel_f32");
        }
        return local;
    }();

    return ctx;
}

struct RhsSpec {
    int mode;
    int n_fields;
    bool use_float;
};

bool has_suffix(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

RhsSpec parse_rhs(std::string rhs) {
    bool use_float = true;
    if (has_suffix(rhs, "_f32")) {
        use_float = true;
        rhs.erase(rhs.size() - 4);
    } else if (has_suffix(rhs, "_f64")) {
        throw std::invalid_argument("Metal f64 mode is not supported on this device/build. Use *_f32.");
    }

    if (rhs == "cartesian_vacuum") {
        return RhsSpec{F3D_RHS_CARTESIAN_VACUUM, 7, use_float};
    }
    if (rhs == "boozer_vacuum") {
        return RhsSpec{F3D_RHS_BOOZER_VACUUM, 6, use_float};
    }
    if (rhs == "boozer_saw_vacuum") {
        return RhsSpec{F3D_RHS_BOOZER_SAW_VACUUM, 10, use_float};
    }
    if (rhs == "boozer") {
        return RhsSpec{F3D_RHS_BOOZER, 12, use_float};
    }
    throw std::invalid_argument("Unsupported rhs: " + rhs);
}

void preprocess_loc_for_cuda_compat(std::vector<double>& loc, int rhs_mode, int n_points) {
    // Keep CUDA host behavior identical: convert (r,phi,z)/(s,theta,zeta) to cartesianized x/y first.
    if (rhs_mode == F3D_RHS_CARTESIAN_VACUUM) {
        for (int i = 0; i < n_points; ++i) {
            const double r = loc[3 * i + 0];
            const double phi = loc[3 * i + 1];
            loc[3 * i + 0] = r * std::cos(phi);
            loc[3 * i + 1] = r * std::sin(phi);
        }
    } else {
        for (int i = 0; i < n_points; ++i) {
            const double s = loc[3 * i + 0];
            const double theta = loc[3 * i + 1];
            loc[3 * i + 0] = s * std::cos(theta);
            loc[3 * i + 1] = s * std::sin(theta);
        }
    }
}

} // namespace

extern "C" py::array_t<double> test_gpu_interpolation(
    py::array_t<double> quad_pts,
    py::array_t<double> x1_range,
    py::array_t<double> x2_range,
    py::array_t<double> x3_range,
    py::array_t<double> loc,
    std::string rhs,
    int n_points
) {
    py::buffer_info quad_buf = quad_pts.request();
    py::buffer_info x1_buf = x1_range.request();
    py::buffer_info x2_buf = x2_range.request();
    py::buffer_info x3_buf = x3_range.request();
    py::buffer_info loc_buf = loc.request();

    double* quad_ptr = static_cast<double*>(quad_buf.ptr);
    double* x1_ptr = static_cast<double*>(x1_buf.ptr);
    double* x2_ptr = static_cast<double*>(x2_buf.ptr);
    double* x3_ptr = static_cast<double*>(x3_buf.ptr);
    double* loc_ptr = static_cast<double*>(loc_buf.ptr);

    const RhsSpec rhs_spec = parse_rhs(rhs);
    const int rhs_mode = rhs_spec.mode;
    const int n_fields = rhs_spec.n_fields;
    const bool use_float = rhs_spec.use_float;

    const int x1_count = static_cast<int>(x1_ptr[2]);
    const int x2_count = static_cast<int>(x2_ptr[2]);
    const int x3_count = static_cast<int>(x3_ptr[2]);

    const double x1_step = (x1_ptr[1] - x1_ptr[0]) / (x1_ptr[2] - 1.0);
    const double x2_step = (x2_ptr[1] - x2_ptr[0]) / (x2_ptr[2] - 1.0);
    const double x3_step = (x3_ptr[1] - x3_ptr[0]) / (x3_ptr[2] - 1.0);

    const int n_x2 = (x2_count - 1) / 3;
    const int n_x3 = (x3_count - 1) / 3;
    const int n_x23 = n_x2 * n_x3;

    std::vector<double> loc_host(loc_ptr, loc_ptr + loc.size());
    preprocess_loc_for_cuda_compat(loc_host, rhs_mode, n_points);

    InterpolationTestConstants constants{};
    constants.n_x2 = n_x2;
    constants.n_x3 = n_x3;
    constants.n_x23 = n_x23;
    constants.n_fields = n_fields;
    constants.n_points = n_points;
    constants.rhs_mode = rhs_mode;
    constants.x1_count = x1_count;
    constants.x2_count = x2_count;
    constants.x3_count = x3_count;
    constants.x1_start = static_cast<float>(x1_ptr[0]);
    constants.x2_start = static_cast<float>(x2_ptr[0]);
    constants.x2_period = static_cast<float>(x2_ptr[1]);
    constants.x3_start = static_cast<float>(x3_ptr[0]);
    constants.x3_period = static_cast<float>(x3_ptr[1]);
    constants.x1_step = static_cast<float>(x1_step);
    constants.x2_step = static_cast<float>(x2_step);
    constants.x3_step = static_cast<float>(x3_step);

    auto& ctx = metal_context();

    @autoreleasepool {
        id<MTLBuffer> c_m = [ctx.device newBufferWithBytes:&constants length:sizeof(constants)
                                                    options:MTLResourceStorageModeShared];
        const NSUInteger total_threads = static_cast<NSUInteger>(n_points);
        const size_t out_count = size_t(n_fields) * size_t(n_points);

        if (use_float) {
            std::vector<float> quad_host(quad_pts.size());
            std::vector<float> loc_host_f(loc_host.size());
            for (size_t i = 0; i < quad_host.size(); ++i) {
                quad_host[i] = static_cast<float>(quad_ptr[i]);
            }
            for (size_t i = 0; i < loc_host_f.size(); ++i) {
                loc_host_f[i] = static_cast<float>(loc_host[i]);
            }

            id<MTLBuffer> quad_m = [ctx.device newBufferWithBytes:quad_host.data()
                                                            length:quad_host.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> loc_m = [ctx.device newBufferWithBytes:loc_host_f.data()
                                                           length:loc_host_f.size() * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
            id<MTLBuffer> out_m = [ctx.device newBufferWithLength:out_count * sizeof(float)
                                                           options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx.pipeline_f32];
            [enc setBuffer:quad_m offset:0 atIndex:0];
            [enc setBuffer:loc_m offset:0 atIndex:1];
            [enc setBuffer:out_m offset:0 atIndex:2];
            [enc setBuffer:c_m offset:0 atIndex:3];

            const NSUInteger threads_per_group = std::min<NSUInteger>(ctx.pipeline_f32.maxTotalThreadsPerThreadgroup, 256);
            [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            if (cb.status == MTLCommandBufferStatusError) {
                std::string err = cb.error ? std::string([[cb.error localizedDescription] UTF8String]) : "unknown";
                throw std::runtime_error("Metal f32 test_gpu_interpolation dispatch failed: " + err);
            }

            py::array_t<double> result(out_count);
            py::buffer_info result_buf = result.request();
            const float* out_ptr = static_cast<const float*>([out_m contents]);
            double* result_ptr = static_cast<double*>(result_buf.ptr);
            for (size_t i = 0; i < out_count; ++i) {
                result_ptr[i] = static_cast<double>(out_ptr[i]);
            }
            return result;
        } else {
            id<MTLBuffer> quad_m = [ctx.device newBufferWithBytes:quad_ptr
                                                            length:quad_pts.size() * sizeof(double)
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> loc_m = [ctx.device newBufferWithBytes:loc_host.data()
                                                           length:loc_host.size() * sizeof(double)
                                                          options:MTLResourceStorageModeShared];
            id<MTLBuffer> out_m = [ctx.device newBufferWithLength:out_count * sizeof(double)
                                                           options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx.pipeline_f64];
            [enc setBuffer:quad_m offset:0 atIndex:0];
            [enc setBuffer:loc_m offset:0 atIndex:1];
            [enc setBuffer:out_m offset:0 atIndex:2];
            [enc setBuffer:c_m offset:0 atIndex:3];

            const NSUInteger threads_per_group = std::min<NSUInteger>(ctx.pipeline_f64.maxTotalThreadsPerThreadgroup, 256);
            [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            if (cb.status == MTLCommandBufferStatusError) {
                std::string err = cb.error ? std::string([[cb.error localizedDescription] UTF8String]) : "unknown";
                throw std::runtime_error("Metal f64 test_gpu_interpolation dispatch failed: " + err);
            }

            py::array_t<double> result(out_count);
            py::buffer_info result_buf = result.request();
            std::memcpy(result_buf.ptr, [out_m contents], out_count * sizeof(double));
            return result;
        }
    }
}
