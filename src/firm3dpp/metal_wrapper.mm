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

    double x1_start;
    double x2_start;
    double x2_period;
    double x3_start;
    double x3_period;

    double x1_step;
    double x2_step;
    double x3_step;
};

struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
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

            id<MTLFunction> fn = [lib newFunctionWithName:@"test_gpu_interpolation_kernel"];
            if (fn == nil) {
                throw std::runtime_error("Kernel test_gpu_interpolation_kernel not found in " + kernel_path);
            }

            NSError* pso_error = nil;
            local.pipeline = [local.device newComputePipelineStateWithFunction:fn error:&pso_error];
            if (local.pipeline == nil) {
                std::string err = pso_error ? std::string([[pso_error localizedDescription] UTF8String]) : "unknown";
                throw std::runtime_error("Failed to create Metal pipeline: " + err);
            }
        }
        return local;
    }();

    return ctx;
}

struct RhsSpec {
    int mode;
    int n_fields;
};

RhsSpec parse_rhs(const std::string& rhs) {
    if (rhs == "cartesian_vacuum") {
        return RhsSpec{F3D_RHS_CARTESIAN_VACUUM, 7};
    }
    if (rhs == "boozer_vacuum") {
        return RhsSpec{F3D_RHS_BOOZER_VACUUM, 6};
    }
    if (rhs == "boozer_saw_vacuum") {
        return RhsSpec{F3D_RHS_BOOZER_SAW_VACUUM, 10};
    }
    if (rhs == "boozer") {
        return RhsSpec{F3D_RHS_BOOZER, 12};
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
    constants.x1_start = x1_ptr[0];
    constants.x2_start = x2_ptr[0];
    constants.x2_period = x2_ptr[1];
    constants.x3_start = x3_ptr[0];
    constants.x3_period = x3_ptr[1];
    constants.x1_step = x1_step;
    constants.x2_step = x2_step;
    constants.x3_step = x3_step;

    auto& ctx = metal_context();

    @autoreleasepool {
        id<MTLBuffer> quad_m = [ctx.device newBufferWithBytes:quad_ptr length:quad_pts.size() * sizeof(double) options:MTLResourceStorageModeShared];
        id<MTLBuffer> loc_m = [ctx.device newBufferWithBytes:loc_host.data() length:loc_host.size() * sizeof(double) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_m = [ctx.device newBufferWithLength:size_t(n_fields) * size_t(n_points) * sizeof(double) options:MTLResourceStorageModeShared];
        id<MTLBuffer> c_m = [ctx.device newBufferWithBytes:&constants length:sizeof(constants) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx.pipeline];
        [enc setBuffer:quad_m offset:0 atIndex:0];
        [enc setBuffer:loc_m offset:0 atIndex:1];
        [enc setBuffer:out_m offset:0 atIndex:2];
        [enc setBuffer:c_m offset:0 atIndex:3];

        const NSUInteger total_threads = static_cast<NSUInteger>(n_points);
        const NSUInteger threads_per_group = std::min<NSUInteger>(ctx.pipeline.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status == MTLCommandBufferStatusError) {
            std::string err = cb.error ? std::string([[cb.error localizedDescription] UTF8String]) : "unknown";
            throw std::runtime_error("Metal test_gpu_interpolation dispatch failed: " + err);
        }

        py::array_t<double> result(n_fields * n_points);
        py::buffer_info result_buf = result.request();
        std::memcpy(result_buf.ptr, [out_m contents], size_t(n_fields) * size_t(n_points) * sizeof(double));
        return result;
    }
}
