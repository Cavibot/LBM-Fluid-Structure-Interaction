// export_curvature_golden.cu
// Generates reference curvature data for PLIC-offset cross-validation.
//
// Each scene writes to its own subdirectory under golden_data/curvature/.
//
// Build: via CMake target export_curvature_golden
// Run:   export_curvature_golden.exe

#include <cstdio>
#include <cmath>
#include <cstdlib>

// MLCUDA_DEVICE must be defined BEFORE including mrUtilFuncGpu3D.h so that
// MLFUNC_TYPE resolves to __host__ __device__ (not __host__ only).
#define MLCUDA_DEVICE

#include "mrUtilFuncGpu3D.h"

// ---- helpers ----
static void wf(const char* fn, const float* d, int n) {
    FILE* f = fopen(fn, "w");
    if (!f) { printf("ERR %s\n", fn); return; }
    for (int i = 0; i < n; i++) fprintf(f, "%.9g\n", d[i]);
    fclose(f);
    printf("  wrote %s (%d values)\n", fn, n);
}

static void wi(const char* fn, const int* d, int n) {
    FILE* f = fopen(fn, "w");
    if (!f) { printf("ERR %s\n", fn); return; }
    for (int i = 0; i < n; i++) fprintf(f, "%d\n", d[i]);
    fclose(f);
    printf("  wrote %s (%d values)\n", fn, n);
}

static void make_dir(const char* path) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir \"%s\" 2>nul", path);
    system(cmd);
}

// ---- CUDA kernel: compute curvature for all TYPE_I cells ----
__global__ void curvature_kernel(
    const float* phi,
    const unsigned char* flag,
    int nx, int ny, int nz,
    float* curvatures,
    int* type_i_indices,
    int* type_i_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = k * ny * nx + j * nx + i;
    unsigned char fl = flag[idx];
    unsigned char su = fl & TYPE_SU;

    if (su == TYPE_I) {
        float phit[27];
        phit[0] = phi[idx];
        for (int di = 1; di < 27; di++) {
            int ni = i - (int)ex3d_gpu[di];
            int nj = j - (int)ey3d_gpu[di];
            int nk = k - (int)ez3d_gpu[di];
            bool oob = (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz);
            if (oob) {
                phit[di] = phi[idx];
            } else {
                int nidx = nk * ny * nx + nj * nx + ni;
                float nphi = phi[nidx];
                // Solid-neighbour fallback (ref: mrLbmSolverGpu3D.cu:843-860)
                if ((flag[nidx] & TYPE_BO) == TYPE_S) {
                    for (int fd = 0; fd < 6; fd++) {
                        int fi = ni - (int)ex3d_gpu[fd + 1];
                        int fj = nj - (int)ey3d_gpu[fd + 1];
                        int fk = nk - (int)ez3d_gpu[fd + 1];
                        if (fi >= 0 && fi < nx && fj >= 0 && fj < ny && fk >= 0 && fk < nz) {
                            int fidx = fk * ny * nx + fj * nx + fi;
                            if ((flag[fidx] & TYPE_BO) != TYPE_S) {
                                nphi = phi[fidx]; break;
                            }
                        }
                    }
                }
                phit[di] = nphi;
            }
        }

        mrUtilFuncGpu3D util;
        float K = util.calculate_curvature(phit);

        int pos = atomicAdd(type_i_count, 1);
        curvatures[pos] = K;
        type_i_indices[pos] = idx;
    }
}

// ---- Scene setup helpers ----
static void setup_sphere(float* phi, unsigned char* flag, int N, double R,
                          double cx, double cy, double cz, double delta = 1.5) {
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int idx = k * N * N + j * N + i;
                double dx = i - cx, dy = j - cy, dz = k - cz;
                double dist = sqrt(dx * dx + dy * dy + dz * dz);
                double phi_raw = (R - dist) / delta + 0.5;
                double phi_val = phi_raw < 0.0 ? 0.0 : (phi_raw > 1.0 ? 1.0 : phi_raw);
                phi[idx] = (float)phi_val;
                if (phi_val >= 1.0 - 1e-6) flag[idx] = TYPE_F;
                else if (phi_val <= 1e-6) flag[idx] = TYPE_G;
                else flag[idx] = TYPE_I;
            }
        }
    }
}

static void setup_plane(float* phi, unsigned char* flag, int N, int z_split,
                         double delta = 1.5) {
    for (int k = 0; k < N; k++) {
        double phi_raw = (z_split - k) / delta + 0.5;
        double phi_val = phi_raw < 0.0 ? 0.0 : (phi_raw > 1.0 ? 1.0 : phi_raw);
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int idx = k * N * N + j * N + i;
                phi[idx] = (float)phi_val;
                if (phi_val >= 1.0 - 1e-6) flag[idx] = TYPE_F;
                else if (phi_val <= 1e-6) flag[idx] = TYPE_G;
                else flag[idx] = TYPE_I;
            }
        }
    }
}

static void setup_cylinder(float* phi, unsigned char* flag, int N, double R,
                            double delta = 1.5) {
    double cx = N / 2.0, cz = N / 2.0;
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int idx = k * N * N + j * N + i;
                double dx = i - cx, dz = k - cz;
                double r = sqrt(dx * dx + dz * dz);
                double phi_raw = (R - r) / delta + 0.5;
                double phi_val = phi_raw < 0.0 ? 0.0 : (phi_raw > 1.0 ? 1.0 : phi_raw);
                phi[idx] = (float)phi_val;
                if (phi_val >= 1.0 - 1e-6) flag[idx] = TYPE_F;
                else if (phi_val <= 1e-6) flag[idx] = TYPE_G;
                else flag[idx] = TYPE_I;
            }
        }
    }
}

// ---- Run curvature computation for a scene and export to subdirectory ----
static void gen_curvature_scene(const char* name, int N,
                                 void (*setup)(float*, unsigned char*, int),
                                 const char* out_dir) {
    printf("\n=== Curvature %s (%d^3) ===\n", name, N);

    int total = N * N * N;

    // Create scene subdirectory
    char scene_dir[512];
    snprintf(scene_dir, sizeof(scene_dir), "%s/%s", out_dir, name);
    make_dir(scene_dir);

    // Host arrays
    float* phi_h = new float[total];
    unsigned char* flag_h = new unsigned char[total];
    setup(phi_h, flag_h, N);

    // Set boundary walls (6 faces) to TYPE_S with phi=0.
    // This matches actual simulation conditions where TYPE_I cells
    // are protected from grid-boundary out-of-bounds by TYPE_S walls.
    for (int k = 0; k < N; k++) for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
        if (i == 0 || i == N-1 || j == 0 || j == N-1 || k == 0 || k == N-1) {
            int idx = k * N * N + j * N + i;
            flag_h[idx] = TYPE_S; phi_h[idx] = 0.0f;
        }
    }

    // Device arrays
    float *phi_d, *curv_d;
    unsigned char* flag_d;
    int *indices_d, *count_d;

    cudaMalloc(&phi_d, total * sizeof(float));
    cudaMalloc(&flag_d, total * sizeof(unsigned char));
    cudaMalloc(&curv_d, total * sizeof(float));
    cudaMalloc(&indices_d, total * sizeof(int));
    cudaMalloc(&count_d, sizeof(int));

    cudaMemcpy(phi_d, phi_h, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(flag_d, flag_h, total * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(count_d, 0, sizeof(int));

    dim3 block(8, 8, 4);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y,
              (N + block.z - 1) / block.z);
    curvature_kernel<<<grid, block>>>(phi_d, flag_d, N, N, N,
                                       curv_d, indices_d, count_d);
    cudaDeviceSynchronize();

    int count_h;
    cudaMemcpy(&count_h, count_d, sizeof(int), cudaMemcpyDeviceToHost);

    float* curv_h = new float[count_h > 0 ? count_h : 1];
    int* indices_h = new int[count_h > 0 ? count_h : 1];
    cudaMemcpy(curv_h, curv_d, count_h * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(indices_h, indices_d, count_h * sizeof(int), cudaMemcpyDeviceToHost);

    printf("  TYPE_I cells: %d\n", count_h);

    char path[512];
    snprintf(path, sizeof(path), "%s/phi.txt", scene_dir);   wf(path, phi_h, total);
    snprintf(path, sizeof(path), "%s/flag.txt", scene_dir);
    int* flag_int = new int[total];
    for (int i = 0; i < total; i++) flag_int[i] = (int)flag_h[i];
    wi(path, flag_int, total);
    delete[] flag_int;

    snprintf(path, sizeof(path), "%s/values.txt", scene_dir);   wf(path, curv_h, count_h);
    snprintf(path, sizeof(path), "%s/indices.txt", scene_dir);   wi(path, indices_h, count_h);

    delete[] phi_h; delete[] flag_h;
    delete[] curv_h; delete[] indices_h;
    cudaFree(phi_d); cudaFree(flag_d); cudaFree(curv_d);
    cudaFree(indices_d); cudaFree(count_d);
}

int main() {
    printf("=== HOME-FSLBM Curvature Golden Data Generator ===\n");

    const char* base = "../../../golden_data/curvature";
    make_dir(base);

    gen_curvature_scene("sphere_r4_16", 16,
        [](float* p, unsigned char* f, int N) {
            setup_sphere(p, f, N, 4.0, N/2.0, N/2.0, N/2.0);
        }, base);

    gen_curvature_scene("sphere_r8_32", 32,
        [](float* p, unsigned char* f, int N) {
            setup_sphere(p, f, N, 8.0, N/2.0, N/2.0, N/2.0);
        }, base);

    gen_curvature_scene("plane_32", 32,
        [](float* p, unsigned char* f, int N) {
            setup_plane(p, f, N, N/2);
        }, base);

    gen_curvature_scene("cylinder_r6_32", 32,
        [](float* p, unsigned char* f, int N) {
            setup_cylinder(p, f, N, 6.0);
        }, base);

    printf("\nDone.\n");
    return 0;
}