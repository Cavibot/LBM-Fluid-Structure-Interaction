// export_gas_foam_golden.cpp
// Phase 4 CMR-MRT gas + foam golden-data generator for Warp regression tests.
 //
 // Build target: export_gas_foam_golden  (see CMakeLists.txt)
 // Run from a CUDA-capable machine after building Home-FSLBM.
 //
 // Default output (from out/build/...):
 //   ../../../golden_data/{gas_*,foam_*}/
 //
 // Then copy those directories into:
 //   wanphys/_src/fluid/fluid_grid/home_fslbm/tests/golden_data/
 //
 // Flat layout: idx = x + nx*(y + ny*z)  (x-fastest; same as mrFlow3D).

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mrSolver3D.h"
#include "mrFlow3D.h"
#include "mrLbmSolverGpu3D.h"
#include "tDCCL.cuh"
#include "mlLbmCommon.h"

using namespace Mfree;

// ============================================================================
// I/O helpers
// ============================================================================

static void make_dir(const char* path) {
    char cmd[512];
#ifdef _WIN32
    snprintf(cmd, sizeof(cmd), "mkdir \"%s\" 2>nul", path);
#else
    snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\"", path);
#endif
    system(cmd);
}

static void wf(const char* fn, const float* d, int n) {
    FILE* f = fopen(fn, "w");
    if (!f) { printf("ERR open %s\n", fn); return; }
    for (int i = 0; i < n; i++) {
        double v = (double)d[i];
        if (std::isnan(v)) fprintf(f, "nan\n");
        else if (std::isinf(v)) fprintf(f, "%s\n", v > 0 ? "inf" : "-inf");
        else fprintf(f, "%.15e\n", v);
    }
    fclose(f);
    printf("  %s (%d floats)\n", fn, n);
}

static void wd(const char* fn, const double* d, int n) {
    FILE* f = fopen(fn, "w");
    if (!f) { printf("ERR open %s\n", fn); return; }
    for (int i = 0; i < n; i++) {
        if (std::isnan(d[i])) fprintf(f, "nan\n");
        else if (std::isinf(d[i])) fprintf(f, "%s\n", d[i] > 0 ? "inf" : "-inf");
        else fprintf(f, "%.15e\n", d[i]);
    }
    fclose(f);
    printf("  %s (%d doubles)\n", fn, n);
}

static void wi(const char* fn, const int* d, int n) {
    FILE* f = fopen(fn, "w");
    if (!f) { printf("ERR open %s\n", fn); return; }
    for (int i = 0; i < n; i++) fprintf(f, "%d\n", d[i]);
    fclose(f);
    printf("  %s (%d ints)\n", fn, n);
}

static void wi_scalar(const char* fn, int v) { wi(fn, &v, 1); }

static void wf_scalar(const char* fn, float v) { wf(fn, &v, 1); }

static long idx3(int i, int j, int k, int nx, int ny) {
    return (long)k * ny * nx + (long)j * nx + i;
}

// ============================================================================
// GPU ↔ host gas/foam transfer
// ============================================================================

static void trans_gas_foam_to_host(mrSolver3D* s) {
    mrFlow3D* host = s->lbmvec;
    mrFlow3D hdev;
    checkCudaErrors(cudaMemcpy(&hdev, s->lbm_dev_gpu, sizeof(mrFlow3D), cudaMemcpyDeviceToHost));

    long N = host->count;
    int maxb = (int)host->bubble.max_bubble_count;

    checkCudaErrors(cudaMemcpy(host->flag, hdev.flag, N * sizeof(MLLATTICENODE_SURFACE_FLAG), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->phi, hdev.phi, N * sizeof(REAL), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->mass, hdev.mass, N * sizeof(REAL), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->massex, hdev.massex, N * sizeof(REAL), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->fMom, hdev.fMom, 10 * N * sizeof(REAL), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->tag_matrix, hdev.tag_matrix, N * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->previous_tag, hdev.previous_tag, N * sizeof(int), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(host->gMom, hdev.gMom, 7 * N * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->gMomPost, hdev.gMomPost, 7 * N * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->delta_g, hdev.delta_g, N * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->c_value, hdev.c_value, N * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->src, hdev.src, N * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->disjoin_force, hdev.disjoin_force, N * sizeof(float), cudaMemcpyDeviceToHost));

    host->merge_flag = hdev.merge_flag;
    host->split_flag = hdev.split_flag;

    checkCudaErrors(cudaMemcpy(host->bubble.volume, hdev.bubble.volume, maxb * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->bubble.init_volume, hdev.bubble.init_volume, maxb * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->bubble.rho, hdev.bubble.rho, maxb * sizeof(double), cudaMemcpyDeviceToHost));
    host->bubble.bubble_count = hdev.bubble.bubble_count;
    host->bubble.label_num = hdev.bubble.label_num;
}

// ============================================================================
// Scene helpers
// ============================================================================

static void export_scalars_grid(const char* out_dir, int nx, int ny, int nz, int steps = -1) {
    char path[512];
    snprintf(path, sizeof(path), "%s/nx.txt", out_dir); wi_scalar(path, nx);
    snprintf(path, sizeof(path), "%s/ny.txt", out_dir); wi_scalar(path, ny);
    snprintf(path, sizeof(path), "%s/nz.txt", out_dir); wi_scalar(path, nz);
    if (steps >= 0) {
        snprintf(path, sizeof(path), "%s/steps.txt", out_dir);
        wi_scalar(path, steps);
    }
}

static void clear_moments(mrFlow3D* fl) {
    long count = fl->count;
    for (long idx = 0; idx < count; idx++) {
        fl->fMom[idx] = 1.0f;
        fl->fMomPost[idx] = 1.0f;
        for (int m = 1; m < 10; m++) {
            fl->fMom[m * count + idx] = 0.0f;
            fl->fMomPost[m * count + idx] = 0.0f;
        }
        fl->forcex[idx] = fl->forcey[idx] = fl->forcez[idx] = 0.0f;
        fl->c_value[idx] = 0.0f;
        fl->src[idx] = 0.0f;
        fl->delta_g[idx] = 0.0f;
        fl->disjoin_force[idx] = 0.0f;
        fl->tag_matrix[idx] = -1;
        fl->previous_tag[idx] = -1;
        fl->previous_merge_tag[idx] = -1;
        fl->input_matrix[idx] = 0;
        fl->label_matrix[idx] = 0;
        fl->merge_detector[idx] = false;
        fl->delta_phi[idx] = 0.0f;
        fl->massex[idx] = 0.0f;
        fl->islet[idx] = 0;
        for (int g = 0; g < 7; g++) {
            fl->gMom[g * count + idx] = 0.0f;
            fl->gMomPost[g * count + idx] = 0.0f;
        }
    }
    fl->merge_flag = 0;
    fl->split_flag = 0;
    fl->bubble.bubble_count = 0;
    fl->bubble.label_num = -1;
}

static void set_boundary_walls(mrFlow3D* fl, int nx, int ny, int nz) {
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
                    long id = idx3(i, j, k, nx, ny);
                    fl->flag[id] = TYPE_S;
                    fl->phi[id] = 0.0f;
                    fl->mass[id] = 0.0f;
                }
}

static void fill_fluid_background(mrFlow3D* fl, int nx, int ny, int nz) {
    clear_moments(fl);
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                long id = idx3(i, j, k, nx, ny);
                fl->flag[id] = TYPE_F;
                fl->phi[id] = 1.0f;
                fl->mass[id] = 1.0f;
            }
}

/** Soft bubble: gas core + interface shell (matches Warp foam/gas examples). */
static void paint_soft_bubble(mrFlow3D* fl, int nx, int ny, int nz,
                              float cx, float cy, float cz, float radius) {
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                float d = std::sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy) + (k - cz) * (k - cz));
                long id = idx3(i, j, k, nx, ny);
                if (d < radius - 0.5f) {
                    fl->flag[id] = TYPE_G;
                    fl->phi[id] = 0.0f;
                    fl->mass[id] = 0.0f;
                } else if (d <= radius + 0.5f) {
                    fl->flag[id] = TYPE_I;
                    fl->phi[id] = 0.5f;
                    fl->mass[id] = 0.5f;
                }
            }
}

static void init_uniform_gas(mrFlow3D* fl, float c0) {
    const float w[7] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    long N = fl->count;
    for (long id = 0; id < N; id++) {
        fl->c_value[id] = c0;
        for (int di = 0; di < 7; di++) {
            fl->gMom[di * N + id] = w[di] * c0;
            fl->gMomPost[di * N + id] = w[di] * c0;
        }
    }
}

static mrSolver3D* build_solver(mrFlow3D** fl_out, int nx, int ny, int nz, REAL gz) {
    REAL Np = 1, l0p = 1, roup = 1, labma = 1, u0p = 1;
    mrFlow3D* fl = new mrFlow3D();
    fl->Create(0, 0, 0, nx, ny, nz, 1, (REAL)nx, (REAL)ny, (REAL)nz, 1.0e-4f, gz);
    fl->BubbleBufferInit(65536);
    mrSolver3D* s = new mrSolver3D();
    s->AttachLbmHost(fl);
    mrFlow3D* fd = 0;
    s->AttachLbmDevice(fd);
    MLMappingParam mp(u0p, labma, l0p, Np, roup);
    s->AttachMapping(mp);
    *fl_out = fl;
    return s;
}

static void export_gas_fields(mrFlow3D* fl, const char* out_dir, int nx, int ny, int nz, int steps,
                              bool write_delta_g, bool write_c_value) {
    make_dir(out_dir);
    export_scalars_grid(out_dir, nx, ny, nz, steps);
    long N = fl->count;
    char path[512];

    int* flag_i = new int[N];
    for (long i = 0; i < N; i++) flag_i[i] = (int)(unsigned char)fl->flag[i];
    snprintf(path, sizeof(path), "%s/flag.txt", out_dir);
    wi(path, flag_i, (int)N);
    delete[] flag_i;

    snprintf(path, sizeof(path), "%s/g_mom.txt", out_dir);
    wf(path, fl->gMom, (int)(7 * N));

    if (write_delta_g) {
        snprintf(path, sizeof(path), "%s/delta_g.txt", out_dir);
        wf(path, fl->delta_g, (int)N);
    }
    if (write_c_value) {
        snprintf(path, sizeof(path), "%s/c_value.txt", out_dir);
        wf(path, fl->c_value, (int)N);
    }
}

// ============================================================================
// Host CMR (G1) — matches mrUtilFuncGpu3D.h:474-518
// ============================================================================

static void host_to_cmr(float ux, float uy, float uz, float* node) {
    float tmp[7];
    for (int i = 0; i < 7; i++) { tmp[i] = node[i]; node[i] = 0.f; }
    const float ex[7] = {0, 1, -1, 0, 0, 0, 0};
    const float ey[7] = {0, 0, 0, 1, -1, 0, 0};
    const float ez[7] = {0, 0, 0, 0, 0, 1, -1};
    for (int k = 0; k < 7; k++) {
        float CX = ex[k] - ux, CY = ey[k] - uy, CZ = ez[k] - uz;
        float f = tmp[k];
        node[0] += f;
        node[1] += f * CX;
        node[2] += f * CY;
        node[3] += f * CZ;
        node[4] += f * (CX * CX - CY * CY);
        node[5] += f * (CX * CX - CZ * CZ);
        node[6] += f * (CX * CX + CY * CY + CZ * CZ);
    }
}

static void host_from_cmr(float U, float V, float W, float* node) {
    float k0 = node[0], k1 = node[1], k2 = node[2], k3 = node[3];
    float k4 = node[4], k5 = node[5], k6 = node[6];
    node[0] = -k0 * U * U - 2 * k1 * U - k0 * V * V - 2 * k2 * V - k0 * W * W - 2 * k3 * W + k0 - k6;
    node[1] = k1 / 2 + k4 / 6 + k5 / 6 + k6 / 6 + (U * k0) / 2 + U * k1 + (U * U * k0) / 2;
    node[2] = k4 / 6 - k1 / 2 + k5 / 6 + k6 / 6 - (U * k0) / 2 + U * k1 + (U * U * k0) / 2;
    node[3] = k2 / 2 - k4 / 3 + k5 / 6 + k6 / 6 + (V * k0) / 2 + V * k2 + (V * V * k0) / 2;
    node[4] = k5 / 6 - k4 / 3 - k2 / 2 + k6 / 6 - (V * k0) / 2 + V * k2 + (V * V * k0) / 2;
    node[5] = k3 / 2 + k4 / 6 - k5 / 3 + k6 / 6 + (W * k0) / 2 + W * k3 + (W * W * k0) / 2;
    node[6] = k4 / 6 - k3 / 2 - k5 / 3 + k6 / 6 - (W * k0) / 2 + W * k3 + (W * W * k0) / 2;
}

static void gen_cmr_vectors(const char* base) {
    printf("\n=== G1 gas_cmr_transform_vectors ===\n");
    std::string dir = std::string(base) + "/gas_cmr_transform_vectors";
    make_dir(dir.c_str());
    float pop_in[7] = {0.4f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float uxuyuz[3] = {0.1f, 0.02f, -0.03f};
    float moment_out[7];
    for (int i = 0; i < 7; i++) moment_out[i] = pop_in[i];
    host_to_cmr(uxuyuz[0], uxuyuz[1], uxuyuz[2], moment_out);

    float moment_in[7] = {1.f, 0.05f, -0.02f, 0.01f, 0.03f, -0.01f, 0.2f};
    float pop_out[7];
    for (int i = 0; i < 7; i++) pop_out[i] = moment_in[i];
    host_from_cmr(uxuyuz[0], uxuyuz[1], uxuyuz[2], pop_out);

    wf((dir + "/pop_in.txt").c_str(), pop_in, 7);
    wf((dir + "/moment_out.txt").c_str(), moment_out, 7);
    wf((dir + "/moment_in.txt").c_str(), moment_in, 7);
    wf((dir + "/pop_out.txt").c_str(), pop_out, 7);
    wf((dir + "/uxuyuz.txt").c_str(), uxuyuz, 3);
}

// ============================================================================
// G2: Henry interface reconstruction (1 step, dump delta_g + g_mom)
// ============================================================================

static void gen_gas_henry_interface_step1(const char* base) {
    printf("\n=== G2 gas_henry_interface_step1 ===\n");
    const int N = 16;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    fill_fluid_background(fl, N, N, N);
    paint_soft_bubble(fl, N, N, N, 8.0f, 8.0f, 8.0f, 4.0f);
    set_boundary_walls(fl, N, N, N);
    init_uniform_gas(fl, 0.01f);

    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());
    // Re-seed gas after Init3D/InitBubble may overwrite distributions
    trans_gas_foam_to_host(s);
    init_uniform_gas(fl, 0.01f);
    s->mlTransData2Gpu();

    launch_g_reconstruction(s->lbm_dev_gpu, fl->param);
    trans_gas_foam_to_host(s);

    std::string out = std::string(base) + "/gas_henry_interface_step1";
    export_gas_fields(fl, out.c_str(), N, N, N, 1, true, true);
    delete s; delete fl;
}

// ============================================================================
// G3: CMR stream-collide for 10 steps
// ============================================================================

static void gen_gas_stream_collide_step10(const char* base) {
    printf("\n=== G3 gas_stream_collide_step10 ===\n");
    const int N = 16;
    const int steps = 10;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    fill_fluid_background(fl, N, N, N);
    paint_soft_bubble(fl, N, N, N, 8.0f, 8.0f, 8.0f, 4.0f);
    set_boundary_walls(fl, N, N, N);
    init_uniform_gas(fl, 0.01f);
    // Mild non-eq bump at domain centre
    {
        long mid = idx3(N / 2, N / 2, N / 2, N, N);
        fl->gMom[1 * fl->count + mid] *= 1.3f;
    }

    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());
    trans_gas_foam_to_host(s);
    init_uniform_gas(fl, 0.01f);
    {
        long mid = idx3(N / 2, N / 2, N / 2, N, N);
        fl->gMom[1 * fl->count + mid] *= 1.3f;
        fl->gMomPost[1 * fl->count + mid] = fl->gMom[1 * fl->count + mid];
    }
    s->mlTransData2Gpu();

    for (int t = 0; t < steps; t++) {
        launch_g_stream_collide(s->lbm_dev_gpu, fl->param, t);
        launch_g_swap(s->lbm_dev_gpu, fl->param);
    }
    trans_gas_foam_to_host(s);

    std::string out = std::string(base) + "/gas_stream_collide_step10";
    export_gas_fields(fl, out.c_str(), N, N, N, steps, false, true);
    delete s; delete fl;
}

// ============================================================================
// G4: gas flux → bubble init_volume
// ============================================================================

static void gen_gas_volume_g_update(const char* base) {
    printf("\n=== G4 gas_volume_g_update ===\n");
    const int N = 16;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    fill_fluid_background(fl, N, N, N);
    paint_soft_bubble(fl, N, N, N, 8.0f, 8.0f, 8.0f, 4.0f);
    set_boundary_walls(fl, N, N, N);

    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());
    trans_gas_foam_to_host(s);

    // Seed synthetic delta_g on TYPE_I tagged cells (IC for volume_g test)
    std::vector<float> delta0((size_t)fl->count, 0.0f);
    for (long id = 0; id < fl->count; id++) {
        if ((int)(unsigned char)fl->flag[id] == (int)TYPE_I && fl->tag_matrix[id] > 0)
            delta0[(size_t)id] = 0.02f;
        fl->delta_g[id] = delta0[(size_t)id];
    }
    // Snapshot init_volume before update
    int bc = fl->bubble.bubble_count > 0 ? fl->bubble.bubble_count : 1;
    std::vector<double> init_before(fl->bubble.init_volume, fl->bubble.init_volume + bc);

    s->mlTransData2Gpu();
    launch_bubble_volume_g_update(s->lbm_dev_gpu, fl->param, 0);
    trans_gas_foam_to_host(s);

    std::string out = std::string(base) + "/gas_volume_g_update";
    make_dir(out.c_str());
    export_scalars_grid(out.c_str(), N, N, N, 1);

    char path[512];
    snprintf(path, sizeof(path), "%s/delta_g.txt", out.c_str());
    wf(path, delta0.data(), (int)fl->count);  // IC (post-run cleared)
    snprintf(path, sizeof(path), "%s/phi.txt", out.c_str());
    wf(path, fl->phi, (int)fl->count);
    snprintf(path, sizeof(path), "%s/tag_matrix.txt", out.c_str());
    wi(path, fl->tag_matrix, (int)fl->count);

    int* flag_i = new int[fl->count];
    for (long i = 0; i < fl->count; i++) flag_i[i] = (int)(unsigned char)fl->flag[i];
    snprintf(path, sizeof(path), "%s/flag.txt", out.c_str());
    wi(path, flag_i, (int)fl->count);
    delete[] flag_i;

    snprintf(path, sizeof(path), "%s/bubble_init_volume.txt", out.c_str());
    wd(path, fl->bubble.init_volume, bc);
    snprintf(path, sizeof(path), "%s/bubble_init_volume_before.txt", out.c_str());
    wd(path, init_before.data(), bc);
    snprintf(path, sizeof(path), "%s/bubble_count.txt", out.c_str());
    wi_scalar(path, fl->bubble.bubble_count);

    delete s; delete fl;
}

// ============================================================================
// F1: disjoint raycast on two close spheres (force before ResetDisjoinForce)
// ============================================================================

static void gen_foam_disjoint_two_spheres(const char* base) {
    printf("\n=== F1 foam_disjoint_two_spheres ===\n");
    const int N = 32;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    fill_fluid_background(fl, N, N, N);
    // r=6, centres 12 and 20 → surface gap ~2
    paint_soft_bubble(fl, N, N, N, 12.0f, 16.0f, 16.0f, 6.0f);
    paint_soft_bubble(fl, N, N, N, 20.0f, 16.0f, 16.0f, 6.0f);
    set_boundary_walls(fl, N, N, N);

    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());

    // Zero force then launch disjoint only (do not run full mrSolver3DGpu — it resets)
    launch_calculate_disjoint(s->lbm_dev_gpu, fl->param);
    trans_gas_foam_to_host(s);

    std::string out = std::string(base) + "/foam_disjoint_two_spheres";
    make_dir(out.c_str());
    export_scalars_grid(out.c_str(), N, N, N, 1);
    char path[512];

    int* flag_i = new int[fl->count];
    for (long i = 0; i < fl->count; i++) flag_i[i] = (int)(unsigned char)fl->flag[i];
    snprintf(path, sizeof(path), "%s/flag.txt", out.c_str());
    wi(path, flag_i, (int)fl->count);
    delete[] flag_i;

    snprintf(path, sizeof(path), "%s/phi.txt", out.c_str());
    wf(path, fl->phi, (int)fl->count);
    snprintf(path, sizeof(path), "%s/tag_matrix.txt", out.c_str());
    wi(path, fl->tag_matrix, (int)fl->count);
    snprintf(path, sizeof(path), "%s/disjoin_force.txt", out.c_str());
    wf(path, fl->disjoin_force, (int)fl->count);

    float sum_dj = 0.f, max_dj = 0.f;
    for (long i = 0; i < fl->count; i++) {
        sum_dj += fl->disjoin_force[i];
        if (fl->disjoin_force[i] > max_dj) max_dj = fl->disjoin_force[i];
    }
    snprintf(path, sizeof(path), "%s/sum_disjoin.txt", out.c_str());
    wf_scalar(path, sum_dj);
    snprintf(path, sizeof(path), "%s/max_disjoin.txt", out.c_str());
    wf_scalar(path, max_dj);
    printf("  sum(disjoin)=%.6g max=%.6g bubble_count=%d\n",
           sum_dj, max_dj, fl->bubble.bubble_count);

    delete s; delete fl;
}

// ============================================================================
// F2: no coalescence under disjoining (gate S2)
// ============================================================================

static void gen_foam_no_coalescence(const char* base) {
    printf("\n=== F2 foam_no_coalescence ===\n");
    const int N = 64;
    const int steps = 250;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    fill_fluid_background(fl, N, N, N);
    // r=6, centres ~15 apart → surface gap ~3 (matches foam_pair example)
    paint_soft_bubble(fl, N, N, N, 24.0f, 32.0f, 32.0f, 6.0f);
    paint_soft_bubble(fl, N, N, N, 39.0f, 32.0f, 32.0f, 6.0f);
    set_boundary_walls(fl, N, N, N);

    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());
    trans_gas_foam_to_host(s);
    int count0 = fl->bubble.bubble_count;
    printf("  initial bubble_count=%d\n", count0);

    int merge_seen = 0;
    for (int t = 0; t < steps; t++) {
        s->mlIterateCouplingGpu(t);
        if ((t + 1) % 50 == 0) {
            trans_gas_foam_to_host(s);
            printf("  step %d: bubbles=%d merge_flag=%d\n",
                   t + 1, fl->bubble.bubble_count, fl->merge_flag);
            if (fl->merge_flag != 0) merge_seen = 1;
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
    trans_gas_foam_to_host(s);

    // COM distance of first two tags
    double cx0 = 0, cy0 = 0, cz0 = 0, n0 = 0;
    double cx1 = 0, cy1 = 0, cz1 = 0, n1 = 0;
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) {
                long id = idx3(i, j, k, N, N);
                int tag = fl->tag_matrix[id];
                if (tag == 1) { cx0 += i; cy0 += j; cz0 += k; n0 += 1; }
                else if (tag == 2) { cx1 += i; cy1 += j; cz1 += k; n1 += 1; }
            }
    float com_dist = -1.f;
    if (n0 > 0 && n1 > 0) {
        cx0 /= n0; cy0 /= n0; cz0 /= n0;
        cx1 /= n1; cy1 /= n1; cz1 /= n1;
        float dx = (float)(cx0 - cx1), dy = (float)(cy0 - cy1), dz = (float)(cz0 - cz1);
        com_dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    std::string out = std::string(base) + "/foam_no_coalescence";
    make_dir(out.c_str());
    export_scalars_grid(out.c_str(), N, N, N, steps);
    char path[512];
    snprintf(path, sizeof(path), "%s/bubble_count.txt", out.c_str());
    wi_scalar(path, fl->bubble.bubble_count);
    snprintf(path, sizeof(path), "%s/merge_flag.txt", out.c_str());
    wi_scalar(path, fl->merge_flag != 0 ? fl->merge_flag : merge_seen);
    snprintf(path, sizeof(path), "%s/com_distance.txt", out.c_str());
    wf_scalar(path, com_dist);
    snprintf(path, sizeof(path), "%s/bubble_count_initial.txt", out.c_str());
    wi_scalar(path, count0);
    printf("  final bubbles=%d merge=%d com_dist=%.3f\n",
           fl->bubble.bubble_count, fl->merge_flag, com_dist);

    delete s; delete fl;
}

// ============================================================================
// F3: open-tank atmosphere rho / init_volume correction
// ============================================================================

static void gen_foam_atmosphere_open_tank(const char* base) {
    printf("\n=== F3 foam_atmosphere_open_tank ===\n");
    const int N = 32;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    fill_fluid_background(fl, N, N, N);
    // Bubble near open region z > nz-10
    paint_soft_bubble(fl, N, N, N, 16.0f, 16.0f, 26.0f, 5.0f);
    set_boundary_walls(fl, N, N, N);

    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());
    trans_gas_foam_to_host(s);

    // Force atmosphere condition: V > 1e6 and place tag in open zone
    int bc = fl->bubble.bubble_count;
    if (bc < 1) {
        printf("  WARN: no bubble tagged; skipping F3 body\n");
    } else {
        fl->bubble.volume[0] = 2000000.0;
        fl->bubble.rho[0] = 1.2;  // not yet atmosphere
        fl->bubble.init_volume[0] = 1500000.0;
        s->mlTransData2Gpu();

        launch_atmosphere_rho_update(
            s->lbm_dev_gpu, fl->param,
            s->mparam.N, s->mparam.l0p, s->mparam.roup, s->mparam.labma, s->mparam.u0p, 0);
        launch_atmosphere_volme_update(s->lbm_dev_gpu);
        trans_gas_foam_to_host(s);
    }

    std::string out = std::string(base) + "/foam_atmosphere_open_tank";
    make_dir(out.c_str());
    export_scalars_grid(out.c_str(), N, N, N, 1);
    char path[512];
    int n_dump = bc > 0 ? bc : 1;
    snprintf(path, sizeof(path), "%s/bubble_rho.txt", out.c_str());
    wd(path, fl->bubble.rho, n_dump);
    snprintf(path, sizeof(path), "%s/bubble_volume.txt", out.c_str());
    wd(path, fl->bubble.volume, n_dump);
    snprintf(path, sizeof(path), "%s/bubble_init_volume.txt", out.c_str());
    wd(path, fl->bubble.init_volume, n_dump);
    snprintf(path, sizeof(path), "%s/bubble_count.txt", out.c_str());
    wi_scalar(path, fl->bubble.bubble_count);
    if (bc > 0)
        printf("  rho[0]=%.6g volume[0]=%.6g init_volume[0]=%.6g\n",
               fl->bubble.rho[0], fl->bubble.volume[0], fl->bubble.init_volume[0]);

    delete s; delete fl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("=== HOME-FSLBM Phase 4 Gas/Foam Golden Generator ===\n");

    const char* base = "../../../golden_data";
    if (argc >= 2) base = argv[1];
    make_dir(base);

    // G1 host CMR vectors (no CUDA required for this scene alone, but linked with CUDA)
    gen_cmr_vectors(base);

    // G2–G4 gas field dumps
    gen_gas_henry_interface_step1(base);
    gen_gas_stream_collide_step10(base);
    gen_gas_volume_g_update(base);

    // F1–F3 foam
    gen_foam_disjoint_two_spheres(base);
    gen_foam_no_coalescence(base);
    gen_foam_atmosphere_open_tank(base);

    printf("\nDone. Copy gas_* / foam_* into:\n");
    printf("  wanphys/_src/fluid/fluid_grid/home_fslbm/tests/golden_data/\n");
    return 0;
}
