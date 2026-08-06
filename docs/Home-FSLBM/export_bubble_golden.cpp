// export_bubble_golden.cpp
// Phase 3 bubble / YACCLAB CCL golden-data generator for Warp regression tests.
//
// Build target: export_bubble_golden  (see CMakeLists.txt)
// Run from a CUDA-capable machine after building Home-FSLBM.
//
// Default output (same root as export_solver_golden; from out/build/... ,
// ../../../golden_data → docs/Home-FSLBM/golden_data/):
//   ../../../golden_data/bubble_*/{nx,input_matrix,label_matrix,...}.txt
//
// Then copy those directories into:
//   wanphys/_src/fluid/fluid_grid/home_fslbm/tests/golden_data/
//
// Scene ICs match wanphys tests/test_regression_bubble.py builders.
// Flat layout: idx = x + nx*(y + ny*z)  (x-fastest; same as mrFlow3D).

#include <cstdio>
#include <cmath>
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
    for (int i = 0; i < n; i++) fprintf(f, "%.15e\n", (double)d[i]);
    fclose(f);
    printf("  %s (%d floats)\n", fn, n);
}

static void wd(const char* fn, const double* d, int n) {
    FILE* f = fopen(fn, "w");
    if (!f) { printf("ERR open %s\n", fn); return; }
    for (int i = 0; i < n; i++) fprintf(f, "%.15e\n", d[i]);
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

static void wi_scalar(const char* fn, int v) {
    wi(fn, &v, 1);
}

static long idx3(int i, int j, int k, int nx, int ny) {
    return (long)k * ny * nx + (long)j * nx + i;
}

// ============================================================================
// GPU ↔ host bubble transfer (mlTransData2Host does not copy bubble fields)
// ============================================================================

static void trans_bubble_to_host(mrSolver3D* s) {
    mrFlow3D* host = s->lbmvec;
    mrFlow3D hdev;
    checkCudaErrors(cudaMemcpy(&hdev, s->lbm_dev_gpu, sizeof(mrFlow3D), cudaMemcpyDeviceToHost));

    long N = host->count;
    int maxb = (int)host->bubble.max_bubble_count;

    checkCudaErrors(cudaMemcpy(host->flag, hdev.flag, N * sizeof(MLLATTICENODE_SURFACE_FLAG), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->phi, hdev.phi, N * sizeof(REAL), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->mass, hdev.mass, N * sizeof(REAL), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->delta_phi, hdev.delta_phi, N * sizeof(REAL), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->fMom, hdev.fMom, 10 * N * sizeof(REAL), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(host->tag_matrix, hdev.tag_matrix, N * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->previous_tag, hdev.previous_tag, N * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->previous_merge_tag, hdev.previous_merge_tag, N * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->input_matrix, hdev.input_matrix, N * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->label_matrix, hdev.label_matrix, N * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->merge_detector, hdev.merge_detector, N * sizeof(bool), cudaMemcpyDeviceToHost));

    host->merge_flag = hdev.merge_flag;
    host->split_flag = hdev.split_flag;

    checkCudaErrors(cudaMemcpy(host->bubble.volume, hdev.bubble.volume, maxb * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->bubble.init_volume, hdev.bubble.init_volume, maxb * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->bubble.rho, hdev.bubble.rho, maxb * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->bubble.label_volume, hdev.bubble.label_volume, maxb * sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host->bubble.label_init_volume, hdev.bubble.label_init_volume, maxb * sizeof(double), cudaMemcpyDeviceToHost));
    host->bubble.bubble_count = hdev.bubble.bubble_count;
    host->bubble.label_num = hdev.bubble.label_num;
}

// ============================================================================
// Export bundle writers
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

static void export_ccl_pair(mrFlow3D* fl, const char* out_dir, int nx, int ny, int nz) {
    make_dir(out_dir);
    export_scalars_grid(out_dir, nx, ny, nz);
    long N = fl->count;
    char path[512];

    int* in_i = new int[N];
    for (long i = 0; i < N; i++) in_i[i] = (int)fl->input_matrix[i];
    snprintf(path, sizeof(path), "%s/input_matrix.txt", out_dir);
    wi(path, in_i, (int)N);
    delete[] in_i;

    snprintf(path, sizeof(path), "%s/label_matrix.txt", out_dir);
    wi(path, fl->label_matrix, (int)N);
}

// Crop a padded even grid (px,py,pz) down to logical (nx,ny,nz) for export.
static void export_ccl_pair_crop(mrFlow3D* fl, const char* out_dir,
                                 int nx, int ny, int nz, int px, int py, int pz) {
    make_dir(out_dir);
    export_scalars_grid(out_dir, nx, ny, nz);
    long n_out = (long)nx * ny * nz;
    char path[512];
    int* in_i = new int[n_out];
    int* lab = new int[n_out];
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                long src = idx3(i, j, k, px, py);
                long dst = idx3(i, j, k, nx, ny);
                in_i[dst] = (int)fl->input_matrix[src];
                lab[dst] = fl->label_matrix[src];
            }
    snprintf(path, sizeof(path), "%s/input_matrix.txt", out_dir);
    wi(path, in_i, (int)n_out);
    snprintf(path, sizeof(path), "%s/label_matrix.txt", out_dir);
    wi(path, lab, (int)n_out);
    delete[] in_i;
    delete[] lab;
}

static void export_bubble_state(mrFlow3D* fl, const char* out_dir, int nx, int ny, int nz, int steps) {
    make_dir(out_dir);
    export_scalars_grid(out_dir, nx, ny, nz, steps);
    long N = fl->count;
    char path[512];

    int* flag_i = new int[N];
    for (long i = 0; i < N; i++) flag_i[i] = (int)(unsigned char)fl->flag[i];
    snprintf(path, sizeof(path), "%s/flag.txt", out_dir);
    wi(path, flag_i, (int)N);
    delete[] flag_i;

    snprintf(path, sizeof(path), "%s/phi.txt", out_dir);
    wf(path, fl->phi, (int)N);

    snprintf(path, sizeof(path), "%s/delta_phi.txt", out_dir);
    wf(path, fl->delta_phi, (int)N);

    snprintf(path, sizeof(path), "%s/tag_matrix.txt", out_dir);
    wi(path, fl->tag_matrix, (int)N);

    snprintf(path, sizeof(path), "%s/previous_tag.txt", out_dir);
    wi(path, fl->previous_tag, (int)N);

    snprintf(path, sizeof(path), "%s/label_matrix.txt", out_dir);
    wi(path, fl->label_matrix, (int)N);

    int* in_i = new int[N];
    for (long i = 0; i < N; i++) in_i[i] = (int)fl->input_matrix[i];
    snprintf(path, sizeof(path), "%s/input_matrix.txt", out_dir);
    wi(path, in_i, (int)N);
    delete[] in_i;

    int* md = new int[N];
    for (long i = 0; i < N; i++) md[i] = fl->merge_detector[i] ? 1 : 0;
    snprintf(path, sizeof(path), "%s/merge_detector.txt", out_dir);
    wi(path, md, (int)N);
    delete[] md;

    snprintf(path, sizeof(path), "%s/merge_flag.txt", out_dir);
    wi_scalar(path, fl->merge_flag);
    snprintf(path, sizeof(path), "%s/split_flag.txt", out_dir);
    wi_scalar(path, fl->split_flag);

    int bc = fl->bubble.bubble_count;
    if (bc < 0) bc = 0;
    snprintf(path, sizeof(path), "%s/bubble_count.txt", out_dir);
    wi_scalar(path, bc);
    snprintf(path, sizeof(path), "%s/label_num.txt", out_dir);
    wi_scalar(path, fl->bubble.label_num);

    // Dump active bubbles (at least 1 line); pad to max(bc,1)
    int n_dump = bc > 0 ? bc : 1;
    snprintf(path, sizeof(path), "%s/bubble_volume.txt", out_dir);
    wd(path, fl->bubble.volume, n_dump);
    snprintf(path, sizeof(path), "%s/bubble_init_volume.txt", out_dir);
    wd(path, fl->bubble.init_volume, n_dump);
    snprintf(path, sizeof(path), "%s/bubble_rho.txt", out_dir);
    wd(path, fl->bubble.rho, n_dump);
}

// ============================================================================
// Scene builders (must match Warp test_regression_bubble.py)
// ============================================================================

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
        fl->tag_matrix[idx] = -1;
        fl->previous_tag[idx] = -1;
        fl->previous_merge_tag[idx] = -1;
        fl->input_matrix[idx] = 0;
        fl->label_matrix[idx] = 0;
        fl->merge_detector[idx] = false;
        fl->delta_phi[idx] = 0.0f;
        fl->massex[idx] = 0.0f;
        fl->islet[idx] = 0;
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
                    fl->input_matrix[id] = 0;
                }
}

static void paint_input_sphere(unsigned char* img, int nx, int ny, int nz,
                               int cx, int cy, int cz, int radius) {
    int r2 = radius * radius;
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                int d2 = (i - cx) * (i - cx) + (j - cy) * (j - cy) + (k - cz) * (k - cz);
                if (d2 <= r2) img[idx3(i, j, k, nx, ny)] = 255;
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

static void paint_hard_bubble(mrFlow3D* fl, int nx, int ny, int nz,
                              int cx, int cy, int cz, int radius) {
    int r2 = radius * radius;
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                int d2 = (i - cx) * (i - cx) + (j - cy) * (j - cy) + (k - cz) * (k - cz);
                if (d2 <= r2) {
                    long id = idx3(i, j, k, nx, ny);
                    fl->flag[id] = TYPE_I;
                    fl->phi[id] = 0.0f;
                    fl->mass[id] = 0.0f;
                }
            }
}

static mrSolver3D* build_solver(mrFlow3D** fl_out, int nx, int ny, int nz, REAL gz) {
    REAL Np = 1, l0p = 1, roup = 1, labma = 1, u0p = 1;
    // vis = 1e-4 → kernel omega ≈ 1/(3*1e-4+0.5)
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

// ============================================================================
// A. Static CCL scenes
// ============================================================================

static void run_ccl_from_host_input(mrSolver3D* s, mrFlow3D* fl, int nx, int ny, int nz) {
    s->mlTransData2Gpu();
    connectedComponentLabeling(s->lbm_dev_gpu, nx, ny, nz);
    checkCudaErrors(cudaDeviceSynchronize());
    trans_bubble_to_host(s);
}

static void gen_ccl_3spheres(const char* base) {
    printf("\n=== bubble_ccl_3spheres_r3 ===\n");
    const int N = 16;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    clear_moments(fl);
    memset(fl->input_matrix, 0, (size_t)fl->count);
    paint_input_sphere(fl->input_matrix, N, N, N, 4, 4, 4, 3);
    paint_input_sphere(fl->input_matrix, N, N, N, 12, 4, 4, 3);
    paint_input_sphere(fl->input_matrix, N, N, N, 8, 12, 12, 3);
    run_ccl_from_host_input(s, fl, N, N, N);
    std::string out = std::string(base) + "/bubble_ccl_3spheres_r3";
    export_ccl_pair(fl, out.c_str(), N, N, N);
    delete s; delete fl;
}

static void gen_ccl_touching_face(const char* base) {
    printf("\n=== bubble_ccl_touching_face ===\n");
    const int N = 16;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    clear_moments(fl);
    memset(fl->input_matrix, 0, (size_t)fl->count);
    paint_input_sphere(fl->input_matrix, N, N, N, 6, 8, 8, 3);
    paint_input_sphere(fl->input_matrix, N, N, N, 10, 8, 8, 3);
    run_ccl_from_host_input(s, fl, N, N, N);
    export_ccl_pair(fl, (std::string(base) + "/bubble_ccl_touching_face").c_str(), N, N, N);
    delete s; delete fl;
}

static void gen_ccl_touching_edge(const char* base) {
    printf("\n=== bubble_ccl_touching_edge ===\n");
    const int N = 16;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    clear_moments(fl);
    memset(fl->input_matrix, 0, (size_t)fl->count);
    for (int k = 4; k < 7; k++)
        for (int j = 4; j < 7; j++)
            for (int i = 4; i < 7; i++)
                fl->input_matrix[idx3(i, j, k, N, N)] = 255;
    for (int k = 4; k < 7; k++)
        for (int j = 6; j < 9; j++)
            for (int i = 6; i < 9; i++)
                fl->input_matrix[idx3(i, j, k, N, N)] = 255;
    run_ccl_from_host_input(s, fl, N, N, N);
    export_ccl_pair(fl, (std::string(base) + "/bubble_ccl_touching_edge").c_str(), N, N, N);
    delete s; delete fl;
}

static void gen_ccl_diagonal_gap(const char* base) {
    printf("\n=== bubble_ccl_diagonal_gap ===\n");
    const int N = 16;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    clear_moments(fl);
    memset(fl->input_matrix, 0, (size_t)fl->count);
    paint_input_sphere(fl->input_matrix, N, N, N, 5, 8, 8, 2);
    paint_input_sphere(fl->input_matrix, N, N, N, 11, 8, 8, 2);
    run_ccl_from_host_input(s, fl, N, N, N);
    export_ccl_pair(fl, (std::string(base) + "/bubble_ccl_diagonal_gap").c_str(), N, N, N);
    delete s; delete fl;
}

static void gen_ccl_odd_dims(const char* base) {
    // Logical scene is 17³ (odd). Reference YACCLAB 2×2×2 kernels misalign on
    // odd pitch (ushort/ulonglong loads), so run CCL on an even 18³ pad with
    // background halo and crop the golden back to 17³ for Warp.
    printf("\n=== bubble_ccl_odd_dims ===\n");
    const int N = 17;
    const int P = 18;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, P, P, P, 0.0f);
    clear_moments(fl);
    memset(fl->input_matrix, 0, (size_t)fl->count);
    paint_input_sphere(fl->input_matrix, P, P, P, 4, 4, 4, 2);
    paint_input_sphere(fl->input_matrix, P, P, P, 12, 4, 4, 2);
    paint_input_sphere(fl->input_matrix, P, P, P, 8, 12, 12, 2);
    run_ccl_from_host_input(s, fl, P, P, P);
    export_ccl_pair_crop(fl, (std::string(base) + "/bubble_ccl_odd_dims").c_str(),
                         N, N, N, P, P, P);
    delete s; delete fl;
}

static void gen_ccl_near_wall(const char* base) {
    printf("\n=== bubble_ccl_near_wall ===\n");
    const int N = 16;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    clear_moments(fl);
    memset(fl->input_matrix, 0, (size_t)fl->count);
    paint_input_sphere(fl->input_matrix, N, N, N, 3, 8, 8, 3);
    run_ccl_from_host_input(s, fl, N, N, N);
    export_ccl_pair(fl, (std::string(base) + "/bubble_ccl_near_wall").c_str(), N, N, N);
    delete s; delete fl;
}

static void gen_ccl_many_small(const char* base) {
    printf("\n=== bubble_ccl_many_small ===\n");
    const int N = 32;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    clear_moments(fl);
    memset(fl->input_matrix, 0, (size_t)fl->count);
    const int centres[8][3] = {
        {4, 4, 4}, {4, 4, 28}, {4, 28, 4}, {4, 28, 28},
        {28, 4, 4}, {28, 4, 28}, {28, 28, 4}, {28, 28, 28},
    };
    for (int c = 0; c < 8; c++)
        paint_input_sphere(fl->input_matrix, N, N, N, centres[c][0], centres[c][1], centres[c][2], 2);
    run_ccl_from_host_input(s, fl, N, N, N);
    export_ccl_pair(fl, (std::string(base) + "/bubble_ccl_many_small").c_str(), N, N, N);
    delete s; delete fl;
}

// ============================================================================
// B. InitBubble scenes
// ============================================================================

static void gen_init_two_bubbles(const char* base) {
    printf("\n=== bubble_init_two_bubbles ===\n");
    const int N = 32;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    fill_fluid_background(fl, N, N, N);
    paint_hard_bubble(fl, N, N, N, 10, 16, 16, 4);
    paint_hard_bubble(fl, N, N, N, 22, 16, 16, 4);
    set_boundary_walls(fl, N, N, N);
    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());
    trans_bubble_to_host(s);
    export_bubble_state(fl, (std::string(base) + "/bubble_init_two_bubbles").c_str(), N, N, N, 0);
    delete s; delete fl;
}

static void gen_init_interface_shell(const char* base) {
    printf("\n=== bubble_init_interface_shell ===\n");
    const int N = 32;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    fill_fluid_background(fl, N, N, N);
    int cx = 16, cy = 16, cz = 16;
    int r_in2 = 4 * 4, r_out2 = 6 * 6;
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) {
                int d2 = (i - cx) * (i - cx) + (j - cy) * (j - cy) + (k - cz) * (k - cz);
                long id = idx3(i, j, k, N, N);
                if (d2 <= r_in2) {
                    fl->flag[id] = TYPE_G;
                    fl->phi[id] = 0.0f;
                    fl->mass[id] = 0.0f;
                } else if (d2 <= r_out2) {
                    fl->flag[id] = TYPE_I;
                    fl->phi[id] = 0.5f;
                    fl->mass[id] = 0.5f;
                }
            }
    set_boundary_walls(fl, N, N, N);
    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());
    trans_bubble_to_host(s);
    export_bubble_state(fl, (std::string(base) + "/bubble_init_interface_shell").c_str(), N, N, N, 0);
    delete s; delete fl;
}

// ============================================================================
// C. Coupling short runs
// ============================================================================

static void run_coupling(mrSolver3D* s, int steps) {
    for (int t = 0; t < steps; t++)
        s->mlIterateCouplingGpu(t);
    checkCudaErrors(cudaDeviceSynchronize());
    trans_bubble_to_host(s);
}

static void gen_coupling_volume_delta_phi(const char* base) {
    printf("\n=== bubble_coupling_volume_delta_phi ===\n");
    const int N = 16;
    const int steps = 5;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    fill_fluid_background(fl, N, N, N);
    paint_hard_bubble(fl, N, N, N, 8, 8, 8, 4);
    set_boundary_walls(fl, N, N, N);
    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());
    trans_bubble_to_host(s);

    // Seed synthetic delta_phi on tagged cells (Warp test seeds this BEFORE steps)
    std::vector<float> delta0((size_t)fl->count, 0.0f);
    for (long id = 0; id < fl->count; id++) {
        if (fl->tag_matrix[id] > 0)
            delta0[(size_t)id] = -0.01f;
        fl->delta_phi[id] = delta0[(size_t)id];
    }
    s->mlTransData2Gpu();
    run_coupling(s, steps);
    std::string out = std::string(base) + "/bubble_coupling_volume_delta_phi";
    export_bubble_state(fl, out.c_str(), N, N, N, steps);
    // Overwrite delta_phi.txt with the *initial* seed used by the Warp test
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/delta_phi.txt", out.c_str());
        wf(path, delta0.data(), (int)delta0.size());
    }
    delete s; delete fl;
}

static void gen_two_bubbles_merge(const char* base) {
    printf("\n=== bubble_two_bubbles_merge ===\n");
    const int N = 32;
    const int steps = 80;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    fill_fluid_background(fl, N, N, N);
    paint_hard_bubble(fl, N, N, N, 12, 16, 16, 4);
    paint_hard_bubble(fl, N, N, N, 18, 16, 16, 4);
    set_boundary_walls(fl, N, N, N);
    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());
    run_coupling(s, steps);
    export_bubble_state(fl, (std::string(base) + "/bubble_two_bubbles_merge").c_str(), N, N, N, steps);
    delete s; delete fl;
}

static void gen_split_via_surface(const char* base) {
    printf("\n=== bubble_split_via_surface ===\n");
    const int N = 32;
    const int steps = 60;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, 0.0f);
    fill_fluid_background(fl, N, N, N);
    paint_hard_bubble(fl, N, N, N, 16, 16, 16, 6);
    set_boundary_walls(fl, N, N, N);
    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());
    run_coupling(s, steps);
    export_bubble_state(fl, (std::string(base) + "/bubble_split_via_surface").c_str(), N, N, N, steps);
    delete s; delete fl;
}

static void gen_translate_no_merge(const char* base) {
    printf("\n=== bubble_translate_no_merge ===\n");
    const int N = 32;
    const int steps = 20;
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_solver(&fl, N, N, N, -1.0e-4f);
    fill_fluid_background(fl, N, N, N);
    paint_hard_bubble(fl, N, N, N, 10, 16, 16, 5);
    set_boundary_walls(fl, N, N, N);
    // Body force gz already set via Create; also stamp forcez for consistency
    for (long id = 0; id < fl->count; id++)
        fl->forcez[id] = -1.0e-4f;
    s->mlTransData2Gpu();
    InitBubble(s->lbm_dev_gpu, fl->param);
    checkCudaErrors(cudaDeviceSynchronize());
    run_coupling(s, steps);
    export_bubble_state(fl, (std::string(base) + "/bubble_translate_no_merge").c_str(), N, N, N, steps);
    delete s; delete fl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("=== HOME-FSLBM Phase 3 Bubble Golden Generator ===\n");

    // Default: docs/Home-FSLBM/golden_data/ when run from out/build/...
    const char* base = "../../../golden_data";
    if (argc >= 2) base = argv[1];
    make_dir(base);

    // A. CCL
    gen_ccl_3spheres(base);
    gen_ccl_touching_face(base);
    gen_ccl_touching_edge(base);
    gen_ccl_diagonal_gap(base);
    gen_ccl_odd_dims(base);
    gen_ccl_near_wall(base);
    gen_ccl_many_small(base);

    // B. InitBubble
    gen_init_two_bubbles(base);
    gen_init_interface_shell(base);

    // C. Coupling
    gen_coupling_volume_delta_phi(base);
    gen_two_bubbles_merge(base);
    gen_split_via_surface(base);
    gen_translate_no_merge(base);

    printf("\nDone. Copy '%s/bubble_*' into:\n", base);
    printf("  wanphys/_src/fluid/fluid_grid/home_fslbm/tests/golden_data/\n");
    return 0;
}
