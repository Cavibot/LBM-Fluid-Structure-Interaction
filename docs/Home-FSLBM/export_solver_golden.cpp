// export_solver_golden.cpp
// Extended: Phase 1 (shear/bounce/turb) + Phase 2 (10 surface scenes) + curvature ref
#include <cstdio>
#include <cmath>
#include <cstring>
#include "mrSolver3D.h"
#include "mrFlow3D.h"
#include "mrLbmSolverGpu3D.h"
#include "mlLbmCommon.h"
using namespace Mfree;

// ============================================================================
// Helpers
// ============================================================================

static void wf(const char* fn, const float* d, int n) {
    FILE* f = fopen(fn, "w");
    if (!f) { printf("ERR %s\n", fn); return; }
    for (int i = 0; i < n; i++) fprintf(f, "%.15e\n", d[i]);
    fclose(f);
    printf("  %s (%d)\n", fn, n);
}

static void wi(const char* fn, const int* d, int n) {
    FILE* f = fopen(fn, "w");
    if (!f) { printf("ERR %s\n", fn); return; }
    for (int i = 0; i < n; i++) fprintf(f, "%d\n", d[i]);
    fclose(f);
    printf("  %s (%d)\n", fn, n);
}

static void make_dir(const char* path) {
    // Simple mkdir for Windows; ignore errors if exists
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir \"%s\" 2>nul", path);
    system(cmd);
}

// ============================================================================
// Phase 1 generators — mlInit() replaced with manual init to avoid OpenMP
// "User Error 1001" from reference code's mlInit3D.h in MSVC Debug builds.
// ============================================================================

// Manual equivalent of mlInit().
// mlInitBoundaryCpu sets 6 faces to TYPE_S; the tank zone (z>0, z<=Nz/3-10, …)
// is empty for N>=32, so all interior stays TYPE_G from Create().
// mlInitFlowVarCpu then sets rho=1, mass=0, phi=0, forcez=-1e-5 for ALL cells.
static void manual_mlinit(mrFlow3D* fl, int nx, int ny, int nz, REAL gz = -1e-5f) {
    long count = fl->count;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                long idx = k * ny * nx + j * nx + i;
                // Boundary walls: exactly 6 faces (mlInitBoundaryCpu)
                bool is_wall = (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1);
                fl->flag[idx] = is_wall ? TYPE_S : TYPE_G;
                // mlInitFlowVarCpu: mass=0, phi=0 for all cells
                fl->mass[idx] = 0.0f;
                fl->phi[idx] = 0.0f;
                // mlInitFlowVarCpu: rho=1 for all cells, u=0, stress=0
                fl->fMom[idx] = 1.0f;
                fl->fMomPost[idx] = 1.0f;
                for (int m = 1; m < 10; m++) {
                    fl->fMom[m * count + idx] = 0.0f;
                    fl->fMomPost[m * count + idx] = 0.0f;
                }
                // mlInitFlowVarCpu: forcez=-1e-5 for all cells
                fl->forcex[idx] = 0.0f;
                fl->forcey[idx] = 0.0f;
                fl->forcez[idx] = gz;
                fl->c_value[idx] = 0.0f;
                fl->tag_matrix[idx] = -1;
                fl->massex[idx] = 0.0f;
                fl->islet[idx] = 0;
            }
        }
    }
}

static void gen_shear() {
    printf("\n=== Shear Decay (32^3,100 steps) ===\n");
    int nx = 32, ny = 32, nz = 32;
    REAL Np = 1, l0p = 1, roup = 1, labma = 1, u0p = 1;
    mrFlow3D* fl = new mrFlow3D();
    fl->Create(0, 0, 0, nx, ny, nz, 1, (REAL)nx, (REAL)ny, (REAL)nz, 1.0f / 6.0f, 0);
    fl->BubbleBufferInit(65536);
    mrSolver3D s; s.AttachLbmHost(fl); mrFlow3D* fd = 0; s.AttachLbmDevice(fd);
    MLMappingParam mp(u0p, labma, l0p, Np, roup); s.AttachMapping(mp);
    manual_mlinit(fl, nx, ny, nz);
    long N = fl->count; REAL A = 0.01f;
    for (int k = 0; k < nz; k++) {
        REAL z = (REAL)k;
        REAL ux = A * sinf(2 * 3.14159265f * z / (REAL)nz);
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                long idx = k * ny * nx + j * nx + i;
                fl->fMom[1 * N + idx] = ux;
                fl->fMomPost[1 * N + idx] = ux;
            }
    }
    s.mlTransData2Gpu();
    for (int step = 0; step < 100; step++)
        mrSolver3DGpu(s.lbm_dev_gpu, fl->param, s.mparam.N, s.mparam.l0p, s.mparam.roup, s.mparam.labma, s.mparam.u0p, step);
    s.mlTransData2Host();
    int im = nx / 2, jm = ny / 2; float* pr = new float[nz];
    for (int k = 0; k < nz; k++) { long idx = k * ny * nx + jm * nx + im; pr[k] = fl->fMom[1 * N + idx] / fl->fMom[0 * N + idx]; }
    wf("../../../golden_data/stream_collide_shear_decay_step100.txt", pr, nz);
    delete[] pr; delete fl;
}

static void gen_bounce() {
    printf("\n=== Solid Bounce-Back (32^3,500 steps) ===\n");
    int nx = 32, ny = 32, nz = 32;
    REAL Np = 1, l0p = 1, roup = 1, labma = 1, u0p = 1;
    mrFlow3D* fl = new mrFlow3D();
    fl->Create(0, 0, 0, nx, ny, nz, 1, (REAL)nx, (REAL)ny, (REAL)nz, 1.0f / 6.0f, -0.001f);
    fl->BubbleBufferInit(65536);
    mrSolver3D s; s.AttachLbmHost(fl); mrFlow3D* fd = 0; s.AttachLbmDevice(fd);
    MLMappingParam mp(u0p, labma, l0p, Np, roup); s.AttachMapping(mp);
    manual_mlinit(fl, nx, ny, nz, -0.001f);
    long N = fl->count;
    for (long idx = 0; idx < N; idx++) { fl->forcez[idx] = -0.001f; }
    s.mlTransData2Gpu();
    for (int step = 0; step < 500; step++)
        mrSolver3DGpu(s.lbm_dev_gpu, fl->param, s.mparam.N, s.mparam.l0p, s.mparam.roup, s.mparam.labma, s.mparam.u0p, step);
    s.mlTransData2Host();
    int im = nx / 2, jm = ny / 2; float* pr = new float[nz * 4];
    for (int k = 0; k < nz; k++) {
        long idx = im * ny * nz + jm * nz + k;
        REAL rho = fl->fMom[0 * N + idx];
        pr[4 * k] = rho; pr[4 * k + 1] = fl->fMom[1 * N + idx] / rho;
        pr[4 * k + 2] = fl->fMom[2 * N + idx] / rho; pr[4 * k + 3] = fl->fMom[3 * N + idx] / rho;
    }
    wf("../../../golden_data/stream_collide_bounce_back_step500.txt", pr, nz * 4);
    delete[] pr; delete fl;
}

static void gen_turb() {
    printf("\n=== Turbulence Omega (16^3,1 step) ===\n");
    int nx = 16, ny = 16, nz = 16;
    REAL Np = 1, l0p = 1, roup = 1, labma = 1, u0p = 1;
    mrFlow3D* fl = new mrFlow3D();
    fl->Create(0, 0, 0, nx, ny, nz, 1, (REAL)nx, (REAL)ny, (REAL)nz, 1.0f / 6.0f, 0);
    fl->BubbleBufferInit(65536);
    mrSolver3D s; s.AttachLbmHost(fl); mrFlow3D* fd = 0; s.AttachLbmDevice(fd);
    MLMappingParam mp(u0p, labma, l0p, Np, roup); s.AttachMapping(mp);
    manual_mlinit(fl, nx, ny, nz);
    long N = fl->count;
    int cx = nx / 2, cy = ny / 2, cz = nz / 2;
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                long idx = i * ny * nz + j * nz + k;
                if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1) {
                    int d2 = (i - cx) * (i - cx) + (j - cy) * (j - cy) + (k - cz) * (k - cz);
                    fl->flag[idx] = (d2 <= 4) ? TYPE_I : TYPE_F;
                    fl->tag_matrix[idx] = (d2 <= 4) ? 1 : -1;
                }
            }
    fl->bubble.volume[0] = 1000; fl->bubble.init_volume[0] = 1000;
    fl->bubble.rho[0] = 1; fl->bubble.bubble_count = 1;
    s.mlTransData2Gpu();
    mrSolver3DGpu(s.lbm_dev_gpu, fl->param, s.mparam.N, s.mparam.l0p, s.mparam.roup, s.mparam.labma, s.mparam.u0p, 0);
    s.mlTransData2Host();
    float* d = new float[10 * N];
    for (long idx = 0; idx < N; idx++)
        for (int m = 0; m < 10; m++) d[m * N + idx] = fl->fMom[m * N + idx];
    wf("../../../golden_data/turbulence_omega_step1.txt", d, 10 * N);
    delete[] d; delete fl;
}

// ============================================================================
// Phase 2: surface scene generators
// ============================================================================

// Build solver + flow — skip mlInit() to avoid OpenMP "User Error 1001"
// in MSVC Debug.  Scene fields set manually by setup_*() helpers.
// mlTransData2Gpu() in run_and_export() handles the GPU upload.
static mrSolver3D* build_surface_solver(mrFlow3D** fl_out, int Nx, int Ny, int Nz,
                                         REAL vis, REAL gz) {
    REAL Np = 1, l0p = 1, roup = 1, labma = 1, u0p = 1;
    mrFlow3D* fl = new mrFlow3D();
    fl->Create(0, 0, 0, Nx, Ny, Nz, 1, (REAL)Nx, (REAL)Ny, (REAL)Nz, vis, gz);
    fl->BubbleBufferInit(65536);
    mrSolver3D* s = new mrSolver3D();
    s->AttachLbmHost(fl);
    mrFlow3D* fd = 0; s->AttachLbmDevice(fd);
    MLMappingParam mp(u0p, labma, l0p, Np, roup);
    s->AttachMapping(mp);
    *fl_out = fl;
    return s;
}

// Set up a droplet: sphere of radius R at centre.
static void setup_droplet(mrFlow3D* fl, int Nx, int Ny, int Nz, double R,
                          double cx, double cy, double cz, double delta = 1.5) {
    long count = fl->count;
    // init fMom: rho=1, u=0 everywhere
    for (long idx = 0; idx < count; idx++) {
        fl->fMom[idx] = 1.0f;              // rho
        fl->fMomPost[idx] = 1.0f;
        for (int m = 1; m < 10; m++) {
            fl->fMom[m * count + idx] = 0.0f;
            fl->fMomPost[m * count + idx] = 0.0f;
        }
        fl->forcex[idx] = 0.0f; fl->forcey[idx] = 0.0f; fl->forcez[idx] = 0.0f;
        fl->c_value[idx] = 0.0f;
        fl->tag_matrix[idx] = -1;
    }
    // phi + flag
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                long idx = k * Ny * Nx + j * Nx + i;
                double dx = i - cx, dy = j - cy, dz = k - cz;
                double dist = sqrt(dx * dx + dy * dy + dz * dz);
                double phi_raw = (R - dist) / delta + 0.5;
                double phi_val = phi_raw < 0.0 ? 0.0 : (phi_raw > 1.0 ? 1.0 : phi_raw);
                fl->phi[idx] = (REAL)phi_val;
                if (phi_val >= 1.0 - 1e-6) {
                    fl->flag[idx] = TYPE_F;
                    fl->mass[idx] = 1.0f;
                } else if (phi_val <= 1e-6) {
                    fl->flag[idx] = TYPE_G;
                    fl->mass[idx] = 0.0f;
                } else {
                    fl->flag[idx] = TYPE_I;
                    fl->mass[idx] = (REAL)phi_val;  // mass = phi * rho (rho=1)
                }
                fl->massex[idx] = 0.0f;
            }
        }
    }
}

// Set up a flat interface: z < z_split = F, z > z_split = G
static void setup_flat_interface(mrFlow3D* fl, int Nx, int Ny, int Nz,
                                 int z_split, double delta = 1.5) {
    long count = fl->count;
    for (long idx = 0; idx < count; idx++) {
        fl->fMom[idx] = 1.0f;
        fl->fMomPost[idx] = 1.0f;
        for (int m = 1; m < 10; m++) {
            fl->fMom[m * count + idx] = 0.0f;
            fl->fMomPost[m * count + idx] = 0.0f;
        }
        fl->forcex[idx] = 0.0f; fl->forcey[idx] = 0.0f; fl->forcez[idx] = 0.0f;
        fl->c_value[idx] = 0.0f;
        fl->tag_matrix[idx] = -1;
    }
    for (int k = 0; k < Nz; k++) {
        double phi_raw = (z_split - k) / delta + 0.5;
        double phi_val = phi_raw < 0.0 ? 0.0 : (phi_raw > 1.0 ? 1.0 : phi_raw);
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                long idx = k * Ny * Nx + j * Nx + i;
                fl->phi[idx] = (REAL)phi_val;
                if (phi_val >= 1.0 - 1e-6) {
                    fl->flag[idx] = TYPE_F; fl->mass[idx] = 1.0f;
                } else if (phi_val <= 1e-6) {
                    fl->flag[idx] = TYPE_G; fl->mass[idx] = 0.0f;
                } else {
                    fl->flag[idx] = TYPE_I; fl->mass[idx] = (REAL)phi_val;
                }
                fl->massex[idx] = 0.0f;
            }
        }
    }
}

// Set up two droplets at (cx1,cy1,cz1) and (cx2,cy2,cz2)
static void setup_near_droplets(mrFlow3D* fl, int Nx, int Ny, int Nz, double R,
                                double cx1, double cy1, double cz1,
                                double cx2, double cy2, double cz2,
                                double delta = 1.5) {
    long count = fl->count;
    for (long idx = 0; idx < count; idx++) {
        fl->fMom[idx] = 1.0f; fl->fMomPost[idx] = 1.0f;
        for (int m = 1; m < 10; m++) {
            fl->fMom[m * count + idx] = 0.0f; fl->fMomPost[m * count + idx] = 0.0f;
        }
        fl->forcex[idx] = 0.0f; fl->forcey[idx] = 0.0f; fl->forcez[idx] = 0.0f;
        fl->c_value[idx] = 0.0f;
        fl->tag_matrix[idx] = -1;
    }
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                long idx = k * Ny * Nx + j * Nx + i;
                double d1 = sqrt((i - cx1) * (i - cx1) + (j - cy1) * (j - cy1) + (k - cz1) * (k - cz1));
                double d2 = sqrt((i - cx2) * (i - cx2) + (j - cy2) * (j - cy2) + (k - cz2) * (k - cz2));
                double phi1 = (R - d1) / delta + 0.5;
                double phi2 = (R - d2) / delta + 0.5;
                double phi_raw = phi1 > phi2 ? phi1 : phi2;  // max of two VOF fields
                double phi_val = phi_raw < 0.0 ? 0.0 : (phi_raw > 1.0 ? 1.0 : phi_raw);
                fl->phi[idx] = (REAL)phi_val;
                if (phi_val >= 1.0 - 1e-6) {
                    fl->flag[idx] = TYPE_F; fl->mass[idx] = 1.0f;
                } else if (phi_val <= 1e-6) {
                    fl->flag[idx] = TYPE_G; fl->mass[idx] = 0.0f;
                } else {
                    fl->flag[idx] = TYPE_I; fl->mass[idx] = (REAL)phi_val;
                }
                fl->massex[idx] = 0.0f;
            }
        }
    }
}

// Set the 6 boundary faces to TYPE_S (solid wall).
// This prevents out-of-bounds memory access in stream_collide_bvh, which
// lacks bounds-checking on neighbour indices for non-solid domain edges.
// Must be called AFTER the setup_* functions (which overwrite all flags)
// and BEFORE mlTransData2Gpu / run_and_export.
static void set_boundary_walls(mrFlow3D* fl, int Nx, int Ny, int Nz) {
    long count = fl->count;
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1) {
                    long idx = k * Ny * Nx + j * Nx + i;
                    fl->flag[idx] = TYPE_S;
                    fl->phi[idx] = 0.0f;
                    fl->mass[idx] = 0.0f;
                    fl->massex[idx] = 0.0f;
                    // f_mom stays at equilibrium (rho=1, u=0, S=0) — walls
                    // skip computation so the moment values are unused.
                }
            }
        }
    }
}

// Generic: iterate steps, trans back, export 5 fields
static void export_surface_fields(mrFlow3D* fl, const char* out_dir) {
    long N = fl->count;
    char path[512];
    make_dir(out_dir);

    // f_mom (post-step): 10*N floats (interleaved layout: f_mom[m*N + idx])
    // NOTE: use fMom, not fMomPost — mlTransData2Host only copies fMom back.
    // After step2Kernel on GPU, fMom == fMomPost, so fMom has the post-step state.
    snprintf(path, sizeof(path), "%s/f_mom_post.txt", out_dir);
    wf(path, fl->fMom, 10 * (int)N);

    // flag: N ints (stored as unsigned char, cast to int)
    int* flag_int = new int[N];
    for (long i = 0; i < N; i++) flag_int[i] = (int)(unsigned char)fl->flag[i];
    snprintf(path, sizeof(path), "%s/flag.txt", out_dir);
    wi(path, flag_int, (int)N);
    delete[] flag_int;

    // mass: N float
    snprintf(path, sizeof(path), "%s/mass.txt", out_dir);
    wf(path, fl->mass, (int)N);

    // phi: N float
    snprintf(path, sizeof(path), "%s/phi.txt", out_dir);
    wf(path, fl->phi, (int)N);

    // tag_matrix: N int
    snprintf(path, sizeof(path), "%s/tag_matrix.txt", out_dir);
    wi(path, fl->tag_matrix, (int)N);
}


// ---- Diagnostic: step-by-step export for droplet_r8 ----
static void run_and_export_steps(mrSolver3D* s, mrFlow3D* fl,
                                  const int* export_at, int num_exports,
                                  const char* out_dir) {
    s->mlTransData2Gpu();
    int max_step = export_at[num_exports - 1];
    int next_export = 0;
    for (int step = 0; step <= max_step; step++) {
        if (step > 0) {
            mrSolver3DGpu(s->lbm_dev_gpu, fl->param,
                          s->mparam.N, s->mparam.l0p, s->mparam.roup,
                          s->mparam.labma, s->mparam.u0p, step - 1);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("  *** CUDA error after step %d: %s ***\n",
                       step, cudaGetErrorString(err));
                return;
            }
        }
        if (step == export_at[next_export]) {
            s->mlTransData2Host();
            char subdir[512];
            snprintf(subdir, sizeof(subdir), "%s/step_%d", out_dir, step);
            export_surface_fields(fl, subdir);
            s->mlTransData2Gpu();  // re-upload for next step
            next_export++;
        }
    }
}

static void gen_droplet_r8_diag() {
    const int N = 32;
    const double R = 8.0;
    const char* out_dir = "../../../golden_data/droplet_r8_diag";

    printf("\n=== Droplet R=8 DIAGNOSTIC (32^3, steps 0/1/2) ===\n");
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_surface_solver(&fl, N, N, N, 1.0 / 6.0, 0.0);
    double c = N / 2.0;
    setup_droplet(fl, N, N, N, R, c, c, c);
    set_boundary_walls(fl, N, N, N);

    int export_at[] = {0, 1, 2};
    run_and_export_steps(s, fl, export_at, 3, out_dir);
    delete s; delete fl;
}

static void run_and_export(mrSolver3D* s, mrFlow3D* fl, int steps,
                           const char* out_dir) {
    // Re-upload if needed
    s->mlTransData2Gpu();
    for (int step = 0; step < steps; step++) {
        mrSolver3DGpu(s->lbm_dev_gpu, fl->param,
                      s->mparam.N, s->mparam.l0p, s->mparam.roup,
                      s->mparam.labma, s->mparam.u0p, step);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  *** CUDA error after step %d: %s ***\n",
                   step, cudaGetErrorString(err));
            return;
        }
    }
    s->mlTransData2Host();
    export_surface_fields(fl, out_dir);
}

// ---- droplet (omega=1.0) ----
static void gen_droplet(int N, double R, int steps, const char* out_dir) {
    printf("\n=== Droplet R=%.0f (%d^3, %d steps) ===\n", R, N, steps);
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_surface_solver(&fl, N, N, N, 1.0 / 6.0, 0.0);
    double c = N / 2.0;
    setup_droplet(fl, N, N, N, R, c, c, c);
    set_boundary_walls(fl, N, N, N);
    run_and_export(s, fl, steps, out_dir);
    delete s; delete fl;
}

// ---- flat interface ----
static void gen_flat_interface(int N, int steps, const char* out_dir) {
    printf("\n=== Flat Interface (%d^3, %d steps) ===\n", N, steps);
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_surface_solver(&fl, N, N, N, 1.0 / 6.0, 0.0);
    setup_flat_interface(fl, N, N, N, N / 2);
    set_boundary_walls(fl, N, N, N);
    run_and_export(s, fl, steps, out_dir);
    delete s; delete fl;
}

// ---- near droplets ----
static void gen_near_droplets(int N, double R, double center_dist,
                              int steps, const char* out_dir) {
    printf("\n=== Near Droplets R=%.0f dist=%.0f (%d^3, %d steps) ===\n",
           R, center_dist, N, steps);
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_surface_solver(&fl, N, N, N, 1.0 / 6.0, 0.0);
    double c = N / 2.0;
    setup_near_droplets(fl, N, N, N, R,
                         c - center_dist / 2, c, c,
                         c + center_dist / 2, c, c);
    set_boundary_walls(fl, N, N, N);
    run_and_export(s, fl, steps, out_dir);
    delete s; delete fl;
}

// ---- ellipsoid ----
static void gen_ellipsoid(int N, double rx, double ry, double rz,
                          int steps, const char* out_dir) {
    printf("\n=== Ellipsoid rx=%.0f ry=%.0f rz=%.0f (%d^3, %d steps) ===\n",
           rx, ry, rz, N, steps);
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_surface_solver(&fl, N, N, N, 1.0 / 6.0, 0.0);
    double cx = N / 2.0, cy = N / 2.0, cz = N / 2.0;
    double delta = 1.5;
    long count = fl->count;
    for (long idx = 0; idx < count; idx++) {
        fl->fMom[idx] = 1.0f; fl->fMomPost[idx] = 1.0f;
        for (int m = 1; m < 10; m++) {
            fl->fMom[m * count + idx] = 0.0f; fl->fMomPost[m * count + idx] = 0.0f;
        }
        fl->forcex[idx] = 0.0f; fl->forcey[idx] = 0.0f; fl->forcez[idx] = 0.0f;
        fl->c_value[idx] = 0.0f;
        fl->tag_matrix[idx] = -1;
    }
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                long idx = k * N * N + j * N + i;
                double dx = (i - cx) / rx, dy = (j - cy) / ry, dz = (k - cz) / rz;
                double dist = sqrt(dx * dx + dy * dy + dz * dz);
                double phi_raw = (1.0 - dist) * rx / delta + 0.5;
                double phi_val = phi_raw < 0.0 ? 0.0 : (phi_raw > 1.0 ? 1.0 : phi_raw);
                fl->phi[idx] = (REAL)phi_val;
                if (phi_val >= 1.0 - 1e-6) {
                    fl->flag[idx] = TYPE_F; fl->mass[idx] = 1.0f;
                } else if (phi_val <= 1e-6) {
                    fl->flag[idx] = TYPE_G; fl->mass[idx] = 0.0f;
                } else {
                    fl->flag[idx] = TYPE_I; fl->mass[idx] = (REAL)phi_val;
                }
                fl->massex[idx] = 0.0f;
            }
        }
    }
    set_boundary_walls(fl, N, N, N);
    run_and_export(s, fl, steps, out_dir);
    delete s; delete fl;
}

// ---- falling droplet ----
static void gen_falling_droplet(int N, double R, double gz,
                                int steps, const char* out_dir) {
    printf("\n=== Falling Droplet R=%.0f gz=%.4f (%d^3, %d steps) ===\n",
           R, gz, N, steps);
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_surface_solver(&fl, N, N, N, 1.0 / 6.0, gz);
    double c = N / 2.0;
    setup_droplet(fl, N, N, N, R, c, c, c);
    // Apply gravity force
    long count = fl->count;
    for (long idx = 0; idx < count; idx++) {
        fl->forcez[idx] = (REAL)gz;
    }
    set_boundary_walls(fl, N, N, N);
    run_and_export(s, fl, steps, out_dir);
    delete s; delete fl;
}

// ---- droplet near wall ----
static void gen_droplet_wall(int N, double R, double z_wall,
                             int steps, const char* out_dir) {
    printf("\n=== Droplet Wall R=%.0f z_wall=%.0f (%d^3, %d steps) ===\n",
           R, z_wall, N, steps);
    mrFlow3D* fl = 0;
    mrSolver3D* s = build_surface_solver(&fl, N, N, N, 1.0 / 6.0, 0.0);
    double cx = N / 2.0, cy = N / 2.0;
    double cz = z_wall + R;  // droplet centre above wall
    setup_droplet(fl, N, N, N, R, cx, cy, cz);
    // Set z=0 plane as TYPE_S (solid wall)
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            long idx = 0 * N * N + j * N + i;  // k=0
            fl->flag[idx] = TYPE_S;
            fl->phi[idx] = 0.0f;
            fl->mass[idx] = 0.0f;
        }
    set_boundary_walls(fl, N, N, N);
    run_and_export(s, fl, steps, out_dir);
    delete s; delete fl;
}

// ---- droplet with different omega ----
static void gen_droplet_omega(int N, double R, double omega,
                              int steps, const char* out_dir) {
    printf("\n=== Droplet R=%.0f omega=%.4f (%d^3, %d steps) ===\n",
           R, omega, N, steps);
    // Note: the GPU kernel hardcodes omega as 1/(3*1e-4+0.5).
    // For parameter-variation tests, the host omega is passed via labma
    // which the kernel uses for turbulence modulation, not collision.
    // The effective collision omega is always ~1/(3*1e-4+0.5) ≈ 1.998.
    // We pass the desired omega as labma for documentation, but the actual
    // collision relaxation is kernel-fixed.
    // For the Warp comparison, the Warp model.omega must be set to the
    // same kernel-effective value (≈1.998) for consistency.
    mrFlow3D* fl = 0;
    // Use vis=omega_to_vis (1/6 for omega=1, but vis doesn't set omega in kernel)
    mrSolver3D* s = build_surface_solver(&fl, N, N, N, 1.0 / 6.0, 0.0);
    // Override labma to pass desired omega to GPU (used in turbulence)
    s->mparam.labma = (REAL)omega;
    double c = N / 2.0;
    setup_droplet(fl, N, N, N, R, c, c, c);
    set_boundary_walls(fl, N, N, N);
    run_and_export(s, fl, steps, out_dir);
    delete s; delete fl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== HOME-FSLBM Phase 1+2 Golden Data Generator ===\n");

    const char* base = "../../../golden_data";
    make_dir(base);  // ensure parent directory exists

    // Phase 1
    gen_shear();
    gen_bounce();
    gen_turb();

    // Phase 2 — surface scenes
    // droplet_r4:  16^3, R=4,  5 steps
    gen_droplet(16, 4, 5, (std::string(base) + "/droplet_r4").c_str());

    // droplet_r8:  32^3, R=8,  10 steps
    gen_droplet(32, 8, 10, (std::string(base) + "/droplet_r8").c_str());

    // droplet_r12: 32^3, R=12, 10 steps
    gen_droplet(32, 12, 10, (std::string(base) + "/droplet_r12").c_str());

    // flat_interface: 32^3, 10 steps
    gen_flat_interface(32, 10, (std::string(base) + "/flat_interface").c_str());

    // near_droplets: 32^3, two R=6 droplets, centre dist 14, 20 steps
    gen_near_droplets(32, 6, 14, 20, (std::string(base) + "/near_droplets").c_str());

    // ellipsoid: 32^3, rx=10 ry=6 rz=6, 50 steps
    gen_ellipsoid(32, 10, 6, 6, 50, (std::string(base) + "/ellipsoid").c_str());

    // falling_droplet: 32^3, R=6, gz=-0.001, 30 steps
    gen_falling_droplet(32, 6, -0.001, 30, (std::string(base) + "/falling_droplet").c_str());

    // droplet_wall: 32^3, R=6, z_wall=0, 10 steps
    gen_droplet_wall(32, 6, 0, 10, (std::string(base) + "/droplet_wall").c_str());

    // droplet_tau055: 32^3, R=8, omega≈1.818, 20 steps
    gen_droplet_omega(32, 8, 1.0 / 0.55, 20, (std::string(base) + "/droplet_tau055").c_str());

    // droplet_tau2: 32^3, R=8, omega=0.5, 20 steps
    gen_droplet_omega(32, 8, 0.5, 20, (std::string(base) + "/droplet_tau2").c_str());

    // ---- Diagnostic: droplet_r8 step 0/1/2 ----
    gen_droplet_r8_diag();

    printf("\nDone. All golden data generated.\n");
    return 0;
}