// export_golden.cu — generates golden data for HOME-FSLBM unit tests
// Build: nvcc -I../common -Iinc/3D/gpu -Iinc/3D/cpu -o export_golden.exe export_golden.cu
// Run:   export_golden.exe

#include <cstdio>
#include <cmath>
#include "mrUtilFuncGpu3D.h"

// Helper to write float array to file, one value per line
void write_floats(const char* filename, const float* data, int count) {
    FILE* f = fopen(filename, "w");
    for (int i = 0; i < count; i++)
        fprintf(f, "%.15e\n", data[i]);
    fclose(f);
    printf("  wrote %s (%d values)\n", filename, count);
}

int main() {
    mrUtilFuncGpu3D util;
    float feq[27];
    float f_out;

    printf("=== calculate_f_eq ===\n");

    // Case 1: rho=1.0, u=(0.02, 0.01, 0.0)
    util.calculate_f_eq(1.0f, 0.02f, 0.01f, 0.0f, feq);
    write_floats("golden_data/f_eq_rho1.0_ux0.02_uy0.01.txt", feq, 27);

    // Case 2: rho=2.5, u=(0, 0, 0) — rest particle weight test
    util.calculate_f_eq(2.5f, 0.0f, 0.0f, 0.0f, feq);
    write_floats("golden_data/f_eq_rho2.5_rest.txt", feq, 27);

    // Case 3: rho=1.5, u=(0, 0, 0) — face direction weights test
    util.calculate_f_eq(1.5f, 0.0f, 0.0f, 0.0f, feq);
    write_floats("golden_data/f_eq_rho1.5_rest.txt", feq, 27);

    printf("\n=== mlCalDistributionFourthOrderD3Q27AtIndex ===\n");

    // Case 1: roundtrip at rest — rho=1.0, u=(0,0,0), pi=cs2 on diag, 0 off-diag
    // Warp test uses CS2 = 1.0/3.0
    float cs2 = 1.0f / 3.0f;
    float recon_rest[27];
    for (int di = 0; di < 27; di++) {
        util.mlCalDistributionFourthOrderD3Q27AtIndex(
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            di, f_out);
        recon_rest[di] = f_out;
    }
    write_floats("golden_data/recon_rest.txt", recon_rest, 27);

    // Case 2: density sum preserved — rho=1.0, u=(0.1,0,0), non-eq stress
    float pi_xx = cs2 + 0.01f;
    float pi_yy = cs2 - 0.005f;
    float pi_zz = cs2 - 0.005f;
    float pi_xy = 0.002f;
    float recon_non_eq[27];
    for (int di = 0; di < 27; di++) {
        util.mlCalDistributionFourthOrderD3Q27AtIndex(
            1.0f, 0.1f, 0.0f, 0.0f,
            0.01f, 0.002f, 0.0f, -0.005f, 0.0f, -0.005f,
            di, f_out);
        recon_non_eq[di] = f_out;
    }
    write_floats("golden_data/recon_rho1.0_ux0.1.txt", recon_non_eq, 27);

    printf("\n=== mlGetPIAfterCollision ===\n");

    // Case 1: equilibrium preserved at rest
    // Input: rho=1.0, u=(0,0,0), F=(0,0,0), omega=1.0, stress=cs2
    float pxx = cs2, pyy = cs2, pzz = cs2, pxy = 0.0f, pxz = 0.0f, pyz = 0.0f;
    util.mlGetPIAfterCollision(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                               pxx, pxy, pxz, pyy, pyz, pzz);
    float collision_rest[6] = {pxx, pxy, pxz, pyy, pyz, pzz};
    write_floats("golden_data/collision_rest.txt", collision_rest, 6);

    // Case 2: relaxation toward equilibrium — omega=1.0, non-eq stress
    // rho=1.0, u=(0.1,0,0), F=(0,0,0), omega=1.0
    pxx = cs2 + 0.1f; pyy = cs2 - 0.05f; pzz = cs2 - 0.05f; pxy = 0.02f; pxz = 0.0f; pyz = 0.0f;
    util.mlGetPIAfterCollision(1.0f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                               pxx, pxy, pxz, pyy, pyz, pzz);
    float collision_relax[6] = {pxx, pxy, pxz, pyy, pyz, pzz};
    write_floats("golden_data/collision_relax.txt", collision_relax, 6);

    // Case 3: force contribution
    // rho=1.0, u=(0.2,0,0), F=(0.01, 0, 0), omega=1.0, stress=cs2
    pxx = cs2; pyy = cs2; pzz = cs2; pxy = 0.0f; pxz = 0.0f; pyz = 0.0f;
    util.mlGetPIAfterCollision(1.0f, 0.2f, 0.0f, 0.0f, 0.01f, 0.0f, 0.0f, 1.0f,
                               pxx, pxy, pxz, pyy, pyz, pzz);
    float collision_force[6] = {pxx, pxy, pxz, pyy, pyz, pzz};
    write_floats("golden_data/collision_force.txt", collision_force, 6);

    printf("\n=== calculate_rho_u ===\n");

    // compute_rho_u analytical expectation:
    //   rho = sum(f_eq) + 1.0 = (rho-1) + 1.0 = rho = 1.0
    //   u   = momentum / rho = rho*ux / rho = ux = 0.02, etc.
    // (avoid calculate_rho_u host call — __constant__ arrays not accessible)
    float rho_u[4] = {1.0f, 0.02f, 0.01f, 0.0f};
    write_floats("golden_data/compute_rho_u_feq.txt", rho_u, 4);

    printf("\nDone.\n");
    return 0;
}
