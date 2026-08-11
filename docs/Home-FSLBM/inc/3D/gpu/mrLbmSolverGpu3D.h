#pragma once
#ifndef _MRLBMSOLVERGPU3DH_
#define _MRLBMSOLVERGPU3DH_

#include "../cpu/mrFlow3D.h"

extern "C"
{
	void mrSolver3DGpu(mrFlow3D* mlflow, MLFluidParam3D* param, float N, float l0p, float roup, float labma,
		float u0p, int time_step);
	void mrInit3DGpu(mrFlow3D* mlflow, MLFluidParam3D* param);
	void coupling(mrFlow3D* mlflow, MLFluidParam3D* param, float N, float l0p, float roup, float labma,
		float u0p, int time_step);
	/** GPU InitBubble pipeline (convert → CCL → parse → create → tag). */
	void InitBubble(mrFlow3D* mlflow, MLFluidParam3D* param);

	/** Full dissolved-gas handle (reconstruction → collide → volume_g → swap → rho). */
	void g_handle(mrFlow3D* mlflow, MLFluidParam3D* param, int time);

	/** Golden / diagnostic launches (individual Phase-4 stages). */
	void launch_g_reconstruction(mrFlow3D* mlflow, MLFluidParam3D* param);
	void launch_g_stream_collide(mrFlow3D* mlflow, MLFluidParam3D* param, int time);
	void launch_g_swap(mrFlow3D* mlflow, MLFluidParam3D* param);
	void launch_bubble_volume_g_update(mrFlow3D* mlflow, MLFluidParam3D* param, int time);
	void launch_bubble_rho_update(mrFlow3D* mlflow);
	void launch_calculate_disjoint(mrFlow3D* mlflow, MLFluidParam3D* param);
	void launch_atmosphere_rho_update(mrFlow3D* mlflow, MLFluidParam3D* param,
		float N, float l0p, float roup, float labma, float u0p, int time_step);
	void launch_atmosphere_volme_update(mrFlow3D* mlflow);
}
#endif // !_MRLBMSOLVERGPU3DH_
