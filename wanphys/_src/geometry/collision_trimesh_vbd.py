# branch: collision_rel/htk
# collision class in original vbd solver

import numpy as np
import warp as wp

from .collision_backend import CollisionBackend
from newton._src.geometry.kernels import (
    compute_edge_aabbs,
    compute_tri_aabbs,
    edge_colliding_edges_detection_kernel,
    init_triangle_collision_data_kernel,
    triangle_triangle_collision_detection_kernel,
    vertex_triangle_collision_detection_kernel,
)
from newton._src.solvers.vbd.particle_vbd_kernels import (
    ParticleForceElementAdjacencyInfo,
    build_edge_n_ring_edge_collision_filter,
    build_vertex_n_ring_tris_collision_filter,
    create_edge_edge_division_plane_closest_pt,
    create_vertex_triangle_division_plane_closest_pt,
    _count_num_adjacent_edges,
    _count_num_adjacent_faces,
    _count_num_adjacent_springs,
    _count_num_adjacent_tets,
    _fill_adjacent_edges,
    _fill_adjacent_faces,
    _fill_adjacent_springs,
    _fill_adjacent_tets,
    get_vertex_adjacent_edge_id_order,
    get_vertex_adjacent_face_id_order,
    get_vertex_num_adjacent_edges,
    get_vertex_num_adjacent_faces,
    planar_truncation_t,
)
from newton._src.sim import Model, State

from .utils.geometry_utils import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    evaluate_body_particle_contact,
)

from .narrow_phase_vf import (
    build_vertex_triangle_contact_geometry_kernel,
    evaluate_vertex_triangle_collision_force_hessian_4_vertices_cached
)
from .narrow_phase_ee import (
    build_edge_edge_contact_geometry_kernel,
    evaluate_edge_edge_contact_2_vertices_cached
)


def _set_to_csr(filter_sets):
    counts = np.fromiter((len(s) for s in filter_sets), dtype=np.int32, count=len(filter_sets))
    offsets = np.zeros((len(filter_sets) + 1,), dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)

    if offsets[-1] == 0:
        values = np.zeros((0,), dtype=np.int32)
    else:
        values = np.empty((offsets[-1],), dtype=np.int32)
        cursor = 0
        for items in filter_sets:
            if not items:
                continue
            ordered = np.fromiter(sorted(items), dtype=np.int32, count=len(items))
            values[cursor : cursor + len(ordered)] = ordered
            cursor += len(ordered)

    return values, offsets

@wp.struct
class CollisionTriMeshVBDInfo:
    # size: 2 x sum(vertex_colliding_triangles_buffer_sizes)
    # every two elements records the vertex index and a triangle index it collides to
    vertex_colliding_triangles: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_buffer_sizes: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32)
    vertex_colliding_triangles_min_dist: wp.array(dtype=float)

    triangle_colliding_vertices: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_offsets: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_buffer_sizes: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_count: wp.array(dtype=wp.int32)
    triangle_colliding_vertices_min_dist: wp.array(dtype=float)

    # size: 2 x sum(edge_colliding_edges_buffer_sizes)
    # every two elements records the edge index and an edge index it collides to
    edge_colliding_edges: wp.array(dtype=wp.int32)
    edge_colliding_edges_offsets: wp.array(dtype=wp.int32)
    edge_colliding_edges_buffer_sizes: wp.array(dtype=wp.int32)
    edge_colliding_edges_count: wp.array(dtype=wp.int32)
    edge_colliding_edges_min_dist: wp.array(dtype=float)

    # Narrow-phase geometry caches (aligned with collision buffer slots)
    vertex_triangle_contact_bary: wp.array(dtype=wp.vec3)
    vertex_triangle_contact_normal: wp.array(dtype=wp.vec3)
    vertex_triangle_contact_dist: wp.array(dtype=float)

    edge_edge_contact_s: wp.array(dtype=float)
    edge_edge_contact_t: wp.array(dtype=float)
    edge_edge_contact_normal: wp.array(dtype=wp.vec3)
    edge_edge_contact_dist: wp.array(dtype=float)


class CollisionTriMeshVBD(CollisionBackend):
    name: str = "vbd"

    def __init__(
        self,
        model: Model,
        record_triangle_contacting_vertices=False,
        vertex_positions=None,
        vertex_collision_buffer_pre_alloc=8,
        vertex_collision_buffer_max_alloc=256,
        vertex_triangle_filtering_list=None,
        vertex_triangle_filtering_list_offsets=None,
        triangle_collision_buffer_pre_alloc=16,
        triangle_collision_buffer_max_alloc=256,
        edge_collision_buffer_pre_alloc=8,
        edge_collision_buffer_max_alloc=256,
        edge_filtering_list=None,
        edge_filtering_list_offsets=None,
        triangle_triangle_collision_buffer_pre_alloc=8,
        triangle_triangle_collision_buffer_max_alloc=256,
        edge_edge_parallel_epsilon=1e-5,
        collision_detection_block_size=16,
    ):
        self.model = model
        self.record_triangle_contacting_vertices = record_triangle_contacting_vertices
        self.vertex_positions = model.particle_q if vertex_positions is None else vertex_positions
        self.device = model.device
        self.vertex_collision_buffer_pre_alloc = vertex_collision_buffer_pre_alloc
        self.vertex_collision_buffer_max_alloc = vertex_collision_buffer_max_alloc
        self.triangle_collision_buffer_pre_alloc = triangle_collision_buffer_pre_alloc
        self.triangle_collision_buffer_max_alloc = triangle_collision_buffer_max_alloc
        self.edge_collision_buffer_pre_alloc = edge_collision_buffer_pre_alloc
        self.edge_collision_buffer_max_alloc = edge_collision_buffer_max_alloc
        self.triangle_triangle_collision_buffer_pre_alloc = triangle_triangle_collision_buffer_pre_alloc
        self.triangle_triangle_collision_buffer_max_alloc = triangle_triangle_collision_buffer_max_alloc
        self.vertex_triangle_filtering_list = vertex_triangle_filtering_list
        self.vertex_triangle_filtering_list_offsets = vertex_triangle_filtering_list_offsets
        self.edge_filtering_list = edge_filtering_list
        self.edge_filtering_list_offsets = edge_filtering_list_offsets
        self.edge_edge_parallel_epsilon = edge_edge_parallel_epsilon
        self.collision_detection_block_size = collision_detection_block_size

        self.lower_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_tris = wp.array(shape=(model.tri_count,), dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, model.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            dim=model.tri_count,
            device=model.device,
        )
        self.bvh_tris = wp.Bvh(self.lower_bounds_tris, self.upper_bounds_tris)

        self.vertex_colliding_triangles = wp.zeros(
            shape=(2 * model.particle_count * self.vertex_collision_buffer_pre_alloc,),
            dtype=wp.int32,
            device=self.device,
        )
        self.vertex_colliding_triangles_count = wp.array(
            shape=(model.particle_count,), dtype=wp.int32, device=self.device
        )
        self.vertex_colliding_triangles_min_dist = wp.array(
            shape=(model.particle_count,), dtype=float, device=self.device
        )
        self.vertex_colliding_triangles_buffer_sizes = wp.full(
            shape=(model.particle_count,),
            value=self.vertex_collision_buffer_pre_alloc,
            dtype=wp.int32,
            device=self.device,
        )
        self.vertex_colliding_triangles_offsets = wp.array(
            shape=(model.particle_count + 1,), dtype=wp.int32, device=self.device
        )
        self.compute_collision_buffer_offsets(
            self.vertex_colliding_triangles_buffer_sizes, self.vertex_colliding_triangles_offsets
        )

        if record_triangle_contacting_vertices:
            self.triangle_colliding_vertices = wp.zeros(
                shape=(model.tri_count * self.triangle_collision_buffer_pre_alloc,),
                dtype=wp.int32,
                device=self.device,
            )
            self.triangle_colliding_vertices_count = wp.zeros(
                shape=(model.tri_count,), dtype=wp.int32, device=self.device
            )
            self.triangle_colliding_vertices_buffer_sizes = wp.full(
                shape=(model.tri_count,),
                value=self.triangle_collision_buffer_pre_alloc,
                dtype=wp.int32,
                device=self.device,
            )
            self.triangle_colliding_vertices_offsets = wp.array(
                shape=(model.tri_count + 1,), dtype=wp.int32, device=self.device
            )
            self.compute_collision_buffer_offsets(
                self.triangle_colliding_vertices_buffer_sizes, self.triangle_colliding_vertices_offsets
            )
        else:
            self.triangle_colliding_vertices = None
            self.triangle_colliding_vertices_count = None
            self.triangle_colliding_vertices_buffer_sizes = None
            self.triangle_colliding_vertices_offsets = None

        self.triangle_colliding_vertices_min_dist = wp.array(shape=(model.tri_count,), dtype=float, device=self.device)

        self.edge_colliding_edges = wp.zeros(
            shape=(2 * model.edge_count * self.edge_collision_buffer_pre_alloc,), dtype=wp.int32, device=self.device
        )
        self.edge_colliding_edges_count = wp.zeros(shape=(model.edge_count,), dtype=wp.int32, device=self.device)
        self.edge_colliding_edges_buffer_sizes = wp.full(
            shape=(model.edge_count,),
            value=self.edge_collision_buffer_pre_alloc,
            dtype=wp.int32,
            device=self.device,
        )
        self.edge_colliding_edges_offsets = wp.array(shape=(model.edge_count + 1,), dtype=wp.int32, device=self.device)
        self.compute_collision_buffer_offsets(self.edge_colliding_edges_buffer_sizes, self.edge_colliding_edges_offsets)
        self.edge_colliding_edges_min_dist = wp.array(shape=(model.edge_count,), dtype=float, device=self.device)

        self.lower_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        self.upper_bounds_edges = wp.array(shape=(model.edge_count,), dtype=wp.vec3, device=model.device)
        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, model.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            dim=model.edge_count,
            device=model.device,
        )
        self.bvh_edges = wp.Bvh(self.lower_bounds_edges, self.upper_bounds_edges)
        self.resize_flags = wp.zeros(shape=(4,), dtype=wp.int32, device=self.device)

        # Narrow-phase geometry caches (aligned with collision buffer slots)
        vertex_contact_capacity = int(self.vertex_colliding_triangles.shape[0] // 2)
        edge_contact_capacity = int(self.edge_colliding_edges.shape[0] // 2)

        self.vertex_triangle_contact_bary = wp.zeros(
            shape=(vertex_contact_capacity,), dtype=wp.vec3, device=self.device
        )
        self.vertex_triangle_contact_normal = wp.zeros(
            shape=(vertex_contact_capacity,), dtype=wp.vec3, device=self.device
        )
        self.vertex_triangle_contact_dist = wp.zeros(shape=(vertex_contact_capacity,), dtype=float, device=self.device)

        self.edge_edge_contact_s = wp.zeros(shape=(edge_contact_capacity,), dtype=float, device=self.device)
        self.edge_edge_contact_t = wp.zeros(shape=(edge_contact_capacity,), dtype=float, device=self.device)
        self.edge_edge_contact_normal = wp.zeros(shape=(edge_contact_capacity,), dtype=wp.vec3, device=self.device)
        self.edge_edge_contact_dist = wp.zeros(shape=(edge_contact_capacity,), dtype=float, device=self.device)

        self.collision_info = self.get_collision_data()

        self.pos_prev_collision_detection = wp.zeros_like(model.particle_q, device=self.device)
        self.particle_conservative_bounds = wp.zeros((model.particle_count,), dtype=float, device=self.device)
        self.particle_adjacency = self._compute_particle_adjacency().to(self.device)

        # data for triangle-triangle intersection; they will only be initialized on demand, as triangle-triangle intersection is not needed for simulation
        self.triangle_intersecting_triangles = None
        self.triangle_intersecting_triangles_count = None
        self.triangle_intersecting_triangles_offsets = None

    def set_collision_filter_list(
        self,
        vertex_triangle_filtering_list,
        vertex_triangle_filtering_list_offsets,
        edge_filtering_list,
        edge_filtering_list_offsets,
    ):
        self.vertex_triangle_filtering_list = vertex_triangle_filtering_list
        self.vertex_triangle_filtering_list_offsets = vertex_triangle_filtering_list_offsets
        self.edge_filtering_list = edge_filtering_list
        self.edge_filtering_list_offsets = edge_filtering_list_offsets

    def build(self, model: Model, device=None):
        """Backend interface. CollisionVBD builds in __init__, so this is a light reinit."""
        self.rebuild(model.particle_q)

    def generate_candidates(self, state: State, params, dt: float, out_pairs, out_pair_count) -> None:
        """Backend interface. Uses internal broad-phase buffers."""
        def _get_param(obj, key, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        max_query_radius = _get_param(params, "max_query_radius", None)
        if max_query_radius is None:
            max_query_radius = _get_param(params, "particle_self_contact_margin", None)
        if max_query_radius is None:
            max_query_radius = _get_param(params, "collision_margin", None)

        min_query_radius = _get_param(params, "min_query_radius", None)
        if min_query_radius is None:
            min_query_radius = _get_param(params, "particle_self_contact_radius", 0.0)

        self.vertex_positions = state.particle_q

        if max_query_radius is not None and max_query_radius > 0.0:
            self.vertex_triangle_collision_detection(max_query_radius, min_query_radius=min_query_radius)
            self.edge_edge_collision_detection(max_query_radius, min_query_radius=min_query_radius)

    def narrow_phase(self, state: State, params, dt: float, pairs, pair_count, out_hits, mode: str) -> None:
        """Backend interface. Builds contact geometry using internal candidates."""
        self.build_contact_geometry(state.particle_q)

    def build_particle_contact_filtering_list(
        self,
        topological_contact_filter_threshold: int,
        external_vertex_contact_filtering_map=None,
        external_edge_contact_filtering_map=None,
    ):
        if not self.model.tri_count:
            self.set_collision_filter_list(None, None, None, None)
            return

        v_tri_filter_sets = None
        edge_edge_filter_sets = None

        if topological_contact_filter_threshold >= 2:
            if self.particle_adjacency.v_adj_faces_offsets.size > 0:
                v_tri_filter_sets = build_vertex_n_ring_tris_collision_filter(
                    topological_contact_filter_threshold,
                    self.model.particle_count,
                    self.model.edge_indices.numpy(),
                    self.particle_adjacency.v_adj_edges.numpy(),
                    self.particle_adjacency.v_adj_edges_offsets.numpy(),
                    self.particle_adjacency.v_adj_faces.numpy(),
                    self.particle_adjacency.v_adj_faces_offsets.numpy(),
                )
            if self.particle_adjacency.v_adj_edges_offsets.size > 0:
                edge_edge_filter_sets = build_edge_n_ring_edge_collision_filter(
                    topological_contact_filter_threshold,
                    self.model.edge_indices.numpy(),
                    self.particle_adjacency.v_adj_edges.numpy(),
                    self.particle_adjacency.v_adj_edges_offsets.numpy(),
                )

        if external_vertex_contact_filtering_map is not None:
            if v_tri_filter_sets is None:
                v_tri_filter_sets = [set() for _ in range(self.model.particle_count)]
            for vertex_id, filter_set in external_vertex_contact_filtering_map.items():
                v_tri_filter_sets[vertex_id].update(filter_set)

        if external_edge_contact_filtering_map is not None:
            if edge_edge_filter_sets is None:
                edge_edge_filter_sets = [set() for _ in range(self.model.edge_indices.shape[0])]
            for edge_id, filter_set in external_edge_contact_filtering_map.items():
                edge_edge_filter_sets[edge_id].update(filter_set)

        vertex_triangle_filtering_list = None
        vertex_triangle_filtering_list_offsets = None
        edge_filtering_list = None
        edge_filtering_list_offsets = None

        if v_tri_filter_sets is not None:
            (
                vertex_triangle_filtering_list,
                vertex_triangle_filtering_list_offsets,
            ) = _set_to_csr(v_tri_filter_sets)
            vertex_triangle_filtering_list = wp.array(
                vertex_triangle_filtering_list, dtype=int, device=self.device
            )
            vertex_triangle_filtering_list_offsets = wp.array(
                vertex_triangle_filtering_list_offsets, dtype=int, device=self.device
            )

        if edge_edge_filter_sets is not None:
            (
                edge_filtering_list,
                edge_filtering_list_offsets,
            ) = _set_to_csr(edge_edge_filter_sets)
            edge_filtering_list = wp.array(edge_filtering_list, dtype=int, device=self.device)
            edge_filtering_list_offsets = wp.array(edge_filtering_list_offsets, dtype=int, device=self.device)

        self.set_collision_filter_list(
            vertex_triangle_filtering_list,
            vertex_triangle_filtering_list_offsets,
            edge_filtering_list,
            edge_filtering_list_offsets,
        )

    def _compute_particle_adjacency(self):
        particle_adjacency = ParticleForceElementAdjacencyInfo()

        with wp.ScopedDevice("cpu"):
            if self.model.edge_indices:
                edges_array = self.model.edge_indices.to("cpu")
                num_vertex_adjacent_edges = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                wp.launch(
                    kernel=_count_num_adjacent_edges,
                    inputs=[edges_array, num_vertex_adjacent_edges],
                    dim=1,
                    device="cpu",
                )

                num_vertex_adjacent_edges = num_vertex_adjacent_edges.numpy()
                vertex_adjacent_edges_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_edges_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_edges)[:]
                vertex_adjacent_edges_offsets[0] = 0
                particle_adjacency.v_adj_edges_offsets = wp.array(vertex_adjacent_edges_offsets, dtype=wp.int32)

                vertex_adjacent_edges_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                edge_adjacency_array_size = 2 * num_vertex_adjacent_edges.sum()
                particle_adjacency.v_adj_edges = wp.empty(shape=(edge_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=_fill_adjacent_edges,
                    inputs=[
                        edges_array,
                        particle_adjacency.v_adj_edges_offsets,
                        vertex_adjacent_edges_fill_count,
                        particle_adjacency.v_adj_edges,
                    ],
                    dim=1,
                    device="cpu",
                )
            else:
                particle_adjacency.v_adj_edges_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_edges = wp.empty(shape=(0,), dtype=wp.int32)

            if self.model.tri_indices:
                face_indices = self.model.tri_indices.to("cpu")
                num_vertex_adjacent_faces = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                wp.launch(
                    kernel=_count_num_adjacent_faces,
                    inputs=[face_indices, num_vertex_adjacent_faces],
                    dim=1,
                    device="cpu",
                )

                num_vertex_adjacent_faces = num_vertex_adjacent_faces.numpy()
                vertex_adjacent_faces_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_faces_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_faces)[:]
                vertex_adjacent_faces_offsets[0] = 0
                particle_adjacency.v_adj_faces_offsets = wp.array(vertex_adjacent_faces_offsets, dtype=wp.int32)

                vertex_adjacent_faces_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                face_adjacency_array_size = 2 * num_vertex_adjacent_faces.sum()
                particle_adjacency.v_adj_faces = wp.empty(shape=(face_adjacency_array_size,), dtype=wp.int32)

                wp.launch(
                    kernel=_fill_adjacent_faces,
                    inputs=[
                        face_indices,
                        particle_adjacency.v_adj_faces_offsets,
                        vertex_adjacent_faces_fill_count,
                        particle_adjacency.v_adj_faces,
                    ],
                    dim=1,
                    device="cpu",
                )
            else:
                particle_adjacency.v_adj_faces_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_faces = wp.empty(shape=(0,), dtype=wp.int32)

            if self.model.tet_indices:
                tet_indices = self.model.tet_indices.to("cpu")
                num_vertex_adjacent_tets = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=_count_num_adjacent_tets,
                    inputs=[tet_indices, num_vertex_adjacent_tets],
                    dim=1,
                    device="cpu",
                )

                num_vertex_adjacent_tets = num_vertex_adjacent_tets.numpy()
                vertex_adjacent_tets_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_tets_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_tets)[:]
                vertex_adjacent_tets_offsets[0] = 0
                particle_adjacency.v_adj_tets_offsets = wp.array(vertex_adjacent_tets_offsets, dtype=wp.int32)

                vertex_adjacent_tets_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                particle_adjacency.v_adj_tets = wp.empty(
                    shape=(2 * num_vertex_adjacent_tets.sum(),), dtype=wp.int32
                )

                wp.launch(
                    kernel=_fill_adjacent_tets,
                    inputs=[
                        tet_indices,
                        particle_adjacency.v_adj_tets_offsets,
                        vertex_adjacent_tets_fill_count,
                        particle_adjacency.v_adj_tets,
                    ],
                    dim=1,
                    device="cpu",
                )
            else:
                particle_adjacency.v_adj_tets_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_tets = wp.empty(shape=(0,), dtype=wp.int32)

            if self.model.spring_indices:
                spring_array = self.model.spring_indices.to("cpu")
                num_vertex_adjacent_spring = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)

                wp.launch(
                    kernel=_count_num_adjacent_springs,
                    inputs=[spring_array, num_vertex_adjacent_spring],
                    dim=1,
                    device="cpu",
                )

                num_vertex_adjacent_spring = num_vertex_adjacent_spring.numpy()
                vertex_adjacent_springs_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
                vertex_adjacent_springs_offsets[1:] = np.cumsum(num_vertex_adjacent_spring)[:]
                vertex_adjacent_springs_offsets[0] = 0
                particle_adjacency.v_adj_springs_offsets = wp.array(vertex_adjacent_springs_offsets, dtype=wp.int32)

                vertex_adjacent_springs_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32)
                particle_adjacency.v_adj_springs = wp.empty(
                    shape=(num_vertex_adjacent_spring.sum(),), dtype=wp.int32
                )

                wp.launch(
                    kernel=_fill_adjacent_springs,
                    inputs=[
                        spring_array,
                        particle_adjacency.v_adj_springs_offsets,
                        vertex_adjacent_springs_fill_count,
                        particle_adjacency.v_adj_springs,
                    ],
                    dim=1,
                    device="cpu",
                )
            else:
                particle_adjacency.v_adj_springs_offsets = wp.empty(shape=(0,), dtype=wp.int32)
                particle_adjacency.v_adj_springs = wp.empty(shape=(0,), dtype=wp.int32)

        return particle_adjacency

    def collision_detection_penetration_free(
        self,
        current_state: State,
        particle_conservative_bound_relaxation: float,
        particle_self_contact_margin: float,
        particle_rest_shape_contact_exclusion_radius: float = 0.0,
        particle_q_rest=None,
    ):
        self.refit(current_state.particle_q)
        self.vertex_triangle_collision_detection(
            particle_self_contact_margin,
            min_query_radius=particle_rest_shape_contact_exclusion_radius,
            min_distance_filtering_ref_pos=particle_q_rest,
        )
        self.edge_edge_collision_detection(
            particle_self_contact_margin,
            min_query_radius=particle_rest_shape_contact_exclusion_radius,
            min_distance_filtering_ref_pos=particle_q_rest,
        )

        self.pos_prev_collision_detection.assign(current_state.particle_q)
        wp.launch(
            kernel=compute_particle_conservative_bound_vbd,
            inputs=[
                particle_conservative_bound_relaxation,
                particle_self_contact_margin,
                self.particle_adjacency,
                self.collision_info,
            ],
            outputs=[
                self.particle_conservative_bounds,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )
        
    def get_collision_data(self) -> CollisionTriMeshVBDInfo:
        collision_info = CollisionTriMeshVBDInfo()

        collision_info.vertex_colliding_triangles = self.vertex_colliding_triangles
        collision_info.vertex_colliding_triangles_offsets = self.vertex_colliding_triangles_offsets
        collision_info.vertex_colliding_triangles_buffer_sizes = self.vertex_colliding_triangles_buffer_sizes
        collision_info.vertex_colliding_triangles_count = self.vertex_colliding_triangles_count
        collision_info.vertex_colliding_triangles_min_dist = self.vertex_colliding_triangles_min_dist

        if self.record_triangle_contacting_vertices:
            collision_info.triangle_colliding_vertices = self.triangle_colliding_vertices
            collision_info.triangle_colliding_vertices_offsets = self.triangle_colliding_vertices_offsets
            collision_info.triangle_colliding_vertices_buffer_sizes = self.triangle_colliding_vertices_buffer_sizes
            collision_info.triangle_colliding_vertices_count = self.triangle_colliding_vertices_count

        collision_info.triangle_colliding_vertices_min_dist = self.triangle_colliding_vertices_min_dist

        collision_info.edge_colliding_edges = self.edge_colliding_edges
        collision_info.edge_colliding_edges_offsets = self.edge_colliding_edges_offsets
        collision_info.edge_colliding_edges_buffer_sizes = self.edge_colliding_edges_buffer_sizes
        collision_info.edge_colliding_edges_count = self.edge_colliding_edges_count
        collision_info.edge_colliding_edges_min_dist = self.edge_colliding_edges_min_dist

        collision_info.vertex_triangle_contact_bary = self.vertex_triangle_contact_bary
        collision_info.vertex_triangle_contact_normal = self.vertex_triangle_contact_normal
        collision_info.vertex_triangle_contact_dist = self.vertex_triangle_contact_dist

        collision_info.edge_edge_contact_s = self.edge_edge_contact_s
        collision_info.edge_edge_contact_t = self.edge_edge_contact_t
        collision_info.edge_edge_contact_normal = self.edge_edge_contact_normal
        collision_info.edge_edge_contact_dist = self.edge_edge_contact_dist

        return collision_info

    def compute_collision_buffer_offsets(
        self, buffer_sizes: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32)
    ):
        assert offsets.size == buffer_sizes.size + 1
        offsets_np = np.empty(shape=(offsets.size,), dtype=np.int32)
        offsets_np[1:] = np.cumsum(buffer_sizes.numpy())[:]
        offsets_np[0] = 0
        offsets.assign(offsets_np)

    def rebuild(self, new_pos=None):
        if new_pos is not None:
            self.vertex_positions = new_pos

        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, self.model.tri_indices],
            outputs=[self.lower_bounds_tris, self.upper_bounds_tris],
            dim=self.model.tri_count,
            device=self.model.device,
        )
        self.bvh_tris.rebuild()

        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, self.model.edge_indices],
            outputs=[self.lower_bounds_edges, self.upper_bounds_edges],
            dim=self.model.edge_count,
            device=self.model.device,
        )
        self.bvh_edges.rebuild()

    def refit(self, new_pos=None):
        if isinstance(new_pos, State):
            new_pos = new_pos.particle_q
        if new_pos is not None:
            self.vertex_positions = new_pos

        self.refit_triangles()
        self.refit_edges()

    def refit_triangles(self):
        wp.launch(
            kernel=compute_tri_aabbs,
            inputs=[self.vertex_positions, self.model.tri_indices, self.lower_bounds_tris, self.upper_bounds_tris],
            dim=self.model.tri_count,
            device=self.model.device,
        )
        self.bvh_tris.refit()

    def refit_edges(self):
        wp.launch(
            kernel=compute_edge_aabbs,
            inputs=[self.vertex_positions, self.model.edge_indices, self.lower_bounds_edges, self.upper_bounds_edges],
            dim=self.model.edge_count,
            device=self.model.device,
        )
        self.bvh_edges.refit()

    def vertex_triangle_collision_detection(
        self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
    ):
        self.vertex_colliding_triangles.fill_(-1)

        if self.record_triangle_contacting_vertices:
            wp.launch(
                kernel=init_triangle_collision_data_kernel,
                inputs=[max_query_radius],
                outputs=[
                    self.triangle_colliding_vertices_count,
                    self.triangle_colliding_vertices_min_dist,
                    self.resize_flags,
                ],
                dim=self.model.tri_count,
                device=self.model.device,
            )
        else:
            self.triangle_colliding_vertices_min_dist.fill_(max_query_radius)

        wp.launch(
            kernel=vertex_triangle_collision_detection_kernel,
            inputs=[
                max_query_radius,
                min_query_radius,
                self.bvh_tris.id,
                self.vertex_positions,
                self.model.tri_indices,
                self.vertex_colliding_triangles_offsets,
                self.vertex_colliding_triangles_buffer_sizes,
                self.triangle_colliding_vertices_offsets,
                self.triangle_colliding_vertices_buffer_sizes,
                self.vertex_triangle_filtering_list,
                self.vertex_triangle_filtering_list_offsets,
                min_distance_filtering_ref_pos if min_distance_filtering_ref_pos is not None else self.vertex_positions,
            ],
            outputs=[
                self.vertex_colliding_triangles,
                self.vertex_colliding_triangles_count,
                self.vertex_colliding_triangles_min_dist,
                self.triangle_colliding_vertices,
                self.triangle_colliding_vertices_count,
                self.triangle_colliding_vertices_min_dist,
                self.resize_flags,
            ],
            dim=self.model.particle_count,
            device=self.model.device,
            block_dim=self.collision_detection_block_size,
        )

    def edge_edge_collision_detection(
        self, max_query_radius, min_query_radius=0.0, min_distance_filtering_ref_pos=None
    ):
        self.edge_colliding_edges.fill_(-1)
        wp.launch(
            kernel=edge_colliding_edges_detection_kernel,
            inputs=[
                max_query_radius,
                min_query_radius,
                self.bvh_edges.id,
                self.vertex_positions,
                self.model.edge_indices,
                self.edge_colliding_edges_offsets,
                self.edge_colliding_edges_buffer_sizes,
                self.edge_edge_parallel_epsilon,
                self.edge_filtering_list,
                self.edge_filtering_list_offsets,
                min_distance_filtering_ref_pos if min_distance_filtering_ref_pos is not None else self.vertex_positions,
            ],
            outputs=[
                self.edge_colliding_edges,
                self.edge_colliding_edges_count,
                self.edge_colliding_edges_min_dist,
                self.resize_flags,
            ],
            dim=self.model.edge_count,
            device=self.model.device,
            block_dim=self.collision_detection_block_size,
        )

    def triangle_triangle_intersection_detection(self):
        if self.triangle_intersecting_triangles is None:
            self.triangle_intersecting_triangles = wp.zeros(
                shape=(self.model.tri_count * self.triangle_triangle_collision_buffer_pre_alloc,),
                dtype=wp.int32,
                device=self.device,
            )

        if self.triangle_intersecting_triangles_count is None:
            self.triangle_intersecting_triangles_count = wp.array(
                shape=(self.model.tri_count,), dtype=wp.int32, device=self.device
            )

        if self.triangle_intersecting_triangles_offsets is None:
            buffer_sizes = np.full((self.model.tri_count,), self.triangle_triangle_collision_buffer_pre_alloc)
            offsets = np.zeros((self.model.tri_count + 1,), dtype=np.int32)
            offsets[1:] = np.cumsum(buffer_sizes)
            self.triangle_intersecting_triangles_offsets = wp.array(offsets, dtype=wp.int32, device=self.device)

        wp.launch(
            kernel=triangle_triangle_collision_detection_kernel,
            inputs=[self.bvh_tris.id, self.vertex_positions, self.model.tri_indices, self.triangle_intersecting_triangles_offsets],
            outputs=[
                self.triangle_intersecting_triangles,
                self.triangle_intersecting_triangles_count,
                self.resize_flags,
            ],
            dim=self.model.tri_count,
            device=self.model.device,
        )

    def build_vertex_triangle_contact_geometry(self, particle_q: wp.array):
        if particle_q is not None:
            self.vertex_positions = particle_q

        wp.launch(
            kernel=build_vertex_triangle_contact_geometry_kernel,
            inputs=[
                self.vertex_positions,
                self.model.tri_indices,
                self.vertex_colliding_triangles,
            ],
            outputs=[
                self.vertex_triangle_contact_bary,
                self.vertex_triangle_contact_normal,
                self.vertex_triangle_contact_dist,
            ],
            dim=self.vertex_triangle_contact_bary.shape[0],
            device=self.device,
        )

    def build_edge_edge_contact_geometry(self, particle_q: wp.array):
        if particle_q is not None:
            self.vertex_positions = particle_q

        wp.launch(
            kernel=build_edge_edge_contact_geometry_kernel,
            inputs=[
                self.vertex_positions,
                self.model.edge_indices,
                self.edge_colliding_edges,
                self.edge_edge_parallel_epsilon,
            ],
            outputs=[
                self.edge_edge_contact_s,
                self.edge_edge_contact_t,
                self.edge_edge_contact_normal,
                self.edge_edge_contact_dist,
            ],
            dim=self.edge_edge_contact_s.shape[0],
            device=self.device,
        )

    def build_contact_geometry(self, particle_q: wp.array) -> CollisionTriMeshVBDInfo:
        self.build_vertex_triangle_contact_geometry(particle_q)
        self.build_edge_edge_contact_geometry(particle_q)
        return self.get_collision_data()


@wp.kernel
def compute_particle_conservative_bound_vbd(
    # inputs
    conservative_bound_relaxation: float,
    collision_query_radius: float,
    adjacency: ParticleForceElementAdjacencyInfo,
    collision_info: CollisionTriMeshVBDInfo,
    # outputs
    particle_conservative_bounds: wp.array(dtype=float),
):
    particle_index = wp.tid()
    min_dist = wp.min(collision_query_radius, collision_info.vertex_colliding_triangles_min_dist[particle_index])

    # bound from neighbor triangles
    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, _vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)
        min_dist = wp.min(min_dist, collision_info.triangle_colliding_vertices_min_dist[tri_index])

    # bound from neighbor edges
    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency,
            particle_index,
            i_adj_edge,
        )
        # vertex is on the edge; otherwise it only effects the bending energy
        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            # collisions of neighbor edges
            min_dist = wp.min(min_dist, collision_info.edge_colliding_edges_min_dist[nei_edge_index])

    particle_conservative_bounds[particle_index] = conservative_bound_relaxation * min_dist


@wp.kernel
def accumulate_contact_force_and_hessian_from_geometry(
    # inputs
    dt: float,
    current_color: int,
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    # self contact
    collision_info_array: wp.array(dtype=CollisionTriMeshVBDInfo),
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    # body-particle contact
    particle_radius: wp.array(dtype=float),
    body_particle_contact_particle: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
    body_particle_contact_max: int,
    # per-contact soft AVBD parameters for body-particle contacts (shared with rigid side)
    body_particle_contact_penalty_k: wp.array(dtype=float),
    body_particle_contact_material_kd: wp.array(dtype=float),
    body_particle_contact_material_mu: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # outputs: particle force and hessian
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process edge-edge collisions
    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
        while collision_buffer_counter < collision_info.edge_colliding_edges_buffer_sizes[primitive_id]:
            slot = collision_buffer_offset + collision_buffer_counter
            e2_idx = collision_info.edge_colliding_edges[2 * slot + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]

                c_e1_v1 = particle_colors[e1_v1]
                c_e1_v2 = particle_colors[e1_v2]
                if c_e1_v1 == current_color or c_e1_v2 == current_color:
                    has_contact, collision_force_0, collision_force_1, collision_hessian_0, collision_hessian_1 = (
                        evaluate_edge_edge_contact_2_vertices_cached(
                            e1_idx,
                            e2_idx,
                            pos,
                            pos_anchor,
                            edge_indices,
                            collision_radius,
                            soft_contact_ke,
                            soft_contact_kd,
                            friction_mu,
                            friction_epsilon,
                            dt,
                            collision_info.edge_edge_contact_s[slot],
                            collision_info.edge_edge_contact_t[slot],
                            collision_info.edge_edge_contact_normal[slot],
                            collision_info.edge_edge_contact_dist[slot],
                        )
                    )

                    if has_contact:
                        # here we only handle the e1 side, because e2 will also detection this contact and add force and hessian on its own
                        if c_e1_v1 == current_color:
                            wp.atomic_add(particle_forces, e1_v1, collision_force_0)
                            wp.atomic_add(particle_hessians, e1_v1, collision_hessian_0)
                        if c_e1_v2 == current_color:
                            wp.atomic_add(particle_forces, e1_v2, collision_force_1)
                            wp.atomic_add(particle_hessians, e1_v2, collision_hessian_1)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process vertex-triangle collisions
    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id
        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
        while collision_buffer_counter < collision_info.vertex_colliding_triangles_buffer_sizes[primitive_id]:
            slot = collision_buffer_offset + collision_buffer_counter
            tri_idx = collision_info.vertex_colliding_triangles[slot * 2 + 1]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                c_v = particle_colors[particle_idx]
                c_tri_a = particle_colors[tri_a]
                c_tri_b = particle_colors[tri_b]
                c_tri_c = particle_colors[tri_c]

                if (
                    c_v == current_color
                    or c_tri_a == current_color
                    or c_tri_b == current_color
                    or c_tri_c == current_color
                ):
                    (
                        has_contact,
                        collision_force_0,
                        collision_force_1,
                        collision_force_2,
                        collision_force_3,
                        collision_hessian_0,
                        collision_hessian_1,
                        collision_hessian_2,
                        collision_hessian_3,
                    ) = evaluate_vertex_triangle_collision_force_hessian_4_vertices_cached(
                        particle_idx,
                        tri_idx,
                        pos,
                        pos_anchor,
                        tri_indices,
                        collision_radius,
                        soft_contact_ke,
                        soft_contact_kd,
                        friction_mu,
                        friction_epsilon,
                        dt,
                        collision_info.vertex_triangle_contact_bary[slot],
                        collision_info.vertex_triangle_contact_normal[slot],
                        collision_info.vertex_triangle_contact_dist[slot],
                    )

                    if has_contact:
                        # particle
                        if c_v == current_color:
                            wp.atomic_add(particle_forces, particle_idx, collision_force_3)
                            wp.atomic_add(particle_hessians, particle_idx, collision_hessian_3)

                        # tri_a
                        if c_tri_a == current_color:
                            wp.atomic_add(particle_forces, tri_a, collision_force_0)
                            wp.atomic_add(particle_hessians, tri_a, collision_hessian_0)

                        # tri_b
                        if c_tri_b == current_color:
                            wp.atomic_add(particle_forces, tri_b, collision_force_1)
                            wp.atomic_add(particle_hessians, tri_b, collision_hessian_1)

                        # tri_c
                        if c_tri_c == current_color:
                            wp.atomic_add(particle_forces, tri_c, collision_force_2)
                            wp.atomic_add(particle_hessians, tri_c, collision_hessian_2)
            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    particle_body_contact_count = min(body_particle_contact_max, body_particle_contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = body_particle_contact_particle[t_id]

        if particle_colors[particle_idx] == current_color:
            # Read per-contact AVBD penalty and material properties shared with the rigid side
            contact_ke = body_particle_contact_penalty_k[t_id]
            contact_kd = body_particle_contact_material_kd[t_id]
            contact_mu = body_particle_contact_material_mu[t_id]

            body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
                particle_idx,
                pos[particle_idx],
                pos_anchor[particle_idx],
                t_id,
                contact_ke,
                contact_kd,
                contact_mu,
                friction_epsilon,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                contact_shape,
                contact_body_pos,
                contact_body_vel,
                contact_normal,
                dt,
            )
            wp.atomic_add(particle_forces, particle_idx, body_contact_force)
            wp.atomic_add(particle_hessians, particle_idx, body_contact_hessian)


@wp.kernel
def apply_planar_truncation_parallel_by_collision_vbd(
    # inputs
    pos: wp.array(dtype=wp.vec3),
    displacement_in: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_info_array: wp.array(dtype=CollisionTriMeshVBDInfo),
    parallel_eps: float,
    gamma: float,
    truncation_t_out: wp.array(dtype=float),
):
    t_id = wp.tid()
    collision_info = collision_info_array[0]

    primitive_id = t_id // NUM_THREADS_PER_COLLISION_PRIMITIVE
    t_id_current_primitive = t_id % NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process edge-edge collisions
    if primitive_id < collision_info.edge_colliding_edges_buffer_sizes.shape[0]:
        e1_idx = primitive_id

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.edge_colliding_edges_offsets[primitive_id]
        while collision_buffer_counter < collision_info.edge_colliding_edges_buffer_sizes[primitive_id]:
            e2_idx = collision_info.edge_colliding_edges[2 * (collision_buffer_offset + collision_buffer_counter) + 1]

            if e1_idx != -1 and e2_idx != -1:
                e1_v1 = edge_indices[e1_idx, 2]
                e1_v2 = edge_indices[e1_idx, 3]

                e1_v1_pos = pos[e1_v1]
                e1_v2_pos = pos[e1_v2]

                delta_e1_v1 = displacement_in[e1_v1]
                delta_e1_v2 = displacement_in[e1_v2]

                e2_v1 = edge_indices[e2_idx, 2]
                e2_v2 = edge_indices[e2_idx, 3]

                e2_v1_pos = pos[e2_v1]
                e2_v2_pos = pos[e2_v2]

                delta_e2_v1 = displacement_in[e2_v1]
                delta_e2_v2 = displacement_in[e2_v2]

                is_dummy, n, d = create_edge_edge_division_plane_closest_pt(
                    e1_v1_pos,
                    delta_e1_v1,
                    e1_v2_pos,
                    delta_e1_v2,
                    e2_v1_pos,
                    delta_e2_v1,
                    e2_v2_pos,
                    delta_e2_v2,
                )

                if not is_dummy[0]:
                    t = planar_truncation_t(e1_v1_pos, delta_e1_v1, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e1_v1, t)
                if not is_dummy[1]:
                    t = planar_truncation_t(e1_v2_pos, delta_e1_v2, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e1_v2, t)
                if not is_dummy[2]:
                    t = planar_truncation_t(e2_v1_pos, delta_e2_v1, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e2_v1, t)
                if not is_dummy[3]:
                    t = planar_truncation_t(e2_v2_pos, delta_e2_v2, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, e2_v2, t)

            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE

    # process vertex-triangle collisions
    if primitive_id < collision_info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        particle_idx = primitive_id

        colliding_particle_pos = pos[particle_idx]
        colliding_particle_displacement = displacement_in[particle_idx]

        collision_buffer_counter = t_id_current_primitive
        collision_buffer_offset = collision_info.vertex_colliding_triangles_offsets[primitive_id]
        while collision_buffer_counter < collision_info.vertex_colliding_triangles_buffer_sizes[primitive_id]:
            tri_idx = collision_info.vertex_colliding_triangles[
                (collision_buffer_offset + collision_buffer_counter) * 2 + 1
            ]

            if particle_idx != -1 and tri_idx != -1:
                tri_a = tri_indices[tri_idx, 0]
                tri_b = tri_indices[tri_idx, 1]
                tri_c = tri_indices[tri_idx, 2]

                t1 = pos[tri_a]
                t2 = pos[tri_b]
                t3 = pos[tri_c]
                delta_t1 = displacement_in[tri_a]
                delta_t2 = displacement_in[tri_b]
                delta_t3 = displacement_in[tri_c]

                is_dummy, n, d = create_vertex_triangle_division_plane_closest_pt(
                    colliding_particle_pos,
                    colliding_particle_displacement,
                    t1,
                    delta_t1,
                    t2,
                    delta_t2,
                    t3,
                    delta_t3,
                )

                if not is_dummy[0]:
                    t = planar_truncation_t(
                        colliding_particle_pos, colliding_particle_displacement, n, d, parallel_eps, gamma
                    )
                    wp.atomic_min(truncation_t_out, particle_idx, t)
                if not is_dummy[1]:
                    t = planar_truncation_t(t1, delta_t1, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_a, t)
                if not is_dummy[2]:
                    t = planar_truncation_t(t2, delta_t2, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_b, t)
                if not is_dummy[3]:
                    t = planar_truncation_t(t3, delta_t3, n, d, parallel_eps, gamma)
                    wp.atomic_min(truncation_t_out, tri_c, t)

            collision_buffer_counter += NUM_THREADS_PER_COLLISION_PRIMITIVE
