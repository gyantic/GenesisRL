import sys
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

import taichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.styles import colors, formats
from genesis.utils.misc import ti_field_to_torch

from .mpr_decomp import MPR
from .gjk_decomp import GJK

from enum import IntEnum

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


class CCD_ALGORITHM_CODE(IntEnum):
    MPR = 0
    MPR_SDF = 1
    GJK = 2


@ti.func
def rotaxis(vecin, i0, i1, i2, f0, f1, f2):
    vecres = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
    vecres[0] = vecin[i0] * f0
    vecres[1] = vecin[i1] * f1
    vecres[2] = vecin[i2] * f2
    return vecres


@ti.func
def rotmatx(matin, i0, i1, i2, f0, f1, f2):
    matres = ti.Matrix.zero(gs.ti_float, 3, 3)
    matres[0, :] = matin[i0, :] * f0
    matres[1, :] = matin[i1, :] * f1
    matres[2, :] = matin[i2, :] * f2
    return matres


@ti.data_oriented
class Collider:
    def __init__(self, rigid_solver: "RigidSolver"):
        self._solver = rigid_solver
        self._init_verts_connectivity()
        self._init_collision_fields()

        # Identify the convex collision detection (ccd) algorithm
        if self._solver._options.use_gjk_collision:
            self.ccd_algorithm = CCD_ALGORITHM_CODE.GJK
        elif self._solver._enable_mujoco_compatibility:
            self.ccd_algorithm = CCD_ALGORITHM_CODE.MPR
        else:
            self.ccd_algorithm = CCD_ALGORITHM_CODE.MPR_SDF

        # FIXME: MPR is necessary because it is used for terrain collision detection
        self._mpr = MPR(rigid_solver)
        self._gjk = GJK(rigid_solver) if self.ccd_algorithm == CCD_ALGORITHM_CODE.GJK else None

        # multi contact perturbation and tolerance
        if self._solver._enable_mujoco_compatibility:
            self._mc_perturbation = 1e-3
            self._mc_tolerance = 1e-3
        else:
            self._mc_perturbation = 1e-2
            self._mc_tolerance = 1e-2
        self._mpr_to_sdf_overlap_ratio = 0.4

    def _init_verts_connectivity(self) -> None:
        vert_neighbors = []
        vert_neighbor_start = []
        vert_n_neighbors = []
        offset = 0
        for geom in self._solver.geoms:
            vert_neighbors.append(geom.vert_neighbors + geom.vert_start)
            vert_neighbor_start.append(geom.vert_neighbor_start + offset)
            vert_n_neighbors.append(geom.vert_n_neighbors)
            offset += len(geom.vert_neighbors)

        if self._solver.n_verts > 0:
            vert_neighbors = np.concatenate(vert_neighbors, dtype=gs.np_int)
            vert_neighbor_start = np.concatenate(vert_neighbor_start, dtype=gs.np_int)
            vert_n_neighbors = np.concatenate(vert_n_neighbors, dtype=gs.np_int)

        self.vert_neighbors = ti.field(dtype=gs.ti_int, shape=max(1, len(vert_neighbors)))
        self.vert_neighbor_start = ti.field(dtype=gs.ti_int, shape=self._solver.n_verts_)
        self.vert_n_neighbors = ti.field(dtype=gs.ti_int, shape=self._solver.n_verts_)

        if self._solver.n_verts > 0:
            self.vert_neighbors.from_numpy(vert_neighbors)
            self.vert_neighbor_start.from_numpy(vert_neighbor_start)
            self.vert_n_neighbors.from_numpy(vert_n_neighbors)

    def _init_collision_fields(self) -> None:
        # Compute collision pair validity
        self.collision_pair_validity = ti.field(gs.ti_int, shape=(self._solver.n_geoms_, self._solver.n_geoms_))

        geoms_link_idx = self._solver.geoms_info.link_idx.to_numpy()
        geoms_contype = self._solver.geoms_info.contype.to_numpy()
        geoms_conaffinity = self._solver.geoms_info.conaffinity.to_numpy()
        links_entity_idx = self._solver.links_info.entity_idx.to_numpy()
        links_root_idx = self._solver.links_info.root_idx.to_numpy()
        links_parent_idx = self._solver.links_info.parent_idx.to_numpy()
        links_is_fixed = self._solver.links_info.is_fixed.to_numpy()
        if self._solver._options.batch_links_info:
            links_entity_idx = links_entity_idx[:, 0]
            links_root_idx = links_root_idx[:, 0]
            links_parent_idx = links_parent_idx[:, 0]
            links_is_fixed = links_is_fixed[:, 0]
        n_possible_pairs = 0
        for i_ga in range(self._solver.n_geoms):
            for i_gb in range(i_ga + 1, self._solver.n_geoms):
                i_la = geoms_link_idx[i_ga]
                i_lb = geoms_link_idx[i_gb]

                # geoms in the same link
                if i_la == i_lb:
                    continue

                # self collision
                if links_root_idx[i_la] == links_root_idx[i_lb]:
                    if not self._solver._enable_self_collision:
                        continue

                    # adjacent links
                    if not self._solver._enable_adjacent_collision and (
                        links_parent_idx[i_la] == i_lb or links_parent_idx[i_lb] == i_la
                    ):
                        continue

                # contype and conaffinity
                if links_entity_idx[i_la] == links_entity_idx[i_lb] and not (
                    (geoms_contype[i_ga] & geoms_conaffinity[i_gb]) or (geoms_contype[i_gb] & geoms_conaffinity[i_ga])
                ):
                    continue

                # pair of fixed links wrt the world
                if links_is_fixed[i_la] and links_is_fixed[i_lb]:
                    continue

                self.collision_pair_validity[i_ga, i_gb] = 1
                n_possible_pairs += 1

        self._n_contacts_per_pair = 5  # CONSTANT. CANNOT NOT BE CHANGED.
        self._max_possible_pairs = n_possible_pairs
        self._max_collision_pairs = min(n_possible_pairs, self._solver._max_collision_pairs)
        self._max_contact_pairs = self._max_collision_pairs * self._n_contacts_per_pair

        self._warn_msg_max_collision_pairs = (
            f"{colors.YELLOW}[Genesis] [00:00:00] [WARNING] Ignoring contact pair to avoid exceeding max "
            f"({self._max_contact_pairs}). Please increase the value of RigidSolver's option "
            f"'max_collision_pairs'.{formats.RESET}"
        )

        ############## broad phase SAP ##############
        # This buffer stores the AABBs along the search axis of all geoms
        struct_sort_buffer = ti.types.struct(value=gs.ti_float, i_g=gs.ti_int, is_max=gs.ti_int)
        self.sort_buffer = struct_sort_buffer.field(
            shape=self._solver._batch_shape(2 * self._solver.n_geoms_),
            layout=ti.Layout.SOA,
        )
        # This buffer stores indexes of active geoms during SAP search
        if self._solver._use_hibernation:
            self.active_buffer_awake = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape(self._solver.n_geoms_))
            self.active_buffer_hib = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape(self._solver.n_geoms_))
        self.active_buffer = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape(self._solver.n_geoms_))

        self.n_broad_pairs = ti.field(dtype=gs.ti_int, shape=self._solver._B)
        self.broad_collision_pairs = ti.Vector.field(
            2, dtype=gs.ti_int, shape=self._solver._batch_shape(max(1, self._max_collision_pairs))
        )

        self.first_time = ti.field(gs.ti_int, shape=self._solver._B)

        ############## narrow phase ##############
        struct_contact_data = ti.types.struct(
            geom_a=gs.ti_int,
            geom_b=gs.ti_int,
            penetration=gs.ti_float,
            normal=gs.ti_vec3,
            pos=gs.ti_vec3,
            friction=gs.ti_float,
            sol_params=gs.ti_vec7,
            force=gs.ti_vec3,
            link_a=gs.ti_int,
            link_b=gs.ti_int,
        )
        self.contact_data = struct_contact_data.field(
            shape=self._solver._batch_shape(max(1, self._max_contact_pairs)),
            layout=ti.Layout.SOA,
        )
        self.n_contacts = ti.field(
            gs.ti_int, shape=self._solver._B
        )  # total number of contacts, including hibernated contacts
        self.n_contacts_hibernated = ti.field(gs.ti_int, shape=self._solver._B)
        self._contacts_info_cache = {}

        # contact caching for warmstart collision detection
        struct_contact_cache = ti.types.struct(
            # i_va_ws=gs.ti_int,
            # penetration=gs.ti_float,
            normal=gs.ti_vec3,
        )
        self.contact_cache = struct_contact_cache.field(
            shape=self._solver._batch_shape((self._solver.n_geoms_, self._solver.n_geoms_)),
            layout=ti.Layout.SOA,
        )

        # for faster compilation
        self._has_nonconvex_nonterrain = np.logical_and(
            self._solver.geoms_info.is_convex.to_numpy() == 0,
            self._solver.geoms_info.type.to_numpy() != gs.GEOM_TYPE.TERRAIN,
        ).any()
        self._has_terrain = (self._solver.geoms_info.type.to_numpy() == gs.GEOM_TYPE.TERRAIN).any()

        if self._has_terrain:
            self.xyz_max_min = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(6))
            self.prism = ti.field(dtype=gs.ti_vec3, shape=self._solver._batch_shape(6))

        ##---------------- box box
        if self._solver._box_box_detection:
            # With the existing Box-Box collision detection algorithm, it is not clear where the contact points are
            # located depending of the pose and size of each box. In practice, up to 11 contact points have been
            # observed. The theoretical worst case scenario would be 2 cubes roughly the same size and same center,
            # with transform RPY = (45, 45, 45), resulting in 3 contact points per faces for a total of 16 points.
            self.box_MAXCONPAIR = 16
            self.box_depth = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.box_MAXCONPAIR))
            self.box_points = ti.field(gs.ti_vec3, shape=self._solver._batch_shape(self.box_MAXCONPAIR))
            self.box_pts = ti.field(gs.ti_vec3, shape=self._solver._batch_shape(6))
            self.box_lines = ti.field(gs.ti_vec6, shape=self._solver._batch_shape(4))
            self.box_linesu = ti.field(gs.ti_vec6, shape=self._solver._batch_shape(4))
            self.box_axi = ti.field(gs.ti_vec3, shape=self._solver._batch_shape(3))
            self.box_ppts2 = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape((4, 2)))
            self.box_pu = ti.field(gs.ti_vec3, shape=self._solver._batch_shape(4))
        ##---------------- box box

        self.reset()

    def reset(self, envs_idx: npt.NDArray[np.int32] | None = None) -> None:
        if envs_idx is None:
            envs_idx = self._solver._scene._envs_idx
        self._kernel_reset(envs_idx)
        self._contacts_info_cache = {}

    @ti.kernel
    def _kernel_reset(
        self,
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            self.first_time[i_b] = 1
            for i_ga in range(self._solver.n_geoms):
                for i_gb in range(self._solver.n_geoms):
                    # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = -1
                    # self.contact_cache[i_ga, i_gb, i_b].penetration = 0.0
                    self.contact_cache[i_ga, i_gb, i_b].normal.fill(0.0)

    def clear(self, envs_idx=None):
        if envs_idx is None:
            envs_idx = self._solver._scene._envs_idx
        self._kernel_clear(envs_idx)

    @ti.kernel
    def _kernel_clear(
        self,
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            if ti.static(self._solver._use_hibernation):
                self.n_contacts_hibernated[i_b] = 0

                # advect hibernated contacts
                for i_c in range(self.n_contacts[i_b]):
                    i_la = self.contact_data[i_c, i_b].link_a
                    i_lb = self.contact_data[i_c, i_b].link_b

                    I_la = [i_la, i_b] if ti.static(self._solver._options.batch_links_info) else i_la
                    I_lb = [i_lb, i_b] if ti.static(self._solver._options.batch_links_info) else i_lb

                    # pair of hibernated-fixed links -> hibernated contact
                    # TODO: we should also include hibernated-hibernated links and wake up the whole contact island
                    # once a new collision is detected
                    if (self._solver.links_state[i_la, i_b].hibernated and self._solver.links_info[I_lb].is_fixed) or (
                        self._solver.links_state[i_lb, i_b].hibernated and self._solver.links_info[I_la].is_fixed
                    ):
                        i_c_hibernated = self.n_contacts_hibernated[i_b]
                        if i_c != i_c_hibernated:
                            self.contact_data[i_c_hibernated, i_b] = self.contact_data[i_c, i_b]
                        self.n_contacts_hibernated[i_b] = i_c_hibernated + 1

                self.n_contacts[i_b] = self.n_contacts_hibernated[i_b]
            else:
                self.n_contacts[i_b] = 0

    def detection(self) -> None:
        # from genesis.utils.tools import create_timer

        self._contacts_info_cache = {}
        # timer = create_timer(name="69477ab0-5e75-47cb-a4a5-d4eebd9336ca", level=3, ti_sync=True, skip_first_call=True)
        self._func_update_aabbs()
        # timer.stamp("func_update_aabbs")
        self._func_broad_phase()
        # timer.stamp("func_broad_phase")
        self._func_narrow_phase_convex_vs_convex()
        self._func_narrow_phase_convex_specializations()
        # timer.stamp("func_narrow_phase")
        if self._has_terrain:
            self._func_narrow_phase_any_vs_terrain()
            # timer.stamp("_func_narrow_phase_any_vs_terrain")
        if self._has_nonconvex_nonterrain:
            self._func_narrow_phase_nonconvex_vs_nonterrain()
            # timer.stamp("_func_narrow_phase_nonconvex_vs_nonterrain")

    @ti.func
    def _func_point_in_geom_aabb(self, point, i_g, i_b):
        return (point < self._solver.geoms_state[i_g, i_b].aabb_max).all() and (
            point > self._solver.geoms_state[i_g, i_b].aabb_min
        ).all()

    @ti.func
    def _func_is_geom_aabbs_overlap(self, i_ga, i_gb, i_b):
        return not (
            (self._solver.geoms_state[i_ga, i_b].aabb_max <= self._solver.geoms_state[i_gb, i_b].aabb_min).any()
            or (self._solver.geoms_state[i_ga, i_b].aabb_min >= self._solver.geoms_state[i_gb, i_b].aabb_max).any()
        )

    @ti.func
    def _func_find_intersect_midpoint(self, i_ga, i_gb):
        # return the center of the intersecting AABB of AABBs of two geoms
        intersect_lower = ti.max(self._solver.geoms_state[i_ga].aabb_min, self._solver.geoms_state[i_gb].aabb_min)
        intersect_upper = ti.min(self._solver.geoms_state[i_ga].aabb_max, self._solver.geoms_state[i_gb].aabb_max)
        return 0.5 * (intersect_lower + intersect_upper)

    @ti.func
    def _func_contact_sphere_sdf(self, i_ga, i_gb, i_b):
        ga_info = self._solver.geoms_info[i_ga]
        is_col = False
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        sphere_center = self._solver.geoms_state[i_ga, i_b].pos
        sphere_radius = ga_info.data[0]

        center_to_b_dist = self._solver.sdf.sdf_world(sphere_center, i_gb, i_b)
        if center_to_b_dist < sphere_radius:
            is_col = True
            normal = self._solver.sdf.sdf_normal_world(sphere_center, i_gb, i_b)
            penetration = sphere_radius - center_to_b_dist
            contact_pos = sphere_center - (sphere_radius - 0.5 * penetration) * normal

        return is_col, normal, penetration, contact_pos

    @ti.func
    def _func_contact_vertex_sdf(self, i_ga, i_gb, i_b):
        ga_info = self._solver.geoms_info[i_ga]
        ga_pos = self._solver.geoms_state[i_ga, i_b].pos
        ga_quat = self._solver.geoms_state[i_ga, i_b].quat

        is_col = False
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        for i_v in range(ga_info.vert_start, ga_info.vert_end):
            vertex_pos = gu.ti_transform_by_trans_quat(self._solver.verts_info[i_v].init_pos, ga_pos, ga_quat)
            if self._func_point_in_geom_aabb(vertex_pos, i_gb, i_b):
                new_penetration = -self._solver.sdf.sdf_world(vertex_pos, i_gb, i_b)
                if new_penetration > penetration:
                    is_col = True
                    contact_pos = vertex_pos
                    penetration = new_penetration

        # Compute contact normal only once, and only in case of contact
        if is_col:
            normal = self._solver.sdf.sdf_normal_world(contact_pos, i_gb, i_b)

        # The contact point must be offsetted by half the penetration depth
        contact_pos += 0.5 * penetration * normal

        return is_col, normal, penetration, contact_pos

    @ti.func
    def _func_contact_edge_sdf(self, i_ga, i_gb, i_b):
        ga_info = self._solver.geoms_info[i_ga]
        ga_state = self._solver.geoms_state[i_ga, i_b]

        is_col = False
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        ga_sdf_cell_size = self._solver.sdf.geoms_info[i_ga].sdf_cell_size

        for i_e in range(ga_info.edge_start, ga_info.edge_end):
            cur_length = self._solver.edges_info[i_e].length
            if cur_length > ga_sdf_cell_size:

                i_v0 = self._solver.edges_info[i_e].v0
                i_v1 = self._solver.edges_info[i_e].v1

                p_0 = gu.ti_transform_by_trans_quat(self._solver.verts_info[i_v0].init_pos, ga_state.pos, ga_state.quat)
                p_1 = gu.ti_transform_by_trans_quat(self._solver.verts_info[i_v1].init_pos, ga_state.pos, ga_state.quat)
                vec_01 = gu.ti_normalize(p_1 - p_0)

                sdf_grad_0_b = self._solver.sdf.sdf_grad_world(p_0, i_gb, i_b)
                sdf_grad_1_b = self._solver.sdf.sdf_grad_world(p_1, i_gb, i_b)

                # check if the edge on a is facing towards mesh b (I am not 100% sure about this, subject to removal)
                sdf_grad_0_a = self._solver.sdf.sdf_grad_world(p_0, i_ga, i_b)
                sdf_grad_1_a = self._solver.sdf.sdf_grad_world(p_1, i_ga, i_b)
                normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
                normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

                if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:

                    # check if closest point is between the two points
                    if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:

                        while cur_length > ga_sdf_cell_size:
                            p_mid = 0.5 * (p_0 + p_1)
                            if self._solver.sdf.sdf_grad_world(p_mid, i_gb, i_b).dot(vec_01) < 0:
                                p_0 = p_mid
                            else:
                                p_1 = p_mid
                            cur_length = 0.5 * cur_length

                        p = 0.5 * (p_0 + p_1)
                        new_penetration = -self._solver.sdf.sdf_world(p, i_gb, i_b)

                        if new_penetration > penetration:
                            is_col = True
                            normal = self._solver.sdf.sdf_normal_world(p, i_gb, i_b)
                            contact_pos = p
                            penetration = new_penetration

        # The contact point must be offsetted by half the penetration depth, for consistency with MPR
        contact_pos += 0.5 * penetration * normal

        return is_col, normal, penetration, contact_pos

    @ti.func
    def _func_contact_convex_convex_sdf(self, i_ga, i_gb, i_b, i_va_ws):
        gb_vert_start = self._solver.geoms_info[i_gb].vert_start
        ga_pos = self._solver.geoms_state[i_ga, i_b].pos
        ga_quat = self._solver.geoms_state[i_ga, i_b].quat
        gb_pos = self._solver.geoms_state[i_gb, i_b].pos
        gb_quat = self._solver.geoms_state[i_gb, i_b].quat

        is_col = False
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        i_va = i_va_ws
        if i_va == -1:
            # start traversing on the vertex graph with a smart initial vertex
            pos_vb = gu.ti_transform_by_trans_quat(self._solver.verts_info[gb_vert_start].init_pos, gb_pos, gb_quat)
            i_va = self._solver.sdf._func_find_closest_vert(pos_vb, i_ga, i_b)
        i_v_closest = i_va
        pos_v_closest = gu.ti_transform_by_trans_quat(self._solver.verts_info[i_v_closest].init_pos, ga_pos, ga_quat)
        sd_v_closest = self._solver.sdf.sdf_world(pos_v_closest, i_gb, i_b)

        while True:
            for i_neighbor_ in range(
                self.vert_neighbor_start[i_va], self.vert_neighbor_start[i_va] + self.vert_n_neighbors[i_va]
            ):
                i_neighbor = self.vert_neighbors[i_neighbor_]
                pos_neighbor = gu.ti_transform_by_trans_quat(
                    self._solver.verts_info[i_neighbor].init_pos, ga_pos, ga_quat
                )
                sd_neighbor = self._solver.sdf.sdf_world(pos_neighbor, i_gb, i_b)
                if (
                    sd_neighbor < sd_v_closest - 1e-5
                ):  # 1e-5 (0.01mm) to avoid endless loop due to numerical instability
                    i_v_closest = i_neighbor
                    sd_v_closest = sd_neighbor
                    pos_v_closest = pos_neighbor

            if i_v_closest == i_va:  # no better neighbor
                break
            else:
                i_va = i_v_closest

        # i_va is the deepest vertex
        pos_a = pos_v_closest
        if sd_v_closest < 0:
            is_col = True
            normal = self._solver.sdf.sdf_normal_world(pos_a, i_gb, i_b)
            contact_pos = pos_a
            penetration = -sd_v_closest

        else:  # check edge surrounding it
            for i_neighbor_ in range(
                self.vert_neighbor_start[i_va], self.vert_neighbor_start[i_va] + self.vert_n_neighbors[i_va]
            ):
                i_neighbor = self.vert_neighbors[i_neighbor_]

                p_0 = pos_v_closest
                p_1 = gu.ti_transform_by_trans_quat(self._solver.verts_info[i_neighbor].init_pos, ga_pos, ga_quat)
                vec_01 = gu.ti_normalize(p_1 - p_0)

                sdf_grad_0_b = self._solver.sdf.sdf_grad_world(p_0, i_gb, i_b)
                sdf_grad_1_b = self._solver.sdf.sdf_grad_world(p_1, i_gb, i_b)

                # check if the edge on a is facing towards mesh b (I am not 100% sure about this, subject to removal)
                sdf_grad_0_a = self._solver.sdf.sdf_grad_world(p_0, i_ga, i_b)
                sdf_grad_1_a = self._solver.sdf.sdf_grad_world(p_1, i_ga, i_b)
                normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
                normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

                if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:
                    # check if closest point is between the two points
                    if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:
                        cur_length = (p_1 - p_0).norm()
                        ga_sdf_cell_size = self._solver.sdf.geoms_info[i_ga].sdf_cell_size
                        while cur_length > ga_sdf_cell_size:
                            p_mid = 0.5 * (p_0 + p_1)
                            if self._solver.sdf.sdf_grad_world(p_mid, i_gb, i_b).dot(vec_01) < 0:
                                p_0 = p_mid
                            else:
                                p_1 = p_mid

                            cur_length = 0.5 * cur_length

                        p = 0.5 * (p_0 + p_1)

                        new_penetration = -self._solver.sdf.sdf_world(p, i_gb, i_b)

                        if new_penetration > 0:
                            is_col = True
                            normal = self._solver.sdf.sdf_normal_world(p, i_gb, i_b)
                            contact_pos = p
                            penetration = new_penetration
                            break

        return is_col, normal, penetration, contact_pos, i_va

    @ti.func
    def _func_contact_mpr_terrain(self, i_ga, i_gb, i_b):
        ga_pos, ga_quat = self._solver.geoms_state[i_ga, i_b].pos, self._solver.geoms_state[i_ga, i_b].quat
        gb_pos, gb_quat = self._solver.geoms_state[i_gb, i_b].pos, self._solver.geoms_state[i_gb, i_b].quat
        margin = gs.ti_float(0.0)

        is_return = False
        tolerance = self._func_compute_tolerance(i_ga, i_gb, i_b)
        # pos = self._solver.geoms_state[i_ga, i_b].pos - self._solver.geoms_state[i_gb, i_b].pos
        # for i in range(3):
        #     if self._solver.terrain_xyz_maxmin[i] < pos[i] - r2 - margin or \
        #         self._solver.terrain_xyz_maxmin[i+3] > pos[i] + r2 + margin:
        #         is_return = True

        if not is_return:
            self._solver.geoms_state[i_ga, i_b].pos, self._solver.geoms_state[i_ga, i_b].quat = (
                gu.ti_transform_pos_quat_by_trans_quat(
                    ga_pos - self._solver.geoms_state[i_gb, i_b].pos,
                    ga_quat,
                    ti.Vector.zero(gs.ti_float, 3),
                    gu.ti_inv_quat(self._solver.geoms_state[i_gb, i_b].quat),
                )
            )

            for i_axis, i_m in ti.ndrange(3, 2):
                direction = ti.Vector.zero(gs.ti_float, 3)
                if i_m == 0:
                    direction[i_axis] = 1.0
                else:
                    direction[i_axis] = -1.0
                v1 = self._mpr.support_driver(direction, i_ga, i_b)
                self.xyz_max_min[3 * i_m + i_axis, i_b] = v1[i_axis]

            for i in ti.static(range(3)):
                self.prism[i, i_b][2] = self._solver.terrain_xyz_maxmin[5]

                if (
                    self._solver.terrain_xyz_maxmin[i] < self.xyz_max_min[i + 3, i_b] - margin
                    or self._solver.terrain_xyz_maxmin[i + 3] > self.xyz_max_min[i, i_b] + margin
                ):
                    is_return = True

            if not is_return:
                sh = self._solver.terrain_scale[0]
                r_min = gs.ti_int(ti.floor((self.xyz_max_min[3, i_b] - self._solver.terrain_xyz_maxmin[3]) / sh))
                r_max = gs.ti_int(ti.ceil((self.xyz_max_min[0, i_b] - self._solver.terrain_xyz_maxmin[3]) / sh))
                c_min = gs.ti_int(ti.floor((self.xyz_max_min[4, i_b] - self._solver.terrain_xyz_maxmin[4]) / sh))
                c_max = gs.ti_int(ti.ceil((self.xyz_max_min[1, i_b] - self._solver.terrain_xyz_maxmin[4]) / sh))

                r_min = ti.max(0, r_min)
                c_min = ti.max(0, c_min)
                r_max = ti.min(self._solver.terrain_rc[0] - 1, r_max)
                c_max = ti.min(self._solver.terrain_rc[1] - 1, c_max)

                cnt = 0
                for r in range(r_min, r_max):
                    nvert = 0
                    for c in range(c_min, c_max + 1):
                        for i in range(2):
                            if cnt < self._n_contacts_per_pair:
                                nvert = nvert + 1
                                self.add_prism_vert(
                                    sh * (r + i) + self._solver.terrain_xyz_maxmin[3],
                                    sh * c + self._solver.terrain_xyz_maxmin[4],
                                    self._solver.terrain_hf[r + i, c] + margin,
                                    i_b,
                                )
                                if nvert > 2 and (
                                    self.prism[3, i_b][2] >= self.xyz_max_min[5, i_b]
                                    or self.prism[4, i_b][2] >= self.xyz_max_min[5, i_b]
                                    or self.prism[5, i_b][2] >= self.xyz_max_min[5, i_b]
                                ):
                                    center_a = gu.ti_transform_by_trans_quat(
                                        self._solver.geoms_info[i_ga].center, ga_pos, ga_quat
                                    )
                                    center_b = ti.Vector.zero(gs.ti_float, 3)
                                    for i_p in ti.static(range(6)):
                                        center_b = center_b + self.prism[i_p, i_b]
                                    center_b = center_b / 6.0

                                    self._solver.geoms_state[i_gb, i_b].pos = ti.Vector.zero(gs.ti_float, 3)
                                    self._solver.geoms_state[i_gb, i_b].quat = gu.ti_identity_quat()

                                    is_col, normal, penetration, contact_pos = self._mpr.func_mpr_contact_from_centers(
                                        i_ga, i_gb, i_b, center_a, center_b
                                    )
                                    if is_col:
                                        normal = gu.ti_transform_by_quat(normal, gb_quat)
                                        contact_pos = gu.ti_transform_by_quat(contact_pos, gb_quat)
                                        contact_pos = contact_pos + gb_pos

                                        valid = True
                                        i_col = self.n_contacts[i_b]
                                        for j in range(cnt):
                                            if (
                                                contact_pos - self.contact_data[i_col - j - 1, i_b].pos
                                            ).norm() < tolerance:
                                                valid = False
                                                break

                                        if valid:
                                            self._func_add_contact(i_ga, i_gb, normal, contact_pos, penetration, i_b)
                                            cnt = cnt + 1

        self._solver.geoms_state[i_ga, i_b].pos, self._solver.geoms_state[i_ga, i_b].quat = ga_pos, ga_quat
        self._solver.geoms_state[i_gb, i_b].pos, self._solver.geoms_state[i_gb, i_b].quat = gb_pos, gb_quat

    @ti.func
    def add_prism_vert(self, x, y, z, i_b):
        self.prism[0, i_b] = self.prism[1, i_b]
        self.prism[1, i_b] = self.prism[2, i_b]
        self.prism[3, i_b] = self.prism[4, i_b]
        self.prism[4, i_b] = self.prism[5, i_b]

        self.prism[2, i_b][0] = x
        self.prism[5, i_b][0] = x
        self.prism[2, i_b][1] = y
        self.prism[5, i_b][1] = y
        self.prism[5, i_b][2] = z

    @ti.kernel
    def _func_update_aabbs(self):
        self._solver._func_update_geom_aabbs()

    @ti.func
    def _func_check_collision_valid(self, i_ga, i_gb, i_b):
        is_valid = self.collision_pair_validity[i_ga, i_gb]

        # hibernated <-> fixed links
        if ti.static(self._solver._use_hibernation):
            i_la = self._solver.geoms_info[i_ga].link_idx
            i_lb = self._solver.geoms_info[i_gb].link_idx
            I_la = [i_la, i_b] if ti.static(self._solver._options.batch_links_info) else i_la
            I_lb = [i_lb, i_b] if ti.static(self._solver._options.batch_links_info) else i_lb

            if (self._solver.links_state[i_la, i_b].hibernated and self._solver.links_info[I_lb].is_fixed) or (
                self._solver.links_state[i_lb, i_b].hibernated and self._solver.links_info[I_la].is_fixed
            ):
                is_valid = False

        return is_valid

    @ti.kernel
    def _func_broad_phase(self):
        """
        Sweep and Prune (SAP) for broad-phase collision detection.

        This function sorts the geometry axis-aligned bounding boxes (AABBs) along a specified axis and checks for potential collision pairs based on the AABB overlap.
        """

        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._solver._B):
            axis = 0

            # copy updated geom aabbs to buffer for sorting
            if self.first_time[i_b]:
                for i in range(self._solver.n_geoms):
                    self.sort_buffer[2 * i, i_b].value = self._solver.geoms_state[i, i_b].aabb_min[axis]
                    self.sort_buffer[2 * i, i_b].i_g = i
                    self.sort_buffer[2 * i, i_b].is_max = 0

                    self.sort_buffer[2 * i + 1, i_b].value = self._solver.geoms_state[i, i_b].aabb_max[axis]
                    self.sort_buffer[2 * i + 1, i_b].i_g = i
                    self.sort_buffer[2 * i + 1, i_b].is_max = 1

                    self._solver.geoms_state[i, i_b].min_buffer_idx = 2 * i
                    self._solver.geoms_state[i, i_b].max_buffer_idx = 2 * i + 1

                self.first_time[i_b] = False

            else:
                # warm start. If `use_hibernation=True`, it's already updated in rigid_solver.
                if ti.static(not self._solver._use_hibernation):
                    for i in range(self._solver.n_geoms * 2):
                        if self.sort_buffer[i, i_b].is_max:
                            self.sort_buffer[i, i_b].value = self._solver.geoms_state[
                                self.sort_buffer[i, i_b].i_g, i_b
                            ].aabb_max[axis]
                        else:
                            self.sort_buffer[i, i_b].value = self._solver.geoms_state[
                                self.sort_buffer[i, i_b].i_g, i_b
                            ].aabb_min[axis]

            # insertion sort, which has complexity near O(n) for nearly sorted array
            for i in range(1, 2 * self._solver.n_geoms):
                key = self.sort_buffer[i, i_b]

                j = i - 1
                while j >= 0 and key.value < self.sort_buffer[j, i_b].value:
                    self.sort_buffer[j + 1, i_b] = self.sort_buffer[j, i_b]

                    if ti.static(self._solver._use_hibernation):
                        if self.sort_buffer[j, i_b].is_max:
                            self._solver.geoms_state[self.sort_buffer[j, i_b].i_g, i_b].max_buffer_idx = j + 1
                        else:
                            self._solver.geoms_state[self.sort_buffer[j, i_b].i_g, i_b].min_buffer_idx = j + 1

                    j -= 1
                self.sort_buffer[j + 1, i_b] = key

                if ti.static(self._solver._use_hibernation):
                    if key.is_max:
                        self._solver.geoms_state[key.i_g, i_b].max_buffer_idx = j + 1
                    else:
                        self._solver.geoms_state[key.i_g, i_b].min_buffer_idx = j + 1

            # sweep over the sorted AABBs to find potential collision pairs
            self.n_broad_pairs[i_b] = 0
            if ti.static(not self._solver._use_hibernation):
                n_active = 0
                for i in range(2 * self._solver.n_geoms):
                    if not self.sort_buffer[i, i_b].is_max:
                        for j in range(n_active):
                            i_ga = self.active_buffer[j, i_b]
                            i_gb = self.sort_buffer[i, i_b].i_g
                            if i_ga > i_gb:
                                i_ga, i_gb = i_gb, i_ga

                            if not self._func_check_collision_valid(i_ga, i_gb, i_b):
                                continue

                            if not self._func_is_geom_aabbs_overlap(i_ga, i_gb, i_b):
                                # Clear collision normal cache if not in contact
                                if ti.static(not self._solver._enable_mujoco_compatibility):
                                    # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = -1
                                    self.contact_cache[i_ga, i_gb, i_b].normal.fill(0.0)
                                continue

                            if self.n_broad_pairs[i_b] == self._max_collision_pairs:
                                ti.static_print(self._warn_msg_max_collision_pairs)
                                break
                            self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][0] = i_ga
                            self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][1] = i_gb
                            self.n_broad_pairs[i_b] = self.n_broad_pairs[i_b] + 1

                        self.active_buffer[n_active, i_b] = self.sort_buffer[i, i_b].i_g
                        n_active = n_active + 1
                    else:
                        i_g_to_remove = self.sort_buffer[i, i_b].i_g
                        for j in range(n_active):
                            if self.active_buffer[j, i_b] == i_g_to_remove:
                                if j < n_active - 1:
                                    for k in range(j, n_active - 1):
                                        self.active_buffer[k, i_b] = self.active_buffer[k + 1, i_b]
                                n_active = n_active - 1
                                break
            else:
                if self._solver.n_awake_dofs[i_b] > 0:
                    n_active_awake = 0
                    n_active_hib = 0
                    for i in range(2 * self._solver.n_geoms):
                        is_incoming_geom_hibernated = self._solver.geoms_state[
                            self.sort_buffer[i, i_b].i_g, i_b
                        ].hibernated

                        if not self.sort_buffer[i, i_b].is_max:
                            # both awake and hibernated geom check with active awake geoms
                            for j in range(n_active_awake):
                                i_ga = self.active_buffer_awake[j, i_b]
                                i_gb = self.sort_buffer[i, i_b].i_g
                                if i_ga > i_gb:
                                    i_ga, i_gb = i_gb, i_ga

                                if not self._func_check_collision_valid(i_ga, i_gb, i_b):
                                    continue

                                if not self._func_is_geom_aabbs_overlap(i_ga, i_gb, i_b):
                                    # Clear collision normal cache if not in contact
                                    if ti.static(not self._solver._enable_mujoco_compatibility):
                                        # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = -1
                                        self.contact_cache[i_ga, i_gb, i_b].normal.fill(0.0)
                                    continue

                                self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][0] = i_ga
                                self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][1] = i_gb
                                self.n_broad_pairs[i_b] = self.n_broad_pairs[i_b] + 1

                            # if incoming geom is awake, also need to check with hibernated geoms
                            if not is_incoming_geom_hibernated:
                                for j in range(n_active_hib):
                                    i_ga = self.active_buffer_hib[j, i_b]
                                    i_gb = self.sort_buffer[i, i_b].i_g
                                    if i_ga > i_gb:
                                        i_ga, i_gb = i_gb, i_ga

                                    if not self._func_check_collision_valid(i_ga, i_gb, i_b):
                                        continue

                                    if not self._func_is_geom_aabbs_overlap(i_ga, i_gb, i_b):
                                        # Clear collision normal cache if not in contact
                                        # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = -1
                                        self.contact_cache[i_ga, i_gb, i_b].normal.fill(0.0)
                                        continue

                                    self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][0] = i_ga
                                    self.broad_collision_pairs[self.n_broad_pairs[i_b], i_b][1] = i_gb
                                    self.n_broad_pairs[i_b] = self.n_broad_pairs[i_b] + 1

                            if is_incoming_geom_hibernated:
                                self.active_buffer_hib[n_active_hib, i_b] = self.sort_buffer[i, i_b].i_g
                                n_active_hib = n_active_hib + 1
                            else:
                                self.active_buffer_awake[n_active_awake, i_b] = self.sort_buffer[i, i_b].i_g
                                n_active_awake = n_active_awake + 1
                        else:
                            i_g_to_remove = self.sort_buffer[i, i_b].i_g
                            if is_incoming_geom_hibernated:
                                for j in range(n_active_hib):
                                    if self.active_buffer_hib[j, i_b] == i_g_to_remove:
                                        if j < n_active_hib - 1:
                                            for k in range(j, n_active_hib - 1):
                                                self.active_buffer_hib[k, i_b] = self.active_buffer_hib[k + 1, i_b]
                                        n_active_hib = n_active_hib - 1
                                        break
                            else:
                                for j in range(n_active_awake):
                                    if self.active_buffer_awake[j, i_b] == i_g_to_remove:
                                        if j < n_active_awake - 1:
                                            for k in range(j, n_active_awake - 1):
                                                self.active_buffer_awake[k, i_b] = self.active_buffer_awake[k + 1, i_b]
                                        n_active_awake = n_active_awake - 1
                                        break

    @ti.kernel
    def _func_narrow_phase_convex_vs_convex(self):
        """
        NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`.
        However, parallelize over both B and collision_pairs (instead of only over B) leads to significantly slow performance for batched scene.
        We can treat B=0 and B>0 separately, but we will end up with messier code.
        Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
        Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
        Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
        """
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._solver._B):
            for i_pair in range(self.n_broad_pairs[i_b]):
                i_ga = self.broad_collision_pairs[i_pair, i_b][0]
                i_gb = self.broad_collision_pairs[i_pair, i_b][1]

                if (
                    self._solver.geoms_info[i_ga].is_convex
                    and self._solver.geoms_info[i_gb].is_convex
                    and not self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.TERRAIN
                    and not (
                        self._solver._enable_multi_contact
                        and self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.PLANE
                        and self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.BOX
                    )
                    and not (
                        self._solver._box_box_detection
                        and self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.BOX
                        and self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.BOX
                    )
                ):
                    self._func_convex_convex_contact(i_ga, i_gb, i_b)

    @ti.kernel
    def _func_narrow_phase_convex_specializations(self):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._solver._B):
            for i_pair in range(self.n_broad_pairs[i_b]):
                i_ga = self.broad_collision_pairs[i_pair, i_b][0]
                i_gb = self.broad_collision_pairs[i_pair, i_b][1]

                if self._solver.geoms_info[i_ga].type > self._solver.geoms_info[i_gb].type:
                    i_ga, i_gb = i_gb, i_ga

                if (
                    self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.PLANE
                    and self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.BOX
                ):
                    if ti.static(sys.platform == "darwin"):
                        # FIXME: It seems redundant, why don't we just call _func_plane_box_contact directly?
                        # Anyway in this function, we will call _func_plane_box_contact.
                        self._func_convex_convex_contact(i_ga, i_gb, i_b)
                    else:
                        self._func_plane_box_contact(i_ga, i_gb, i_b)

                if ti.static(self._solver._box_box_detection):
                    if (
                        self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.BOX
                        and self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.BOX
                    ):
                        self._func_box_box_contact(i_ga, i_gb, i_b)

    @ti.kernel
    def _func_narrow_phase_any_vs_terrain(self):
        """
        NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`. However, parallelize over both B and collisioin_pairs (instead of only over B) leads to significantly slow performance for batched scene. We can treat B=0 and B>0 separately, but we will end up with messier code.
        Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
        Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
        Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
        """
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._solver._B):
            for i_pair in range(self.n_broad_pairs[i_b]):
                i_ga = self.broad_collision_pairs[i_pair, i_b][0]
                i_gb = self.broad_collision_pairs[i_pair, i_b][1]

                if ti.static(self._has_terrain):
                    if self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.TERRAIN:
                        i_ga, i_gb = i_gb, i_ga

                    if self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.TERRAIN:
                        self._func_contact_mpr_terrain(i_ga, i_gb, i_b)

    @ti.kernel
    def _func_narrow_phase_nonconvex_vs_nonterrain(self):
        """
        NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`. However, parallelize over both B and collisioin_pairs (instead of only over B) leads to significantly slow performance for batched scene. We can treat B=0 and B>0 separately, but we will end up with messier code.
        Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
        Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
        Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
        """
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._solver._B):
            for i_pair in range(self.n_broad_pairs[i_b]):
                i_ga = self.broad_collision_pairs[i_pair, i_b][0]
                i_gb = self.broad_collision_pairs[i_pair, i_b][1]

                if ti.static(self._has_nonconvex_nonterrain):
                    if (
                        not (self._solver.geoms_info[i_ga].is_convex and self._solver.geoms_info[i_gb].is_convex)
                        and self._solver.geoms_info[i_gb].type != gs.GEOM_TYPE.TERRAIN
                    ):
                        is_col = False
                        tolerance = self._func_compute_tolerance(i_ga, i_gb, i_b)
                        for i in range(2):
                            if i == 1:
                                i_ga, i_gb = i_gb, i_ga

                            # initial point
                            is_col_i = False
                            normal_i = ti.Vector.zero(gs.ti_float, 3)
                            contact_pos_i = ti.Vector.zero(gs.ti_float, 3)
                            if not is_col:
                                is_col_i, normal_i, penetration_i, contact_pos_i = self._func_contact_vertex_sdf(
                                    i_ga, i_gb, i_b
                                )
                                if is_col_i:
                                    self._func_add_contact(i_ga, i_gb, normal_i, contact_pos_i, penetration_i, i_b)

                            if ti.static(self._solver._enable_multi_contact):
                                if not is_col and is_col_i:
                                    ga_state = self._solver.geoms_state[i_ga, i_b]
                                    gb_state = self._solver.geoms_state[i_gb, i_b]
                                    ga_pos, ga_quat = ga_state.pos, ga_state.quat
                                    gb_pos, gb_quat = gb_state.pos, gb_state.quat

                                    # Perturb geom_a around two orthogonal axes to find multiple contacts
                                    axis_0, axis_1 = self._func_contact_orthogonals(i_ga, i_gb, normal_i, i_b)

                                    n_con = 1
                                    for i_rot in range(1, 5):
                                        axis = (2 * (i_rot % 2) - 1) * axis_0 + (1 - 2 * ((i_rot // 2) % 2)) * axis_1

                                        qrot = gu.ti_rotvec_to_quat(self._mc_perturbation * axis)
                                        self._func_rotate_frame(i_ga, contact_pos_i, qrot, i_b)
                                        self._func_rotate_frame(i_gb, contact_pos_i, gu.ti_inv_quat(qrot), i_b)

                                        is_col, normal, penetration, contact_pos = self._func_contact_vertex_sdf(
                                            i_ga, i_gb, i_b
                                        )

                                        if is_col:
                                            if ti.static(not self._solver._enable_mujoco_compatibility):
                                                # 1. Project the contact point on both geometries
                                                # 2. Revert the effect of small rotation
                                                # 3. Update contact point
                                                contact_point_a = (
                                                    gu.ti_transform_by_quat(
                                                        (contact_pos - 0.5 * penetration * normal) - contact_pos_i,
                                                        gu.ti_inv_quat(qrot),
                                                    )
                                                    + contact_pos_i
                                                )
                                                contact_point_b = (
                                                    gu.ti_transform_by_quat(
                                                        (contact_pos + 0.5 * penetration * normal) - contact_pos_i,
                                                        qrot,
                                                    )
                                                    + contact_pos_i
                                                )
                                                contact_pos = 0.5 * (contact_point_a + contact_point_b)

                                                # First-order correction of the normal direction
                                                twist_rotvec = ti.math.clamp(
                                                    normal.cross(normal_i),
                                                    -self._mc_perturbation,
                                                    self._mc_perturbation,
                                                )
                                                normal += twist_rotvec.cross(normal)

                                                # Make sure that the penetration is still positive
                                                penetration = normal.dot(contact_point_b - contact_point_a)

                                            # Discard contact point is repeated
                                            repeated = False
                                            for i_con in range(n_con):
                                                if not repeated:
                                                    idx_prev = self.n_contacts[i_b] - 1 - i_con
                                                    prev_contact = self.contact_data[idx_prev, i_b].pos
                                                    if (contact_pos - prev_contact).norm() < tolerance:
                                                        repeated = True

                                            if not repeated:
                                                if penetration > -tolerance:
                                                    penetration = ti.max(penetration, 0.0)
                                                    self._func_add_contact(
                                                        i_ga, i_gb, normal, contact_pos, penetration, i_b
                                                    )
                                                    n_con += 1

                                        self._solver.geoms_state[i_ga, i_b].pos = ga_pos
                                        self._solver.geoms_state[i_ga, i_b].quat = ga_quat
                                        self._solver.geoms_state[i_gb, i_b].pos = gb_pos
                                        self._solver.geoms_state[i_gb, i_b].quat = gb_quat

                        if not is_col:  # check edge-edge if vertex-face is not detected
                            is_col, normal, penetration, contact_pos = self._func_contact_edge_sdf(i_ga, i_gb, i_b)
                            if is_col:
                                self._func_add_contact(i_ga, i_gb, normal, contact_pos, penetration, i_b)

    @ti.func
    def _func_plane_box_contact(self, i_ga, i_gb, i_b):
        ga_info = self._solver.geoms_info[i_ga]
        gb_info = self._solver.geoms_info[i_gb]
        ga_state = self._solver.geoms_state[i_ga, i_b]
        gb_state = self._solver.geoms_state[i_gb, i_b]

        plane_dir = ti.Vector([ga_info.data[0], ga_info.data[1], ga_info.data[2]], dt=gs.ti_float)
        plane_dir = gu.ti_transform_by_quat(plane_dir, ga_state.quat)
        normal = -plane_dir.normalized()

        v1, _ = self._mpr.support_field._func_support_box(normal, i_gb, i_b)
        penetration = normal.dot(v1 - ga_state.pos)

        if penetration > 0.0:
            contact_pos = v1 - 0.5 * penetration * normal
            self._func_add_contact(i_ga, i_gb, normal, contact_pos, penetration, i_b)

            if ti.static(self._solver._enable_multi_contact):
                n_con = 1
                contact_pos_0 = contact_pos
                tolerance = self._func_compute_tolerance(i_ga, i_gb, i_b)
                for i_v in range(gb_info.vert_start, gb_info.vert_end):
                    if n_con < self._n_contacts_per_pair:
                        pos_corner = gu.ti_transform_by_trans_quat(
                            self._solver.verts_info[i_v].init_pos, gb_state.pos, gb_state.quat
                        )
                        penetration = normal.dot(pos_corner - ga_state.pos)
                        if penetration > 0.0:
                            contact_pos = pos_corner - 0.5 * penetration * normal
                            if (contact_pos - contact_pos_0).norm() > tolerance:
                                self._func_add_contact(i_ga, i_gb, normal, contact_pos, penetration, i_b)
                                n_con = n_con + 1

    @ti.func
    def _func_add_contact(self, i_ga, i_gb, normal, contact_pos, penetration, i_b):
        i_col = self.n_contacts[i_b]

        if i_col == self._max_contact_pairs:
            ti.static_print(self._warn_msg_max_collision_pairs)
        else:
            ga_info = self._solver.geoms_info[i_ga]
            gb_info = self._solver.geoms_info[i_gb]

            friction_a = ga_info.friction * self._solver.geoms_state[i_ga, i_b].friction_ratio
            friction_b = gb_info.friction * self._solver.geoms_state[i_gb, i_b].friction_ratio

            # b to a
            self.contact_data[i_col, i_b].geom_a = i_ga
            self.contact_data[i_col, i_b].geom_b = i_gb
            self.contact_data[i_col, i_b].normal = normal
            self.contact_data[i_col, i_b].pos = contact_pos
            self.contact_data[i_col, i_b].penetration = penetration
            self.contact_data[i_col, i_b].friction = ti.max(ti.max(friction_a, friction_b), 1e-2)
            self.contact_data[i_col, i_b].sol_params = 0.5 * (ga_info.sol_params + gb_info.sol_params)
            self.contact_data[i_col, i_b].link_a = ga_info.link_idx
            self.contact_data[i_col, i_b].link_b = gb_info.link_idx

            self.n_contacts[i_b] = i_col + 1

    @ti.func
    def _func_compute_tolerance(self, i_ga, i_gb, i_b):
        # Note that the original world-aligned bounding box is used to computed the absolute tolerance from the
        # relative one. This way, it is a constant that does not depends on the orientation of the geometry, which
        # makes sense since the scale of the geometries is an intrinsic property and not something that is supposed
        # to change dynamically.
        aabb_size_b = (self._solver.geoms_init_AABB[i_gb, 7] - self._solver.geoms_init_AABB[i_gb, 0]).norm()
        aabb_size = aabb_size_b
        if self._solver.geoms_info[i_ga].type != gs.GEOM_TYPE.PLANE:
            aabb_size_a = (self._solver.geoms_init_AABB[i_ga, 7] - self._solver.geoms_init_AABB[i_ga, 0]).norm()
            aabb_size = ti.min(aabb_size_a, aabb_size_b)

        return 0.5 * self._mc_tolerance * aabb_size

    @ti.func
    def _func_contact_orthogonals(self, i_ga, i_gb, normal, i_b):
        axis_0 = ti.Vector.zero(gs.ti_float, 3)
        axis_1 = ti.Vector.zero(gs.ti_float, 3)

        if ti.static(self._solver._enable_mujoco_compatibility):
            # Choose between world axes Y or Z to avoid colinearity issue
            if ti.abs(normal[1]) < 0.5:
                axis_0[1] = 1.0
            else:
                axis_0[2] = 1.0

            # Project axis on orthogonal plane to contact normal
            axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()

            # Perturb with some noise so that they do not align with world axes to avoid denegerated cases
            axis_1 = (normal.cross(axis_0) + 0.1 * axis_0).normalized()
            axis_0 = axis_1.cross(normal)
        else:
            # The reference geometry is the one that will have the largest impact on the position of
            # the contact point. Basically, the smallest one between the two, which can be approximated
            # by the volume of their respective bounding box.
            i_g = i_gb
            if self._solver.geoms_info[i_ga].type != gs.GEOM_TYPE.PLANE:
                size_ga = self._solver.geoms_init_AABB[i_ga, 7]
                volume_ga = size_ga[0] * size_ga[1] * size_ga[2]
                size_gb = self._solver.geoms_init_AABB[i_gb, 7]
                volume_gb = size_gb[0] * size_gb[1] * size_gb[2]
                i_g = i_ga if volume_ga < volume_gb else i_gb

            # Compute orthogonal basis mixing principal inertia axes of geometry with contact normal
            i_l = self._solver.geoms_info[i_g].link_idx
            rot = gu.ti_quat_to_R(self._solver.links_state[i_l, i_b].i_quat)
            axis_idx = gs.ti_int(0)
            axis_angle_max = gs.ti_float(0.0)
            for i in ti.static(range(3)):
                axis_angle = ti.abs(rot[:, i].dot(normal))
                if axis_angle > axis_angle_max:
                    axis_angle_max = axis_angle
                    axis_idx = i
            axis_idx = (axis_idx + 1) % 3
            axis_0 = rot[:, axis_idx]
            axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()
            axis_1 = normal.cross(axis_0)

        return axis_0, axis_1

    @ti.func
    def _func_convex_convex_contact(self, i_ga, i_gb, i_b):
        if self._solver.geoms_info[i_ga].type > self._solver.geoms_info[i_gb].type:
            i_ga, i_gb = i_gb, i_ga

        if (
            self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.PLANE
            and self._solver.geoms_info[i_gb].type == gs.GEOM_TYPE.BOX
        ):
            if ti.static(sys.platform == "darwin"):
                self._func_plane_box_contact(i_ga, i_gb, i_b)
        else:
            # Disabling multi-contact for pairs of decomposed geoms would speed up simulation but may cause physical
            # instabilities in the few cases where multiple contact points are actually need. Increasing the tolerance
            # criteria to get rid of redundant contact points seems to be a better option.
            multi_contact = (
                self._solver._enable_multi_contact
                # and not (self._solver.geoms_info[i_ga].is_decomposed and self._solver.geoms_info[i_gb].is_decomposed)
                and self._solver.geoms_info[i_ga].type != gs.GEOM_TYPE.SPHERE
                and self._solver.geoms_info[i_ga].type != gs.GEOM_TYPE.ELLIPSOID
                and self._solver.geoms_info[i_gb].type != gs.GEOM_TYPE.SPHERE
                and self._solver.geoms_info[i_gb].type != gs.GEOM_TYPE.ELLIPSOID
            )

            tolerance = self._func_compute_tolerance(i_ga, i_gb, i_b)

            # Backup state before local perturbation
            ga_state = self._solver.geoms_state[i_ga, i_b]
            gb_state = self._solver.geoms_state[i_gb, i_b]
            ga_pos, ga_quat = ga_state.pos, ga_state.quat
            gb_pos, gb_quat = gb_state.pos, gb_state.quat

            # Pre-allocate some buffers
            is_col_0 = False
            penetration_0 = gs.ti_float(0.0)
            normal_0 = ti.Vector.zero(gs.ti_float, 3)
            contact_pos_0 = ti.Vector.zero(gs.ti_float, 3)

            is_col = False
            penetration = gs.ti_float(0.0)
            normal = ti.Vector.zero(gs.ti_float, 3)
            contact_pos = ti.Vector.zero(gs.ti_float, 3)

            n_con = gs.ti_int(0)
            axis_0 = ti.Vector.zero(gs.ti_float, 3)
            axis_1 = ti.Vector.zero(gs.ti_float, 3)
            qrot = ti.Vector.zero(gs.ti_float, 4)

            for i_detection in range(5):
                if multi_contact and is_col_0:
                    # Perturbation axis must not be aligned with the principal axes of inertia the geometry,
                    # otherwise it would be more sensitive to ill-conditionning.
                    axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                    qrot = gu.ti_rotvec_to_quat(self._mc_perturbation * axis)
                    self._func_rotate_frame(i_ga, contact_pos_0, qrot, i_b)
                    self._func_rotate_frame(i_gb, contact_pos_0, gu.ti_inv_quat(qrot), i_b)

                if (multi_contact and is_col_0) or (i_detection == 0):
                    try_sdf = False
                    if self._solver.geoms_info[i_ga].type == gs.GEOM_TYPE.PLANE:
                        ga_info = self._solver.geoms_info[i_ga]
                        plane_dir = ti.Vector([ga_info.data[0], ga_info.data[1], ga_info.data[2]], dt=gs.ti_float)
                        plane_dir = gu.ti_transform_by_quat(plane_dir, self._solver.geoms_state[i_ga, i_b].quat)
                        normal = -plane_dir.normalized()

                        v1 = self._mpr.support_driver(normal, i_gb, i_b)
                        penetration = normal.dot(v1 - self._solver.geoms_state[i_ga, i_b].pos)
                        contact_pos = v1 - 0.5 * penetration * normal
                        is_col = penetration > 0
                    else:
                        ### MPR, MPR + SDF
                        if ti.static(self.ccd_algorithm != CCD_ALGORITHM_CODE.GJK):
                            # Try using MPR before anything else
                            is_mpr_updated = False
                            is_mpr_guess_direction_available = True
                            normal_ws = self.contact_cache[i_ga, i_gb, i_b].normal
                            for i_mpr in range(2):
                                if i_mpr == 1:
                                    # Try without warm-start if no contact was detected using it.
                                    # When penetration depth is very shallow, MPR may wrongly classify two geometries as not in
                                    # contact while they actually are. This helps to improve contact persistence without increasing
                                    # much the overall computational cost since the fallback should not be triggered very often.
                                    is_mpr_guess_direction_available = (ti.abs(normal_ws) > gs.EPS).any()
                                    if (i_detection == 0) and not is_col and is_mpr_guess_direction_available:
                                        normal_ws = ti.Vector.zero(gs.ti_float, 3)
                                        is_mpr_updated = False

                                if not is_mpr_updated:
                                    is_col, normal, penetration, contact_pos = self._mpr.func_mpr_contact(
                                        i_ga, i_gb, i_b, normal_ws
                                    )
                                    is_mpr_updated = True

                            # Fallback on SDF if collision is detected by MPR but no collision direction was cached but the
                            # initial penetration is already quite large, because the contact information provided by MPR
                            # may be unreliable in such a case.
                            # Here it is assumed that generic SDF is much slower than MPR, so it is faster in average
                            # to first make sure that the geometries are truly colliding and only after to run SDF if
                            # necessary. This would probably not be the case anymore if it was possible to rely on
                            # specialized SDF implementation for convex-convex collision detection in the first place.
                            if is_col and penetration > tolerance and not is_mpr_guess_direction_available:
                                # Note that SDF may detect different collision points depending on geometry ordering.
                                # Because of this, it is necessary to run it twice and take the contact information
                                # associated with the point of deepest penetration.
                                try_sdf = True

                        ### GJK
                        elif ti.static(self.ccd_algorithm == CCD_ALGORITHM_CODE.GJK):
                            # If it was not the first detection, only detect single contact point.
                            self._gjk.func_gjk_contact(i_ga, i_gb, i_b, i_detection == 0)

                            is_col = self._gjk.is_col[i_b] == 1
                            penetration = self._gjk.penetration[i_b]
                            n_contacts = self._gjk.n_contacts[i_b]

                            if is_col:
                                if self._gjk.multi_contact_flag[i_b]:
                                    # Used MuJoCo's multi-contact algorithm to find multiple contact points. Therefore,
                                    # add the discovered contact points and stop multi-contact search.
                                    for i_c in range(n_contacts):
                                        # Ignore contact points if the number of contacts exceeds the limit.
                                        if i_c < self._n_contacts_per_pair:
                                            contact_pos = self._gjk.contact_pos[i_b, i_c]
                                            normal = self._gjk.normal[i_b, i_c]
                                            self._func_add_contact(i_ga, i_gb, normal, contact_pos, penetration, i_b)

                                    break
                                else:
                                    contact_pos = self._gjk.contact_pos[i_b, 0]
                                    normal = self._gjk.normal[i_b, 0]

                    if ti.static(self.ccd_algorithm == CCD_ALGORITHM_CODE.MPR_SDF):
                        if try_sdf:
                            is_col_a = False
                            is_col_b = False
                            normal_a = ti.Vector.zero(gs.ti_float, 3)
                            normal_b = ti.Vector.zero(gs.ti_float, 3)
                            penetration_b = gs.ti_float(0.0)
                            penetration_a = gs.ti_float(0.0)
                            contact_pos_a = ti.Vector.zero(gs.ti_float, 3)
                            contact_pos_b = ti.Vector.zero(gs.ti_float, 3)
                            for i_sdf in range(2):
                                # FIXME: It is impossible to rely on `_func_contact_convex_convex_sdf` to get the contact
                                # information because the compilation times skyrockets from 42s for `_func_contact_vertex_sdf`
                                # to 2min51s on Apple Silicon M4 Max, which is not acceptable.
                                # is_col_i, normal_i, penetration_i, contact_pos_i, i_va = (
                                #     self._func_contact_convex_convex_sdf(
                                #         i_ga if i_sdf == 0 else i_gb,
                                #         i_gb if i_sdf == 0 else i_ga,
                                #         i_b,
                                #         self.contact_cache[i_ga, i_gb, i_b].i_va_ws,
                                #     )
                                # )
                                # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = i_va
                                is_col_i, normal_i, penetration_i, contact_pos_i = self._func_contact_vertex_sdf(
                                    i_ga if i_sdf == 0 else i_gb, i_gb if i_sdf == 0 else i_ga, i_b
                                )
                                if i_sdf == 0:
                                    is_col_a = is_col_i
                                    normal_a = normal_i
                                    penetration_a = penetration_i
                                    contact_pos_a = contact_pos_i
                                else:
                                    is_col_b = is_col_i
                                    normal_b = -normal_i
                                    penetration_b = penetration_i
                                    contact_pos_b = contact_pos_i

                            # MPR cannot handle collision detection for fully enclosed geometries. Falling back to SDF.
                            # Note that SDF does not take into account to direction of interest. As such, it cannot be
                            # used reliably for anything else than the point of deepest penetration.
                            prefer_sdf = self._mc_tolerance * penetration >= self._mpr_to_sdf_overlap_ratio * tolerance

                            if is_col_a and (
                                not is_col_b or penetration_a >= max(penetration_b, (not prefer_sdf) * penetration)
                            ):
                                is_col = is_col_a
                                normal = normal_a
                                penetration = penetration_a
                                contact_pos = contact_pos_a
                            elif is_col_b and (
                                not is_col_a or penetration_b > max(penetration_a, (not prefer_sdf) * penetration)
                            ):
                                is_col = is_col_b
                                normal = normal_b
                                penetration = penetration_b
                                contact_pos = contact_pos_b

                if i_detection == 0:
                    is_col_0, normal_0, penetration_0, contact_pos_0 = is_col, normal, penetration, contact_pos
                    if is_col_0:
                        self._func_add_contact(i_ga, i_gb, normal_0, contact_pos_0, penetration_0, i_b)
                        if multi_contact:
                            # perturb geom_a around two orthogonal axes to find multiple contacts
                            axis_0, axis_1 = self._func_contact_orthogonals(i_ga, i_gb, normal, i_b)
                            n_con = 1

                        if ti.static(not self._solver._enable_mujoco_compatibility):
                            self.contact_cache[i_ga, i_gb, i_b].normal = normal
                    else:
                        # Clear collision normal cache if not in contact
                        # self.contact_cache[i_ga, i_gb, i_b].i_va_ws = -1
                        self.contact_cache[i_ga, i_gb, i_b].normal.fill(0.0)

                elif multi_contact and is_col_0 > 0 and is_col > 0:
                    if ti.static(self.ccd_algorithm == CCD_ALGORITHM_CODE.MPR_SDF):
                        # 1. Project the contact point on both geometries
                        # 2. Revert the effect of small rotation
                        # 3. Update contact point
                        contact_point_a = (
                            gu.ti_transform_by_quat(
                                (contact_pos - 0.5 * penetration * normal) - contact_pos_0,
                                gu.ti_inv_quat(qrot),
                            )
                            + contact_pos_0
                        )
                        contact_point_b = (
                            gu.ti_transform_by_quat(
                                (contact_pos + 0.5 * penetration * normal) - contact_pos_0,
                                qrot,
                            )
                            + contact_pos_0
                        )
                        contact_pos = 0.5 * (contact_point_a + contact_point_b)

                        # First-order correction of the normal direction.
                        # The way the contact normal gets twisted by applying perturbation of geometry poses is
                        # unpredictable as it depends on the final portal discovered by MPR. Alternatively, let compute
                        # the mininal rotation that makes the corrected twisted normal as closed as possible to the
                        # original one, up to the scale of the perturbation, then apply first-order Taylor expension of
                        # Rodrigues' rotation formula.
                        twist_rotvec = ti.math.clamp(
                            normal.cross(normal_0), -self._mc_perturbation, self._mc_perturbation
                        )
                        normal += twist_rotvec.cross(normal)

                        # Make sure that the penetration is still positive before adding contact point.
                        # Note that adding some negative tolerance improves physical stability by encouraging persistent
                        # contact points and thefore more continuous contact forces, without changing the mean-field
                        # dynamics since zero-penetration contact points should not induce any force.
                        penetration = normal.dot(contact_point_b - contact_point_a)

                    elif ti.static(self.ccd_algorithm == CCD_ALGORITHM_CODE.GJK):
                        # Only change penetration to the initial one, because the normal vector could change abruptly
                        # under GJK-EPA as the nearest simplex is determined by discrete logic, unlike MPR.
                        penetration = penetration_0

                    # Discard contact point is repeated
                    repeated = False
                    for i_con in range(n_con):
                        if not repeated:
                            idx_prev = self.n_contacts[i_b] - 1 - i_con
                            prev_contact = self.contact_data[idx_prev, i_b].pos
                            if (contact_pos - prev_contact).norm() < tolerance:
                                repeated = True

                    if not repeated:
                        if penetration > -tolerance:
                            penetration = ti.max(penetration, 0.0)
                            self._func_add_contact(i_ga, i_gb, normal, contact_pos, penetration, i_b)
                            n_con = n_con + 1

                    self._solver.geoms_state[i_ga, i_b].pos = ga_pos
                    self._solver.geoms_state[i_ga, i_b].quat = ga_quat
                    self._solver.geoms_state[i_gb, i_b].pos = gb_pos
                    self._solver.geoms_state[i_gb, i_b].quat = gb_quat

    @ti.func
    def _func_rotate_frame(self, i_g, contact_pos, qrot, i_b):
        self._solver.geoms_state[i_g, i_b].quat = gu.ti_transform_quat_by_quat(
            self._solver.geoms_state[i_g, i_b].quat, qrot
        )

        rel = contact_pos - self._solver.geoms_state[i_g, i_b].pos
        vec = gu.ti_transform_by_quat(rel, qrot)
        vec = vec - rel
        self._solver.geoms_state[i_g, i_b].pos = self._solver.geoms_state[i_g, i_b].pos - vec

    @ti.func
    def _func_box_box_contact(self, i_ga: ti.i32, i_gb: ti.i32, i_b: ti.i32):
        """
        Use Mujoco's box-box contact detection algorithm for more stable collision detection.

        The compilation and running time of this function is longer than the MPR-based contact detection.

        Algorithm is from

        https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_box.c
        """
        n = 0
        code = -1
        margin = gs.ti_float(0.0)
        is_return = False
        cle1, cle2 = 0, 0
        in_ = 0
        tmp2 = ti.Vector.zero(gs.ti_float, 3)
        margin2 = margin * margin
        rotmore = ti.Matrix.zero(gs.ti_float, 3, 3)

        ga_info = self._solver.geoms_info[i_ga]
        gb_info = self._solver.geoms_info[i_gb]
        ga_state = self._solver.geoms_state[i_ga, i_b]
        gb_state = self._solver.geoms_state[i_gb, i_b]

        size1 = ti.Vector([ga_info.data[0], ga_info.data[1], ga_info.data[2]], dt=gs.ti_float) / 2
        size2 = ti.Vector([gb_info.data[0], gb_info.data[1], gb_info.data[2]], dt=gs.ti_float) / 2

        pos1, pos2 = ga_state.pos, gb_state.pos
        mat1, mat2 = gu.ti_quat_to_R(ga_state.quat), gu.ti_quat_to_R(gb_state.quat)

        tmp1 = pos2 - pos1
        pos21 = mat1.transpose() @ tmp1

        tmp1 = pos1 - pos2
        pos12 = mat2.transpose() @ tmp1

        rot = mat1.transpose() @ mat2
        rott = rot.transpose()

        rotabs = ti.abs(rot)
        rottabs = ti.abs(rott)

        plen2 = rotabs @ size2
        plen1 = rotabs.transpose() @ size1
        penetration = margin
        for i in ti.static(range(3)):
            penetration = penetration + size1[i] * 3 + size2[i] * 3
        for i in ti.static(range(3)):
            c1 = -ti.abs(pos21[i]) + size1[i] + plen2[i]
            c2 = -ti.abs(pos12[i]) + size2[i] + plen1[i]

            if (c1 < -margin) or (c2 < -margin):
                is_return = True

            if c1 < penetration:
                penetration = c1
                code = i + 3 * (pos21[i] < 0) + 0

            if c2 < penetration:
                penetration = c2
                code = i + 3 * (pos12[i] < 0) + 6
        clnorm = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
        for i, j in ti.static(ti.ndrange(3, 3)):
            rj0 = rott[j, 0]
            rj1 = rott[j, 1]
            rj2 = rott[j, 2]
            if i == 0:
                tmp2 = ti.Vector([0.0, -rj2, +rj1], dt=gs.ti_float)
            elif i == 1:
                tmp2 = ti.Vector([+rj2, 0.0, -rj0], dt=gs.ti_float)
            else:
                tmp2 = ti.Vector([-rj1, +rj0, 0.0], dt=gs.ti_float)

            c1 = tmp2.norm()
            tmp2 = tmp2 / c1
            if c1 >= gs.EPS:
                c2 = pos21.dot(tmp2)

                c3 = gs.ti_float(0.0)

                for k in ti.static(range(3)):
                    if k != i:
                        c3 = c3 + size1[k] * ti.abs(tmp2[k])

                for k in ti.static(range(3)):
                    if k != j:
                        m = i
                        n = 3 - k - j
                        if k - j > 3:
                            m = m - 1
                            n = n + 3
                        c3 = c3 + size2[k] * rotabs[m, n] / c1

                        3 * i + 3 - k - j

                c3 = c3 - ti.abs(c2)

                if c3 < -margin:
                    is_return = True

                if c3 < penetration * (1.0 - 1e-12):
                    penetration = c3
                    cle1 = 0
                    for k in ti.static(range(3)):
                        if (k != i) and ((tmp2[k] > 0) ^ (c2 < 0)):
                            cle1 = cle1 + (1 << k)

                    cle2 = 0
                    for k in ti.static(range(3)):
                        if k != j:
                            m = i
                            n = 3 - k - j
                            if k - j > 3:
                                m = m - 1
                                n = n + 3
                            if (rot[m, n] > 0) ^ (c2 < 0) ^ (ti.raw_mod(k - j + 3, 3) == 1):
                                cle2 = cle2 + (1 << k)

                    code = 12 + i * 3 + j
                    clnorm = tmp2
                    in_ = c2 < 0
        if code == -1:
            is_return = True

        if not is_return:
            if code < 12:
                q1 = code % 6
                q2 = code // 6

                if q1 == 0:
                    rotmore[0, 2] = -1
                    rotmore[1, 1] = +1
                    rotmore[2, 0] = +1
                elif q1 == 1:
                    rotmore[0, 0] = +1
                    rotmore[1, 2] = -1
                    rotmore[2, 1] = +1
                elif q1 == 2:
                    rotmore[0, 0] = +1
                    rotmore[1, 1] = +1
                    rotmore[2, 2] = +1
                elif q1 == 3:
                    rotmore[0, 2] = +1
                    rotmore[1, 1] = +1
                    rotmore[2, 0] = -1
                elif q1 == 4:
                    rotmore[0, 0] = +1
                    rotmore[1, 2] = +1
                    rotmore[2, 1] = -1
                elif q1 == 5:
                    rotmore[0, 0] = -1
                    rotmore[1, 1] = +1
                    rotmore[2, 2] = -1

                i0 = 0
                i1 = 1
                i2 = 2
                f0 = f1 = f2 = 1
                if q1 == 0:
                    i0 = 2
                    f0 = -1
                    i2 = 0
                elif q1 == 1:
                    i1 = 2
                    f1 = -1
                    i2 = 1
                elif q1 == 3:
                    i0 = 2
                    i2 = 0
                    f2 = -1
                elif q1 == 4:
                    i1 = 2
                    i2 = 1
                    f2 = -1
                elif q1 == 5:
                    f0 = -1
                    f2 = -1

                r = ti.Matrix.zero(gs.ti_float, 3, 3)
                p = ti.Vector.zero(gs.ti_float, 3)
                s = ti.Vector.zero(gs.ti_float, 3)
                if q2:
                    r = rotmore @ rot.transpose()
                    p = rotaxis(pos12, i0, i1, i2, f0, f1, f2)
                    tmp1 = rotaxis(size2, i0, i1, i2, f0, f1, f2)
                    s = size1
                else:
                    r = rotmatx(rot, i0, i1, i2, f0, f1, f2)
                    p = rotaxis(pos21, i0, i1, i2, f0, f1, f2)
                    tmp1 = rotaxis(size1, i0, i1, i2, f0, f1, f2)
                    s = size2

                rt = r.transpose()
                ss = ti.abs(tmp1)
                lx = ss[0]
                ly = ss[1]
                hz = ss[2]
                p[2] = p[2] - hz
                lp = p

                clcorner = 0

                for i in ti.static(range(3)):
                    if r[2, i] < 0:
                        clcorner = clcorner + (1 << i)

                for i in ti.static(range(3)):
                    lp = lp + rt[i, :] * s[i] * (1 if (clcorner & (1 << i)) else -1)

                m, k = 0, 0
                self.box_pts[m, i_b] = lp
                m = m + 1

                for i in ti.static(range(3)):
                    if ti.abs(r[2, i]) < 0.5:
                        self.box_pts[m, i_b] = rt[i, :] * s[i] * (-2 if (clcorner & (1 << i)) else 2)
                        m = m + 1

                self.box_pts[3, i_b] = self.box_pts[0, i_b] + self.box_pts[1, i_b]
                self.box_pts[4, i_b] = self.box_pts[0, i_b] + self.box_pts[2, i_b]
                self.box_pts[5, i_b] = self.box_pts[3, i_b] + self.box_pts[2, i_b]

                if m > 1:
                    self.box_lines[k, i_b][0:3] = self.box_pts[0, i_b]
                    self.box_lines[k, i_b][3:6] = self.box_pts[1, i_b]
                    k = k + 1

                if m > 2:
                    self.box_lines[k, i_b][0:3] = self.box_pts[0, i_b]
                    self.box_lines[k, i_b][3:6] = self.box_pts[2, i_b]
                    k = k + 1

                    self.box_lines[k, i_b][0:3] = self.box_pts[3, i_b]
                    self.box_lines[k, i_b][3:6] = self.box_pts[2, i_b]
                    k = k + 1

                    self.box_lines[k, i_b][0:3] = self.box_pts[4, i_b]
                    self.box_lines[k, i_b][3:6] = self.box_pts[1, i_b]
                    k = k + 1

                for i in range(k):
                    for q in ti.static(range(2)):
                        a = self.box_lines[i, i_b][0 + q]
                        b = self.box_lines[i, i_b][3 + q]
                        c = self.box_lines[i, i_b][1 - q]
                        d = self.box_lines[i, i_b][4 - q]
                        if ti.abs(b) > gs.EPS:
                            for _j in ti.static(range(2)):
                                j = 2 * _j - 1
                                l = ss[q] * j
                                c1 = (l - a) / b
                                if 0 <= c1 and c1 <= 1:
                                    c2 = c + d * c1
                                    if ti.abs(c2) <= ss[1 - q]:
                                        self.box_points[n, i_b] = (
                                            self.box_lines[i, i_b][0:3] + self.box_lines[i, i_b][3:6] * c1
                                        )
                                        n = n + 1
                a = self.box_pts[1, i_b][0]
                b = self.box_pts[2, i_b][0]
                c = self.box_pts[1, i_b][1]
                d = self.box_pts[2, i_b][1]
                c1 = a * d - b * c

                if m > 2:
                    for i in ti.static(range(4)):
                        llx = lx if (i // 2) else -lx
                        lly = ly if (i % 2) else -ly

                        x = llx - self.box_pts[0, i_b][0]
                        y = lly - self.box_pts[0, i_b][1]

                        u = (x * d - y * b) / c1
                        v = (y * a - x * c) / c1

                        if 0 < u and u < 1 and 0 < v and v < 1:
                            self.box_points[n, i_b] = ti.Vector(
                                [
                                    llx,
                                    lly,
                                    self.box_pts[0, i_b][2] + u * self.box_pts[1, i_b][2] + v * self.box_pts[2, i_b][2],
                                ]
                            )
                            n = n + 1

                for i in range(1 << (m - 1)):
                    tmp1 = self.box_pts[0 if i == 0 else i + 2, i_b]
                    if not (i and (tmp1[0] <= -lx or tmp1[0] >= lx or tmp1[1] <= -ly or tmp1[1] >= ly)):
                        self.box_points[n, i_b] = tmp1
                        n = n + 1
                m = n
                n = 0

                for i in range(m):
                    if self.box_points[i, i_b][2] <= margin:
                        self.box_points[n, i_b] = self.box_points[i, i_b]
                        self.box_depth[n, i_b] = self.box_points[n, i_b][2]
                        self.box_points[n, i_b][2] = self.box_points[n, i_b][2] * 0.5
                        n = n + 1
                r = (mat2 if q2 else mat1) @ rotmore.transpose()
                p = pos2 if q2 else pos1
                tmp2 = ti.Vector(
                    [(-1 if q2 else 1) * r[0, 2], (-1 if q2 else 1) * r[1, 2], (-1 if q2 else 1) * r[2, 2]],
                    dt=gs.ti_float,
                )
                normal_0 = tmp2
                for i in range(n):
                    dist = self.box_points[i, i_b][2]
                    self.box_points[i, i_b][2] = self.box_points[i, i_b][2] + hz
                    tmp2 = r @ self.box_points[i, i_b]
                    contact_pos = tmp2 + p
                    self._func_add_contact(i_ga, i_gb, -normal_0, contact_pos, -dist, i_b)

            else:
                code = code - 12

                q1 = code // 3
                q2 = code % 3

                ax1, ax2 = 0, 0
                pax1, pax2 = 0, 0

                if q2 == 0:
                    ax1, ax2 = 1, 2
                elif q2 == 1:
                    ax1, ax2 = 0, 2
                elif q2 == 2:
                    ax1, ax2 = 1, 0

                if q1 == 0:
                    pax1, pax2 = 1, 2
                elif q1 == 1:
                    pax1, pax2 = 0, 2
                elif q1 == 2:
                    pax1, pax2 = 1, 0
                if rotabs[q1, ax1] < rotabs[q1, ax2]:
                    ax1 = ax2
                    ax2 = 3 - q2 - ax1

                if rottabs[q2, pax1] < rottabs[q2, pax2]:
                    pax1 = pax2
                    pax2 = 3 - q1 - pax1

                clface = 0
                if cle1 & (1 << pax2):
                    clface = pax2
                else:
                    clface = pax2 + 3

                rotmore.fill(0.0)
                if clface == 0:
                    rotmore[0, 2], rotmore[1, 1], rotmore[2, 0] = -1, +1, +1
                elif clface == 1:
                    rotmore[0, 0], rotmore[1, 2], rotmore[2, 1] = +1, -1, +1
                elif clface == 2:
                    rotmore[0, 0], rotmore[1, 1], rotmore[2, 2] = +1, +1, +1
                elif clface == 3:
                    rotmore[0, 2], rotmore[1, 1], rotmore[2, 0] = +1, +1, -1
                elif clface == 4:
                    rotmore[0, 0], rotmore[1, 2], rotmore[2, 1] = +1, +1, -1
                elif clface == 5:
                    rotmore[0, 0], rotmore[1, 1], rotmore[2, 2] = -1, +1, -1

                i0, i1, i2 = 0, 1, 2
                f0, f1, f2 = 1, 1, 1

                if clface == 0:
                    i0, i2, f0 = 2, 0, -1
                elif clface == 1:
                    i1, i2, f1 = 2, 1, -1
                elif clface == 3:
                    i0, i2, f2 = 2, 0, -1
                elif clface == 4:
                    i1, i2, f2 = 2, 1, -1
                elif clface == 5:
                    f0, f2 = -1, -1

                p = rotaxis(pos21, i0, i1, i2, f0, f1, f2)
                rnorm = rotaxis(clnorm, i0, i1, i2, f0, f1, f2)
                r = rotmatx(rot, i0, i1, i2, f0, f1, f2)

                # TODO
                tmp1 = rotmore.transpose() @ size1

                s = ti.abs(tmp1)
                rt = r.transpose()

                lx, ly, hz = s[0], s[1], s[2]
                p[2] = p[2] - hz

                n = 0
                self.box_points[n, i_b] = p

                self.box_points[n, i_b] = self.box_points[n, i_b] + rt[ax1, :] * size2[ax1] * (
                    1 if (cle2 & (1 << ax1)) else -1
                )
                self.box_points[n, i_b] = self.box_points[n, i_b] + rt[ax2, :] * size2[ax2] * (
                    1 if (cle2 & (1 << ax2)) else -1
                )

                self.box_points[n + 1, i_b] = self.box_points[n, i_b]
                self.box_points[n, i_b] = self.box_points[n, i_b] + rt[q2, :] * size2[q2]

                n = 1
                self.box_points[n, i_b] = self.box_points[n, i_b] - rt[q2, :] * size2[q2]

                n = 2
                self.box_points[n, i_b] = p
                self.box_points[n, i_b] = self.box_points[n, i_b] + rt[ax1, :] * size2[ax1] * (
                    -1 if (cle2 & (1 << ax1)) else 1
                )
                self.box_points[n, i_b] = self.box_points[n, i_b] + rt[ax2, :] * size2[ax2] * (
                    1 if (cle2 & (1 << ax2)) else -1
                )

                self.box_points[n + 1, i_b] = self.box_points[n, i_b]
                self.box_points[n, i_b] = self.box_points[n, i_b] + rt[q2, :] * size2[q2]

                n = 3
                self.box_points[n, i_b] = self.box_points[n, i_b] - rt[q2, :] * size2[q2]

                n = 4
                self.box_axi[0, i_b] = self.box_points[0, i_b]
                self.box_axi[1, i_b] = self.box_points[1, i_b] - self.box_points[0, i_b]
                self.box_axi[2, i_b] = self.box_points[2, i_b] - self.box_points[0, i_b]

                if ti.abs(rnorm[2]) < gs.EPS:
                    is_return = True
                if not is_return:
                    innorm = (1 / rnorm[2]) * (-1 if in_ else 1)

                    for i in ti.static(range(4)):
                        c1 = -self.box_points[i, i_b][2] / rnorm[2]
                        self.box_pu[i, i_b] = self.box_points[i, i_b]
                        self.box_points[i, i_b] = self.box_points[i, i_b] + c1 * rnorm

                        self.box_ppts2[i, 0, i_b] = self.box_points[i, i_b][0]
                        self.box_ppts2[i, 1, i_b] = self.box_points[i, i_b][1]
                    self.box_pts[0, i_b] = self.box_points[0, i_b]
                    self.box_pts[1, i_b] = self.box_points[1, i_b] - self.box_points[0, i_b]
                    self.box_pts[2, i_b] = self.box_points[2, i_b] - self.box_points[0, i_b]

                    m = 3
                    k = 0
                    n = 0

                    if m > 1:
                        self.box_lines[k, i_b][0:3] = self.box_pts[0, i_b]
                        self.box_lines[k, i_b][3:6] = self.box_pts[1, i_b]
                        self.box_linesu[k, i_b][0:3] = self.box_axi[0, i_b]
                        self.box_linesu[k, i_b][3:6] = self.box_axi[1, i_b]
                        k = k + 1

                    if m > 2:
                        self.box_lines[k, i_b][0:3] = self.box_pts[0, i_b]
                        self.box_lines[k, i_b][3:6] = self.box_pts[2, i_b]
                        self.box_linesu[k, i_b][0:3] = self.box_axi[0, i_b]
                        self.box_linesu[k, i_b][3:6] = self.box_axi[2, i_b]
                        k = k + 1

                        self.box_lines[k, i_b][0:3] = self.box_pts[0, i_b] + self.box_pts[1, i_b]
                        self.box_lines[k, i_b][3:6] = self.box_pts[2, i_b]
                        self.box_linesu[k, i_b][0:3] = self.box_axi[0, i_b] + self.box_axi[1, i_b]
                        self.box_linesu[k, i_b][3:6] = self.box_axi[2, i_b]
                        k = k + 1

                        self.box_lines[k, i_b][0:3] = self.box_pts[0, i_b] + self.box_pts[2, i_b]
                        self.box_lines[k, i_b][3:6] = self.box_pts[1, i_b]
                        self.box_linesu[k, i_b][0:3] = self.box_axi[0, i_b] + self.box_axi[2, i_b]
                        self.box_linesu[k, i_b][3:6] = self.box_axi[1, i_b]
                        k = k + 1

                    for i in range(k):
                        for q in ti.static(range(2)):
                            a = self.box_lines[i, i_b][q]
                            b = self.box_lines[i, i_b][q + 3]
                            c = self.box_lines[i, i_b][1 - q]
                            d = self.box_lines[i, i_b][4 - q]

                            if ti.abs(b) > gs.EPS:
                                for _j in ti.static(range(2)):
                                    j = 2 * _j - 1
                                    if n < self.box_MAXCONPAIR:
                                        l = s[q] * j
                                        c1 = (l - a) / b
                                        if 0 <= c1 and c1 <= 1:
                                            c2 = c + d * c1
                                            if (ti.abs(c2) <= s[1 - q]) and (
                                                (self.box_linesu[i, i_b][2] + self.box_linesu[i, i_b][5] * c1) * innorm
                                                <= margin
                                            ):
                                                self.box_points[n, i_b] = (
                                                    self.box_linesu[i, i_b][0:3] * 0.5
                                                    + c1 * 0.5 * self.box_linesu[i, i_b][3:6]
                                                )
                                                self.box_points[n, i_b][q] = self.box_points[n, i_b][q] + 0.5 * l
                                                self.box_points[n, i_b][1 - q] = (
                                                    self.box_points[n, i_b][1 - q] + 0.5 * c2
                                                )
                                                self.box_depth[n, i_b] = self.box_points[n, i_b][2] * innorm * 2
                                                n = n + 1

                    nl = n
                    a = self.box_pts[1, i_b][0]
                    b = self.box_pts[2, i_b][0]
                    c = self.box_pts[1, i_b][1]
                    d = self.box_pts[2, i_b][1]
                    c1 = a * d - b * c

                    for i in range(4):
                        if n < self.box_MAXCONPAIR:
                            llx = lx if (i // 2) else -lx
                            lly = ly if (i % 2) else -ly

                            x = llx - self.box_pts[0, i_b][0]
                            y = lly - self.box_pts[0, i_b][1]

                            u = (x * d - y * b) / c1
                            v = (y * a - x * c) / c1

                            if nl == 0:
                                if (u < 0 or u > 1) and (v < 0 or v > 1):
                                    continue
                            elif u < 0 or u > 1 or v < 0 or v > 1:
                                continue

                            u = ti.math.clamp(u, 0, 1)
                            v = ti.math.clamp(v, 0, 1)
                            tmp1 = self.box_pu[0, i_b] * (1 - u - v) + self.box_pu[1, i_b] * u + self.box_pu[2, i_b] * v
                            self.box_points[n, i_b][0] = llx
                            self.box_points[n, i_b][1] = lly
                            self.box_points[n, i_b][2] = 0

                            tmp2 = self.box_points[n, i_b] - tmp1

                            c2 = tmp2.dot(tmp2)

                            if not (tmp1[2] > 0 and c2 > margin2):
                                self.box_points[n, i_b] = self.box_points[n, i_b] + tmp1
                                self.box_points[n, i_b] = self.box_points[n, i_b] * 0.5

                                self.box_depth[n, i_b] = ti.sqrt(c2) * (-1 if tmp1[2] < 0 else 1)
                                n = n + 1

                    nf = n

                    for i in range(4):
                        if n < self.box_MAXCONPAIR:
                            x, y = self.box_ppts2[i, 0, i_b], self.box_ppts2[i, 1, i_b]

                            if nl == 0:
                                if (nf != 0) and (x < -lx or x > lx) and (y < -ly or y > ly):
                                    continue
                            elif x < -lx or x > lx or y < -ly or y > ly:
                                continue

                            c1 = 0
                            for j in ti.static(range(2)):
                                if self.box_ppts2[i, j, i_b] < -s[j]:
                                    c1 = c1 + (self.box_ppts2[i, j, i_b] + s[j]) ** 2
                                elif self.box_ppts2[i, j, i_b] > s[j]:
                                    c1 = c1 + (self.box_ppts2[i, j, i_b] - s[j]) ** 2

                            c1 = c1 + (self.box_pu[i, i_b][2] * innorm) ** 2

                            if self.box_pu[i, i_b][2] > 0 and c1 > margin2:
                                continue

                            tmp1 = ti.Vector(
                                [self.box_ppts2[i, 0, i_b] * 0.5, self.box_ppts2[i, 1, i_b] * 0.5, 0], dt=gs.ti_float
                            )

                            for j in ti.static(range(2)):
                                if self.box_ppts2[i, j, i_b] < -s[j]:
                                    tmp1[j] = -s[j] * 0.5
                                elif self.box_ppts2[i, j, i_b] > s[j]:
                                    tmp1[j] = s[j] * 0.5

                            tmp1 = tmp1 + self.box_pu[i, i_b] * 0.5
                            self.box_points[n, i_b] = tmp1

                            self.box_depth[n, i_b] = ti.sqrt(c1) * (-1 if self.box_pu[i, i_b][2] < 0 else 1)
                            n = n + 1

                    r = mat1 @ rotmore.transpose()

                    tmp1 = r @ rnorm
                    normal_0 = tmp1 * (-1 if in_ else 1)

                    for i in range(n):
                        dist = self.box_depth[i, i_b]
                        self.box_points[i, i_b][2] = self.box_points[i, i_b][2] + hz
                        tmp2 = r @ self.box_points[i, i_b]
                        contact_pos = tmp2 + pos1
                        self._func_add_contact(i_ga, i_gb, -normal_0, contact_pos, -dist, i_b)

    def get_contacts(self, as_tensor: bool = True, to_torch: bool = True):
        # Early return if already pre-computed
        contacts_info = self._contacts_info_cache.get((as_tensor, to_torch))
        if contacts_info is not None:
            return contacts_info.copy()

        # Find out how much dynamic memory must be allocated
        n_contacts = tuple(self.n_contacts.to_numpy())
        n_envs = len(n_contacts)
        n_contacts_max = max(n_contacts)
        if as_tensor:
            out_size = n_contacts_max * n_envs
        else:
            *n_contacts_starts, out_size = np.cumsum(n_contacts)

        # Allocate output buffer
        if to_torch:
            iout = torch.full((out_size, 4), -1, dtype=gs.tc_int, device=gs.device)
            fout = torch.zeros((out_size, 10), dtype=gs.tc_float, device=gs.device)
        else:
            iout = np.full((out_size, 4), -1, dtype=gs.np_int)
            fout = np.zeros((out_size, 10), dtype=gs.np_float)

        # Copy contact data
        if n_contacts_max > 0:
            self._kernel_get_contacts(as_tensor, iout, fout)

        # Build structured view (no copy)
        if as_tensor:
            if self._solver.n_envs > 0:
                iout = iout.reshape((n_envs, n_contacts_max, 4))
                fout = fout.reshape((n_envs, n_contacts_max, 10))
            iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
            fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
            values = (*iout_chunks, *fout_chunks)
        else:
            # Split smallest dimension first, then largest dimension
            if self._solver.n_envs == 0:
                iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
                fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
                values = (*iout_chunks, *fout_chunks)
            elif n_contacts_max >= n_envs:
                if to_torch:
                    iout_chunks = torch.split(iout, n_contacts)
                    fout_chunks = torch.split(fout, n_contacts)
                else:
                    iout_chunks = np.split(iout, n_contacts_starts)
                    fout_chunks = np.split(fout, n_contacts_starts)
                iout_chunks = ((out[..., 0], out[..., 1], out[..., 2], out[..., 3]) for out in iout_chunks)
                fout_chunks = ((out[..., 0], out[..., 1:4], out[..., 4:7], out[..., 7:]) for out in fout_chunks)
                values = (*zip(*iout_chunks), *zip(*fout_chunks))
            else:
                iout_chunks = (iout[..., 0], iout[..., 1], iout[..., 2], iout[..., 3])
                fout_chunks = (fout[..., 0], fout[..., 1:4], fout[..., 4:7], fout[..., 7:])
                if self._solver.n_envs == 1:
                    values = [(value,) for value in (*iout_chunks, *fout_chunks)]
                else:
                    if to_torch:
                        iout_chunks = (torch.split(out, n_contacts) for out in iout_chunks)
                        fout_chunks = (torch.split(out, n_contacts) for out in fout_chunks)
                    else:
                        iout_chunks = (np.split(out, n_contacts_starts) for out in iout_chunks)
                        fout_chunks = (np.split(out, n_contacts_starts) for out in fout_chunks)
                    values = (*iout_chunks, *fout_chunks)

        contacts_info = dict(
            zip(("link_a", "link_b", "geom_a", "geom_b", "penetration", "position", "normal", "force"), values)
        )

        # Cache contact information before returning
        self._contacts_info_cache[(as_tensor, to_torch)] = contacts_info

        return contacts_info.copy()

    @ti.kernel
    def _kernel_get_contacts(self, is_padded: ti.template(), iout: ti.types.ndarray(), fout: ti.types.ndarray()):
        n_contacts_max = gs.ti_int(0)
        for i_b in range(self._solver._B):
            n_contacts = self.n_contacts[i_b]
            if n_contacts > n_contacts_max:
                n_contacts_max = n_contacts

        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._solver._B):
            i_c_start = gs.ti_int(0)
            if ti.static(is_padded):
                i_c_start = i_b * n_contacts_max
            else:
                for j_b in range(i_b):
                    i_c_start = i_c_start + self.n_contacts[j_b]

            for i_c_ in range(self.n_contacts[i_b]):
                i_c = i_c_start + i_c_

                iout[i_c, 0] = self.contact_data[i_c_, i_b].link_a
                iout[i_c, 1] = self.contact_data[i_c_, i_b].link_b
                iout[i_c, 2] = self.contact_data[i_c_, i_b].geom_a
                iout[i_c, 3] = self.contact_data[i_c_, i_b].geom_b
                fout[i_c, 0] = self.contact_data[i_c_, i_b].penetration
                for j in ti.static(range(3)):
                    fout[i_c, 1 + j] = self.contact_data[i_c_, i_b].pos[j]
                    fout[i_c, 4 + j] = self.contact_data[i_c_, i_b].normal[j]
                    fout[i_c, 7 + j] = self.contact_data[i_c_, i_b].force[j]
