import copy

import numpy as np
from commonroad.prediction.prediction import Occupancy
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import State

from crpred.advanced_models.agent import Agent
from crpred.utility.config import MobilParams


class MOBILAgent(Agent):
    """MOBIL Predictor."""

    def __init__(self, obstacle: DynamicObstacle, dt: float = 0.1, config: MobilParams = MobilParams()):
        # initialize the parent class
        super().__init__(obstacle, dt)

        self.dict_all_merged_lanelets = {}
        self.dict_all_merged_lanelets_containing_id = {}

        self.dict_list_lanelets_merged_ego = {}
        self.dict_clcs_ego = {}
        self.dict_lanelet_merge_ids_ego = {}

        self.list_lanelets_merged_left = []
        self.dict_clcs_left = {}
        self.dict_lanelet_merge_ids_left = {}

        self.list_lanelets_merged_right = []
        self.dict_clcs_right = {}
        self.dict_lanelet_merge_ids_right = {}

        self.set_ids_lanelets_current_pred_removed = set()

        self._retrieve_merged_lanelet_ids_and_create_clcs()

        main_lanelet_id = list(self.set_ids_lanelets_current_pred_removed)[0]

        self.clcs_main = \
            self.dict_clcs_ego[main_lanelet_id][self.dict_list_lanelets_merged_ego[main_lanelet_id][0].lanelet_id]

    def step_forward(self, time_step: int):
        if self.time_step_initial >= time_step or not self.valid:
            return

        state_new = self._calculate_new_state(time_step)
        if not state_new:
            self.state_current = None
            self.time_step_current = None
            self.occ_current = None
            self.set_ids_lanelets_current = set()
            self.valid = False

            return

        else:
            # add new state and related attributes
            shape_new = self.shape.rotate_translate_local(state_new.position, state_new.orientation)
            occ_new = Occupancy(time_step, shape_new)

            self.obstacle.prediction.trajectory.state_list.append(state_new)
            self.obstacle.prediction.occupancy_set.append(occ_new)
            self.state_current = state_new
            self.time_step_current = state_new.time_step
            self.occ_current = occ_new

            ids_lanelets_old = copy.copy(self.set_ids_lanelets_current)
            self.set_ids_lanelets_current = \
                set(self.cls_scenario_current.lanelet_network.find_lanelet_by_shape(self.occ_current.shape))

            # update lanelet info
            if ids_lanelets_old != self.set_ids_lanelets_current:

                self.dict_list_lanelets_merged_ego = {}
                self.dict_clcs_ego = {}
                self.dict_lanelet_merge_ids_ego = {}

                self.list_lanelets_merged_left = []
                self.dict_clcs_left = {}
                self.dict_lanelet_merge_ids_left = {}

                self.list_lanelets_merged_right = []
                self.dict_clcs_right = {}
                self.dict_lanelet_merge_ids_right = {}

                self.set_ids_lanelets_current_pred_removed = set()
                self._retrieve_merged_lanelet_ids_and_create_clcs()

    def _calculate_new_state(self, time_step: int):
        try:
            # update leader info at the previous time step
            self._update_followers_and_leaders_at_time_step(time_step - 1)

            # convert state to curvilinear coordinate system
            ego_state_current = self.state_current
            p_lon_current, p_lat_current, v_lon_current, v_lat_current, o_ref_current = \
                self._convert_to_curvilinear_state(ego_state_current, self.ego_clcs)

            a_ego_current = self.a_ego_current

            v_lon_follower_old_current = 0
            if self.id_follower:
                obs_follower_old_current = self.cls_scenario_current.obstacle_by_id(self.id_follower)
                follower_old_state_current = self.state_at_step(time_step-1, obs_follower_old_current)

                p_lon_follower_old_current, p_lat_follower_old_current, v_lon_follower_old_current, \
                    v_lat_follower_old_current, o_ref_follower_old_current = \
                    self._convert_to_curvilinear_state(follower_old_state_current, self.ego_clcs)

            a_follower_old_current = self._calculate_acceleration(self.id_follower, v_lon_follower_old_current,
                                                                  self.id_agent, self.dist_to_follower_min,
                                                                  self.rate_approaching_follower)

            a_follower_old_change = self._calculate_acceleration(self.id_follower, v_lon_follower_old_current,
                                                                 self.id_leader,
                                                                 self.dist_to_leader_min + self.dist_to_follower_min,
                                                                 self.rate_approaching_leader +
                                                                 self.rate_approaching_follower)

            if self.left_clcs:
                v_lon_follower_left_current = 0
                if self.id_left_follower:
                    obs_follower_left_current = self.cls_scenario_current.obstacle_by_id(self.id_left_follower)
                    follower_left_state_current = self.state_at_step(time_step - 1, obs_follower_left_current)

                    p_lon_follower_left_current, p_lat_follower_left_current, v_lon_follower_left_current, \
                        v_lat_follower_left_current, o_ref_follower_left_current = \
                        self._convert_to_curvilinear_state(follower_left_state_current, self.left_clcs)

                a_follower_left_current = self._calculate_acceleration(self.id_left_follower,
                                                                       v_lon_follower_left_current,
                                                                       self.id_left_leader,
                                                                       self.dist_to_left_leader_min +
                                                                       self.dist_to_left_follower_min,
                                                                       self.rate_approaching_left_leader +
                                                                       self.rate_approaching_left_follower)

                # project ego to left clcs
                p_lon_change_left, p_lat_change_left, v_lon_change_left, v_lat_change_left, o_ref_change_left = \
                    self._convert_to_curvilinear_state(ego_state_current, self.left_clcs)

                a_ego_change_left = self._calculate_acceleration(self.id_agent, v_lon_change_left,
                                                                 self.id_left_leader, self.dist_to_left_leader_min,
                                                                 self.rate_approaching_left_leader)

                a_follower_left_change = self._calculate_acceleration(self.id_left_follower,
                                                                      v_lon_follower_left_current,
                                                                      self.id_agent,
                                                                      self.dist_to_left_follower_min,
                                                                      self.rate_approaching_left_follower)

                a_total_change_left = a_ego_change_left - a_ego_current + self.p * (
                            a_follower_left_change - a_follower_left_current +
                            a_follower_old_change - a_follower_old_current)
            else:
                a_total_change_left = 0
                a_follower_left_change = -np.infty

            if self.right_clcs:
                v_lon_follower_right_current = 0
                if self.id_right_follower:
                    obs_follower_right_current = self.cls_scenario_current.obstacle_by_id(self.id_right_follower)
                    follower_right_state_current = self.state_at_step(time_step - 1, obs_follower_right_current)

                    p_lon_follower_right_current, p_lat_follower_right_current, v_lon_follower_right_current, \
                        v_lat_follower_right_current, o_ref_follower_right_current = \
                        self._convert_to_curvilinear_state(follower_right_state_current, self.right_clcs)

                a_follower_right_current = self._calculate_acceleration(self.id_right_follower,
                                                                        v_lon_follower_right_current,
                                                                        self.id_right_leader,
                                                                        self.dist_to_right_leader_min +
                                                                        self.dist_to_right_follower_min,
                                                                        self.rate_approaching_right_leader +
                                                                        self.rate_approaching_right_follower)

                # project ego to right clcs
                p_lon_change_right, p_lat_change_right, v_lon_change_right, v_lat_change_right, o_ref_change_right = \
                    self._convert_to_curvilinear_state(ego_state_current, self.right_clcs)

                a_ego_change_right = self._calculate_acceleration(self.id_agent, v_lon_change_right,
                                                                  self.id_right_leader, self.dist_to_right_leader_min,
                                                                  self.rate_approaching_right_leader)

                a_follower_right_change = self._calculate_acceleration(self.id_right_follower,
                                                                       v_lon_follower_right_current,
                                                                       self.id_agent,
                                                                       self.dist_to_right_follower_min,
                                                                       self.rate_approaching_right_follower)

                a_total_change_right = a_ego_change_right - a_ego_current + self.p * (
                            a_follower_right_change - a_follower_right_current +
                            a_follower_old_change - a_follower_old_current)
            else:
                a_total_change_right = 0
                a_follower_right_change = -np.infty

            if a_total_change_right < a_total_change_left and a_total_change_left > self.a_th and \
                    a_follower_left_change >= -self.b_safe:
                # change to left lane
                self.clcs_main = self.left_clcs
                a_lon_new = a_ego_change_left
                p_lon_current = p_lon_change_left
                p_lat_current = p_lat_change_left
                v_lon_current = v_lon_change_left
                v_lat_current = v_lat_change_left
                o_ref = o_ref_change_left
                lateral_coeff = -0.1

            elif a_total_change_right > self.a_th and a_follower_right_change >= -self.b_safe:
                # change to right lane
                self.clcs_main = self.right_clcs
                a_lon_new = a_ego_change_right
                p_lon_current = p_lon_change_right
                p_lat_current = p_lat_change_right
                v_lon_current = v_lon_change_right
                v_lat_current = v_lat_change_right
                o_ref = o_ref_change_right
                lateral_coeff = -0.1

            else:
                # no lane change
                self.clcs_main = self.ego_clcs
                a_lon_new = a_ego_current
                o_ref = o_ref_current

                # change to the best current lanelet
                if len(self.set_ids_lanelets_current_pred_removed) > 1:
                    lateral_coeff = -0.1
                else:
                    lateral_coeff = -0.0

            orientation = ego_state_current.orientation
            # covered distance along the center line of the lanelet
            dist_p_lon = 0.5 * a_lon_new * self.dt ** 2 + v_lon_current * self.dt
            p_lon_new = p_lon_current + dist_p_lon
            # shift lateral position towards the centerline
            p_lat_new = p_lat_current + (lateral_coeff * np.sign(p_lat_current) if abs(p_lat_current) > 0.2 else 0)

            v_lon_new = v_lon_current + a_lon_new * self.dt
            v_lat_new = v_lat_current + (-0.1 * np.sign(v_lat_current) if abs(v_lat_current > 0.2) else 0)
            v_new = np.sqrt(v_lon_new ** 2 + v_lat_new ** 2)

            # new position in Cartesian coordinate system
            x_new, y_new = self.clcs_main.convert_to_cartesian_coords(p_lon_new, p_lat_new)
            position_new = np.array([x_new, y_new])

            # steers towards the centerline
            diff_o = orientation - o_ref
            o_new = orientation + (-0.05 * np.sign(diff_o) if abs(diff_o) > 0.05 else 0)

            # create new state
            state_new = State(position=position_new, orientation=o_new, velocity=v_new,
                              acceleration=a_lon_new, time_step=time_step)

            return state_new

        except ValueError:
            obstacle = self.obstacle
            for lanelet in self.cls_scenario_current.lanelet_network.lanelets:
                try:
                    lanelet.dynamic_obstacles_on_lanelet.get(time_step - 1).discard(obstacle.obstacle_id)

                except AttributeError:
                    lanelet.dynamic_obstacles_on_lanelet[time_step - 1] = set()
            return None

    def _update_followers_and_leaders_at_time_step(self, time_step: int):
        """
        Updates follower and leader agents with the minimum distance to the ego vehicle.
        """
        state_ego = self.state_current

        # iterate through lanelet and its obstacles, find the one with the minimum distance to ego
        self.left_clcs = self.right_clcs = self.ego_clcs = None

        self.rate_approaching_leader = self.rate_approaching_follower = self.rate_approaching_left_leader = \
            self.rate_approaching_left_follower = self.rate_approaching_right_leader = \
            self.rate_approaching_right_follower = np.infty

        self.dist_to_leader_min = self.dist_to_follower_min = self.dist_to_left_leader_min = \
            self.dist_to_left_follower_min = self.dist_to_right_leader_min = self.dist_to_right_follower_min = np.infty

        self.id_leader = self.id_follower = self.id_left_leader = self.id_left_follower = \
            self.id_right_leader = self.id_right_follower = None

        # find the best current lanelet based on ego acceleration
        a_ego_best = -np.infty

        for lanelet_id in self.set_ids_lanelets_current_pred_removed:
            ego_clcs, id_leader, dist_to_leader_min, rate_approaching_leader, \
                id_follower, dist_to_follower_min, rate_approaching_follower = \
                self._find_followers_and_leaders_at_time_step(time_step, self.dict_list_lanelets_merged_ego[lanelet_id],
                                                              self.dict_clcs_ego[lanelet_id],
                                                              self.dict_lanelet_merge_ids_ego[lanelet_id],
                                                              state_ego)
            if not ego_clcs:
                ego_clcs = self.dict_clcs_ego[lanelet_id][self.dict_list_lanelets_merged_ego[lanelet_id][0].lanelet_id]

            # convert state to curvilinear coordinate system
            p_lon_current, p_lat_current, v_lon_current, v_lat_current, o_ref_current = \
                self._convert_to_curvilinear_state(state_ego, ego_clcs)

            a_ego_temp = self._calculate_acceleration(self.id_agent, v_lon_current,
                                                      id_leader, dist_to_leader_min,
                                                      rate_approaching_leader)
            if a_ego_temp > a_ego_best:
                self.ego_current_lanelet_id = lanelet_id
                self.ego_clcs = ego_clcs
                self.id_leader = id_leader
                self.dist_to_leader_min = dist_to_leader_min
                self.rate_approaching_leader = rate_approaching_leader
                self.id_follower = id_follower
                self.dist_to_follower_min = dist_to_follower_min
                self.rate_approaching_follower = rate_approaching_follower
                a_ego_best = a_ego_temp

        self.a_ego_current = a_ego_best

        self.left_clcs, self.id_left_leader, self.dist_to_left_leader_min, self.rate_approaching_left_leader, \
            self.id_left_follower, self.dist_to_left_follower_min, self.rate_approaching_left_follower = \
            self._find_followers_and_leaders_at_time_step(time_step, self.list_lanelets_merged_left,
                                                          self.dict_clcs_left, self.dict_lanelet_merge_ids_left,
                                                          state_ego)

        self.right_clcs, self.id_right_leader, self.dist_to_right_leader_min, self.rate_approaching_right_leader, \
            self.id_right_follower, self.dist_to_right_follower_min, self.rate_approaching_right_follower = \
            self._find_followers_and_leaders_at_time_step(time_step, self.list_lanelets_merged_right,
                                                          self.dict_clcs_right, self.dict_lanelet_merge_ids_right,
                                                          state_ego)

        if not self.ego_clcs:
            self.ego_clcs = self.clcs_main

        if not self.left_clcs and len(self.list_lanelets_merged_left) > 0:
            self.left_clcs = self.dict_clcs_left[self.list_lanelets_merged_left[0].lanelet_id]

        if not self.right_clcs and len(self.list_lanelets_merged_right) > 0:
            self.right_clcs = self.dict_clcs_right[self.list_lanelets_merged_right[0].lanelet_id]

    def _calculate_acceleration(self, follower_id: int, v_lon_current: float, leader_id: int,
                                dis_to_leader: float, rate_approach: float):
        """
        Get acceleration.
        This method calculates the new acceleration depending on the leading vehicle and the desired velocity
        """
        # This returns 0 instead of None to ensure that the difference between accelerations of follower
        # accelerations is 0 in MOBIL lane change criterion if no follower exists for a given lane
        if not follower_id:
            return 0

        # in standstill
        if self.v_desired == 0:
            if v_lon_current > 0:
                return self.a_lon_min

            else:
                return 0

        # free road term
        a_free = self.a_lon_max * (1 - (v_lon_current / self.v_desired) ** self.coef)

        # interaction term
        if leader_id:
            term_1 = self.dist_keep_min
            term_2 = v_lon_current * self.time_headway
            term_3 = (v_lon_current * rate_approach) / (2 * np.sqrt(self.a_lon_max * self.a_lon_comfort))
            a_interact = -self.a_lon_max * ((term_1 + term_2 + term_3) / dis_to_leader) ** 2

        else:
            a_interact = 0

        a_sum = a_free + a_interact
        # disable going backwards
        if v_lon_current <= 0 and a_sum <= 0:
            return 0

        return min(max(a_sum, self.a_lon_min), self.a_lon_max)

    def _retrieve_merged_lanelet_ids_and_create_clcs(self):
        """
        Retrieve all lanelets by merging the successor and predecessor lanelets, create curvilinear coordinate systems.
        """
        for id_lanelet_current in self.set_ids_lanelets_current:
            lanelet_current = self.cls_scenario_current.lanelet_network.find_lanelet_by_id(id_lanelet_current)

            # only create one merged lanelet and CLCS for longitudinally adjacent current lanelets
            if any(lanelet_id in self.set_ids_lanelets_current for lanelet_id in lanelet_current.successor):
                continue

            self.set_ids_lanelets_current_pred_removed.add(id_lanelet_current)
            list_lanelets_merged, dict_clcs, dict_merge_ids = self._get_merged_lanelet_and_clcs(id_lanelet_current)
            self.dict_clcs_ego[id_lanelet_current] = dict_clcs
            self.dict_lanelet_merge_ids_ego[id_lanelet_current] = dict_merge_ids
            self.dict_list_lanelets_merged_ego[id_lanelet_current] = list_lanelets_merged

            if lanelet_current.adj_left is not None:
                if lanelet_current.adj_left_same_direction and \
                        lanelet_current.adj_left not in self.set_ids_lanelets_current:
                    list_lanelets_merged_left, dict_clcs_left, dict_merge_ids_left = \
                        self._get_merged_lanelet_and_clcs(lanelet_current.adj_left)

                    self.dict_clcs_left = {**self.dict_clcs_left, **dict_clcs_left}
                    self.dict_lanelet_merge_ids_left = {**self.dict_lanelet_merge_ids_left, **dict_merge_ids_left}
                    self.list_lanelets_merged_left += list_lanelets_merged_left

            if lanelet_current.adj_right is not None:
                if lanelet_current.adj_right_same_direction and \
                        lanelet_current.adj_right not in self.set_ids_lanelets_current:
                    list_lanelets_merged_right, dict_clcs_right, dict_merge_ids_right = \
                        self._get_merged_lanelet_and_clcs(lanelet_current.adj_right)

                    self.dict_clcs_right = {**self.dict_clcs_right, **dict_clcs_right}
                    self.dict_lanelet_merge_ids_right = {**self.dict_lanelet_merge_ids_right, **dict_merge_ids_right}
                    self.list_lanelets_merged_right += list_lanelets_merged_right

    def _get_merged_lanelet_and_clcs(self, lanelet_id: int):
        if lanelet_id in self.dict_all_merged_lanelets_containing_id:
            list_ids_lanelets_merged = self.dict_all_merged_lanelets_containing_id[lanelet_id]

            list_lanelets_merged = []
            dict_clcs = {}
            dict_merge_ids = {}
            for merged_id in list_ids_lanelets_merged:
                list_lanelets_merged.append(self.dict_all_merged_lanelets[merged_id]["lanelet"])
                dict_clcs[merged_id] = self.dict_all_merged_lanelets[merged_id]["CLCS"]
                dict_merge_ids[merged_id] = self.dict_all_merged_lanelets[merged_id]["merges"]
        else:
            lnlt = self.cls_scenario_current.lanelet_network.find_lanelet_by_id(lanelet_id)
            list_lanelets_merged, dict_clcs, dict_merge_ids = self._merge_lanelet_and_create_clcs(lnlt)

            for lanelet in list_lanelets_merged:
                if lanelet.lanelet_id in self.dict_all_merged_lanelets:
                    continue
                else:
                    self.dict_all_merged_lanelets[lanelet.lanelet_id] = {}
                    self.dict_all_merged_lanelets[lanelet.lanelet_id]["lanelet"] = lanelet
                    self.dict_all_merged_lanelets[lanelet.lanelet_id]["CLCS"] = dict_clcs[lanelet.lanelet_id]
                    self.dict_all_merged_lanelets[lanelet.lanelet_id]["merges"] = dict_merge_ids[
                        lanelet.lanelet_id]

                    for partial_id in dict_merge_ids[lanelet.lanelet_id]:
                        if partial_id not in self.dict_all_merged_lanelets_containing_id:
                            self.dict_all_merged_lanelets_containing_id[partial_id] = set()
                        self.dict_all_merged_lanelets_containing_id[partial_id].add(lanelet.lanelet_id)

        return list_lanelets_merged, dict_clcs, dict_merge_ids
