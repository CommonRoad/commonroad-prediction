import numpy as np
from commonroad.prediction.prediction import Occupancy
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import CustomState

from crpred.advanced_models.agent import Agent
from crpred.utility.config import IDMParams


class IDMAgent(Agent):
    """Intelligent Driver Model Predictor."""

    def __init__(self, obstacle: DynamicObstacle, sc: Scenario, config: IDMParams = IDMParams()):
        super().__init__(obstacle, sc)

        self._list_lanelets_merged = []
        self._dict_clcs = {}
        self._dict_lanelet_merge_ids = {}

        self._retrieve_merged_lanelet_ids_and_create_clcs()

        self._clcs_main = self._dict_clcs[self._list_lanelets_merged[0].lanelet_id]

        self._config = config

    def step_forward(self, time_step: int):
        if self._obstacle.initial_state.time_step >= time_step or not self._valid:
            return

        state_new = self._calculate_new_state(time_step)
        if not state_new:
            self._state_current = None
            self._occ_current = None
            self._set_ids_lanelets_current = set()
            self._valid = False

            return

        else:
            # add new state and related attributes
            shape_new = self._obstacle.obstacle_shape.rotate_translate_local(state_new.position, state_new.orientation)
            occ_new = Occupancy(time_step, shape_new)
            self._traj_state_list.append(state_new)
            self._pred_occ.append(occ_new)
            self._state_current = state_new
            self._occ_current = occ_new
            self._set_ids_lanelets_current = set(
                self._current_scenario.lanelet_network.find_lanelet_by_shape(self._occ_current.shape)
            )

    def _calculate_new_state(self, time_step: int):
        try:
            # update leader info at the previous time step
            self._update_leader_at_time_step(time_step - 1)

            # convert state to curvilinear coordinate system
            state_current = self._state_current
            p_lon_current, p_lat_current, v_lon_current, v_lat_current, o_ref = self._convert_to_curvilinear_state(
                state_current, self._clcs_main
            )

            a_lon_new = self._calculate_acceleration(v_lon_current)

            # covered distance along the center line of the lanelet
            dist_p_lon = 0.5 * a_lon_new * self._current_scenario.dt**2 + v_lon_current * self._current_scenario.dt
            p_lon_new = p_lon_current + dist_p_lon
            # shift lateral position towards the centerline
            p_lat_new = p_lat_current + (-0.0 * np.sign(p_lat_current) if abs(p_lat_current) > 0.2 else 0)

            v_lon_new = v_lon_current + a_lon_new * self._current_scenario.dt
            v_lat_new = v_lat_current + (-0.1 * np.sign(v_lat_current) if abs(v_lat_current > 0.2) else 0)
            v_new = np.sqrt(v_lon_new**2 + v_lat_new**2)

            # new position in Cartesian coordinate system
            x_new, y_new = self._clcs_main.convert_to_cartesian_coords(p_lon_new, p_lat_new)
            position_new = np.array([x_new, y_new])

            # steers towards the centerline
            diff_o = state_current.orientation - o_ref
            o_new = state_current.orientation + (-0.05 * np.sign(diff_o) if abs(diff_o) > 0.05 else 0)

            # create new state
            state_new = CustomState(
                position=position_new, orientation=o_new, velocity=v_new, acceleration=a_lon_new, time_step=time_step
            )

            return state_new

        except ValueError:
            obstacle = self._obstacle
            for lanelet in self._current_scenario.lanelet_network.lanelets:
                try:
                    lanelet.dynamic_obstacles_on_lanelet.get(time_step - 1).discard(obstacle.obstacle_id)

                except AttributeError:
                    lanelet.dynamic_obstacles_on_lanelet[time_step - 1] = set()
            return None

    def _update_leader_at_time_step(self, time_step: int):
        """
        Updates leader agent with the minimum distance to the ego vehicle.
        """
        self._id_agent_leader = self.dis_to_leader = self.rate_approach = None

        obs_ego = self._obstacle
        state_ego = self.state_at_step(time_step, obs_ego, self._traj_state_list)

        # iterate through lanelet and its obstacles, find the one with the minimum positive distance to ego
        dist_to_leader_min = np.infty
        id_leader = None
        for lanelet in self._list_lanelets_merged:
            clcs = self._dict_clcs[lanelet.lanelet_id]
            p_lon_ego, _ = clcs.convert_to_curvilinear_coords(state_ego.position[0], state_ego.position[1])

            set_ids_obstacles_in_lanelet = self._dynamic_obstacles_in_lanelet_set(
                self._dict_lanelet_merge_ids[lanelet.lanelet_id], time_step
            )
            set_ids_obstacles_in_lanelet.difference_update({self._obstacle.obstacle_id})
            list_ids_obstacles_in_lanelet = list(set_ids_obstacles_in_lanelet)

            if not list_ids_obstacles_in_lanelet:
                continue

            # list of distances along the lanelet spline
            list_tuples_obstacles_in_lanelet = []
            for id_obs in list_ids_obstacles_in_lanelet:
                obs_leader = self._current_scenario.obstacle_by_id(id_obs)

                # workaround for bug in deletion of removed obstacles from lanelet
                if not obs_leader:
                    continue

                if time_step == obs_leader.initial_state.time_step:
                    state_obs = obs_leader.initial_state
                else:
                    state_obs = None
                    for state in obs_leader.prediction.trajectory.state_list:
                        if state.time_step == time_step:
                            state_obs = state
                            break
                    if not state_obs:
                        continue

                try:
                    p_lon_obs, _ = clcs.convert_to_curvilinear_coords(state_obs.position[0], state_obs.position[1])

                except ValueError:
                    continue

                dist_to_obs = p_lon_obs - p_lon_ego

                if dist_to_obs > 0:
                    list_tuples_obstacles_in_lanelet.append((dist_to_obs, id_obs))

            if list_tuples_obstacles_in_lanelet:
                dist_to_leader_min_temp, id_obs_min = min(list_tuples_obstacles_in_lanelet)

                if dist_to_leader_min_temp < dist_to_leader_min:
                    # new smallest distance
                    dist_to_leader_min = dist_to_leader_min_temp
                    id_leader = id_obs_min

        if not id_leader:
            # no leader in the current lanelet, return None
            return

        # calculate the approaching rate
        obs_follow = obs_ego
        obs_lead = self._current_scenario.obstacle_by_id(id_leader)
        rate_approaching = self.calculate_approaching_rate(obs_follow, obs_lead, time_step)

        self._id_agent_leader = id_leader
        self._dis_to_leader = dist_to_leader_min
        self._rate_approach = rate_approaching

    def _calculate_acceleration(self, v_lon_current: float):
        """
        This method calculates the new acceleration depending on the leading vehicle and the desired velocity
        """
        # in standstill
        if self._config.v_desired == 0:
            if v_lon_current > 0:
                return self._config.a_lon_min

            else:
                return 0

        # free road term
        a_free = self._config.a_lon_max * (1 - (v_lon_current / self._config.v_desired) ** self._config.coef)

        # interaction term
        if self._id_agent_leader:
            term_1 = self._config.dist_min
            term_2 = v_lon_current * self._config.time_headway
            term_3 = (v_lon_current * self._rate_approach) / (
                2 * np.sqrt(self._config.a_lon_max * self._config.a_lon_comfort)
            )
            a_interact = -self._config.a_lon_max * ((term_1 + term_2 + term_3) / self._dis_to_leader) ** 2

        else:
            a_interact = 0

        a_sum = a_free + a_interact
        # disable going backwards
        if v_lon_current <= 0 and a_sum <= 0:
            return 0

        return min(max(a_sum, self._config.a_lon_min), self._config.a_lon_max)

    def _retrieve_merged_lanelet_ids_and_create_clcs(self):
        """
        Retrieve all lanelets by merging the successor lanelets, create curvilinear coordinate systems.
        """
        set_ids_lanelet_current = self._set_ids_lanelets_current
        list_all_lanelets_merged = self._list_lanelets_merged
        dict_all_clcs = self._dict_clcs
        dict_all_lanelet_merge_ids = self._dict_lanelet_merge_ids
        for id_lanelet_current in set_ids_lanelet_current:
            lanelet_current = self._current_scenario.lanelet_network.find_lanelet_by_id(id_lanelet_current)

            # only create one merged lanelet and CLCS for longitudinally adjacent current lanelets
            if any(lanelet_id in set_ids_lanelet_current for lanelet_id in lanelet_current.predecessor):
                continue

            list_lanelets_merged, dict_clcs, dict_merge_ids = self._merge_lanelet_and_create_clcs(
                lanelet=lanelet_current, merge_predecessors=False
            )
            dict_all_clcs = {**dict_all_clcs, **dict_clcs}
            dict_all_lanelet_merge_ids = {**dict_all_lanelet_merge_ids, **dict_merge_ids}
            list_all_lanelets_merged += list_lanelets_merged

        self._dict_clcs = dict_all_clcs
        self._list_lanelets_merged = list_all_lanelets_merged
        self._dict_lanelet_merge_ids = dict_all_lanelet_merge_ids
