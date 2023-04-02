import abc
import copy
from typing import List, Set, Dict
import numpy as np

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.state import TraceState
from commonroad.prediction.prediction import TrajectoryPrediction, Trajectory
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from commonroad_dc.geometry.util import compute_orientation_from_polyline, compute_pathlength_from_polyline,\
    chaikins_corner_cutting, resample_polyline

from crpred.advanced_models.utility.lanelets import all_lanelets_by_merging_predecessors_from_lanelet


class Agent:
    """Agent class for advanced prediction models."""

    def __init__(self, obstacle: DynamicObstacle, sc: Scenario):
        self._obstacle: DynamicObstacle = obstacle
        self._valid = True
        self._current_scenario = sc
        self._state_current = copy.deepcopy(self._obstacle.initial_state)
        self._occ_current = obstacle.occupancy_at_time(self._state_current.time_step)
        self._traj_state_list = []
        self._pred_occ = []
        self._set_ids_lanelets_current = \
            self._current_scenario.lanelet_network.find_lanelet_by_shape(self._occ_current.shape)

    @property
    def agent_id(self):
        """Obstacle ID"""
        return self._obstacle.obstacle_id

    @property
    def time_step_current(self):
        """Last time step of prediction"""
        return self._state_current.time_step

    @property
    def set_ids_lanelets_current(self):
        """IDs of occupied lanelets at current state."""
        return self._set_ids_lanelets_current

    def create_cr_trajectory_prediction(self) -> TrajectoryPrediction:
        """
        Creates CommonRoad trajectory prediction from state list.

        :return: CommonRoad trajectory prediction.
        """
        return TrajectoryPrediction(Trajectory(self._traj_state_list[0].time_step, self._traj_state_list),
                                    self._obstacle.obstacle_shape)

    def _merge_lanelet_and_create_clcs(self, lanelet: Lanelet, merge_predecessors: bool = True):
        # get merged lanelets
        list_lanelets_merged, dict_merge_ids = self._retrieve_merged_lanelets(lanelet, merge_predecessors)

        # create CLCS
        dict_clcs = {}
        for lanelet in list_lanelets_merged:
            # pre-process reference path
            ref_path = np.array(chaikins_corner_cutting(polyline=lanelet.center_vertices, refinements=1))
            ref_path = resample_polyline(ref_path, 0.5)

            clcs = CurvilinearCoordinateSystem(ref_path)
            clcs.compute_and_set_curvature()
            dict_clcs[lanelet.lanelet_id] = clcs

        return list_lanelets_merged, dict_clcs, dict_merge_ids

    def _retrieve_merged_lanelets(self, lanelet: Lanelet, merge_predecessors: bool = True):
        list_lanelets_merged_suc, list_suc_merge_ids = \
            Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                lanelet=lanelet,
                network=self._current_scenario.lanelet_network,
                max_length=300.0)

        list_lanelets_merged = []
        dict_merge_ids = {}
        if merge_predecessors:
            for lnlt, suc_merge_ids in zip(list_lanelets_merged_suc, list_suc_merge_ids):
                list_lanelets_merged_pred, list_pred_merge_ids = \
                    all_lanelets_by_merging_predecessors_from_lanelet(lnlt,
                                                                      self._current_scenario.lanelet_network,
                                                                      300.0)

                for lanelet_merged, pred_merge_ids in zip(list_lanelets_merged_pred, list_pred_merge_ids):
                    set_merge_ids = set(pred_merge_ids + suc_merge_ids)
                    dict_merge_ids[lanelet_merged.lanelet_id] = set_merge_ids

                list_lanelets_merged += list_lanelets_merged_pred

        else:
            for lanelet_merged, merge_ids in zip(list_lanelets_merged_suc, list_suc_merge_ids):
                dict_merge_ids[lanelet_merged.lanelet_id] = set(merge_ids)

            list_lanelets_merged = list_lanelets_merged_suc

        return list_lanelets_merged, dict_merge_ids

    def _find_followers_and_leaders_at_time_step(self, time_step: int, list_lanelets: List[Lanelet],
                                                 sc: Scenario,
                                                 dict_clcs: Dict[int, CurvilinearCoordinateSystem],
                                                 dict_merge_ids: Dict[int, Set[int]], state_ego: TraceState):
        obs_ego = self._obstacle
        dist_to_leader_min = dist_to_follower_min = np.infty
        leader_clcs = None
        id_leader = id_follower = None
        rate_approaching_leader = rate_approaching_follower = np.infty

        for lanelet in list_lanelets:
            clcs = dict_clcs[lanelet.lanelet_id]
            p_lon_ego, _ = clcs.convert_to_curvilinear_coords(state_ego.position[0], state_ego.position[1])

            set_ids_obstacles_in_lanelet = \
                self._dynamic_obstacles_in_lanelet_set(dict_merge_ids[lanelet.lanelet_id], time_step)
            set_ids_obstacles_in_lanelet.difference_update({self._obstacle.obstacle_id})
            list_ids_obstacles_in_lanelet = list(set_ids_obstacles_in_lanelet)

            if not list_ids_obstacles_in_lanelet:
                continue

            # list of distances along the lanelet spline
            list_tuples_obstacles_in_lanelet_leader = []
            list_tuples_obstacles_in_lanelet_follower = []
            for id_obs in list_ids_obstacles_in_lanelet:
                obs_candidate = sc.obstacle_by_id(id_obs)

                if time_step == 0:
                    state_obs = obs_candidate.initial_state
                else:
                    state_obs = obs_candidate.prediction.trajectory.state_list[time_step - 1]

                try:
                    p_lon_obs, _ = clcs.convert_to_curvilinear_coords(state_obs.position[0], state_obs.position[1])
                except ValueError:
                    continue

                dist_to_obs_leader = p_lon_obs - p_lon_ego
                dist_to_obs_follower = p_lon_ego - p_lon_obs

                if dist_to_obs_leader > 0:
                    list_tuples_obstacles_in_lanelet_leader.append((dist_to_obs_leader, id_obs))

                if dist_to_obs_follower > 0:
                    list_tuples_obstacles_in_lanelet_follower.append((dist_to_obs_follower, id_obs))

            if list_tuples_obstacles_in_lanelet_leader:
                dist_to_leader_min_temp, id_obs_min = min(list_tuples_obstacles_in_lanelet_leader)

                if dist_to_leader_min_temp < dist_to_leader_min:
                    # new smallest distance
                    dist_to_leader_min = dist_to_leader_min_temp
                    id_leader = id_obs_min
                    leader_clcs = clcs

            if list_tuples_obstacles_in_lanelet_follower:
                dist_to_follower_min_temp, id_obs_min = min(list_tuples_obstacles_in_lanelet_follower)

                if dist_to_follower_min_temp < dist_to_follower_min:
                    # new smallest distance
                    dist_to_follower_min = dist_to_follower_min_temp
                    id_follower = id_obs_min

        if id_leader:
            # calculate the approaching rate
            obs_follow = obs_ego
            obs_lead = sc.obstacle_by_id(id_leader)
            rate_approaching_leader = self.calculate_approaching_rate(obs_follow, obs_lead, time_step)

        if id_follower:
            # calculate the approaching rate
            obs_follow = sc.obstacle_by_id(id_follower)
            obs_lead = obs_ego
            rate_approaching_follower = self.calculate_approaching_rate(obs_follow, obs_lead, time_step)

        return leader_clcs, id_leader, dist_to_leader_min, rate_approaching_leader, \
            id_follower, dist_to_follower_min, rate_approaching_follower

    def _dynamic_obstacles_in_lanelet_set(self, set_lanelet_ids: Set[int], time_step: int):
        """
        Returns the set of all ids of obstacles that are in set_lanelet_ids at time_step.
        """
        set_ids_obstacles_in_lanelet_set = set()
        for l_id in set_lanelet_ids:
            lanelet = self._current_scenario.lanelet_network.find_lanelet_by_id(l_id)
            if lanelet:
                set_obstacle_ids = copy.copy(lanelet.dynamic_obstacle_by_time_step(time_step))
                set_ids_obstacles_in_lanelet_set = set_ids_obstacles_in_lanelet_set.union(set_obstacle_ids)

        return set_ids_obstacles_in_lanelet_set

    @staticmethod
    def _convert_to_curvilinear_state(state: TraceState, clcs: CurvilinearCoordinateSystem):
        x, y = state.position
        o = state.orientation
        v = state.velocity

        p_lon, p_lat = clcs.convert_to_curvilinear_coords(x, y)
        ref_path = np.array(clcs.reference_path())
        orientation_ref = compute_orientation_from_polyline(ref_path)
        length_ref = compute_pathlength_from_polyline(ref_path)

        o_ref = np.interp(p_lon, length_ref, orientation_ref)
        v_lon = v * np.cos(o - o_ref)
        v_lat = v * np.sin(o - o_ref)

        return p_lon, p_lat, v_lon, v_lat, o_ref

    @staticmethod
    def state_at_step(step: int, obstacle: DynamicObstacle, state_list: List[TraceState]):
        if step < obstacle.initial_state.time_step:
            return None

        elif step == obstacle.initial_state.time_step:
            return obstacle.initial_state

        else:
            return state_list[step - obstacle.initial_state.time_step - 1]

    @staticmethod
    def calculate_approaching_rate(obs_follower: DynamicObstacle, obs_leader: DynamicObstacle, time_step: int):
        assert time_step >= obs_follower.initial_state.time_step and time_step >= obs_leader.initial_state.time_step, \
            "<Prediction Agent>: time_step out of range"

        follower_initial_time_step = obs_follower.initial_state.time_step
        leader_initial_time_step = obs_leader.initial_state.time_step
        if time_step > follower_initial_time_step and time_step > leader_initial_time_step:
            v_follower = \
                obs_follower.prediction.trajectory.state_list[time_step - follower_initial_time_step - 1].velocity
            v_leader = obs_leader.prediction.trajectory.state_list[time_step - leader_initial_time_step - 1].velocity

        elif time_step > follower_initial_time_step:
            v_follower = \
                obs_follower.prediction.trajectory.state_list[time_step - follower_initial_time_step - 1].velocity
            v_leader = obs_leader.initial_state.velocity

        elif time_step > leader_initial_time_step:
            v_follower = obs_follower.initial_state.velocity
            v_leader = obs_leader.prediction.trajectory.state_list[time_step - leader_initial_time_step - 1].velocity

        else:
            v_follower = obs_follower.initial_state.velocity
            v_leader = obs_leader.initial_state.velocity

        return v_follower - v_leader

    @abc.abstractmethod
    def step_forward(self, time_step: int):
        pass
