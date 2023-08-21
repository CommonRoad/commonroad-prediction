import copy
from typing import Tuple

import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.route_matcher import create_cosy_from_lanelet, get_orientation_at_position
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from crpred.predictor_interface import PredictorInterface
from crpred.utility.common import get_merged_laneletes_from_position
from crpred.utility.config import PredictorParams


class ConstantAccelerationLinearPredictor(PredictorInterface):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super().__init__(config=config)

    def predict(self, sc: Scenario, initial_time_step: int = 0) -> Scenario:
        print()
        pred_sc = copy.deepcopy(sc)

        for idx, dyno in enumerate(sc.dynamic_obstacles):
            pred_state_list = [CustomState(time_step=i) for i in range(self._config.num_steps_prediction)]

            dt = self._config.dt
            if sc.dt != dt:
                print(f"Warning: dt from config ({dt}) is not the same as dt from the scenario ({sc.dt})")

            v_0: float = dyno.initial_state.velocity
            a_0: float = dyno.initial_state.acceleration
            pos_0: np.ndarray = dyno.initial_state.position
            orientation_0: float = dyno.initial_state.orientation

            merged_lanelets, merged_lanelets_id = get_merged_laneletes_from_position(sc.lanelet_network, pos_0)
            curvilinear_cosy: CurvilinearCoordinateSystem = create_cosy_from_lanelet(merged_lanelets[0])
            curvilinear_pos: Tuple[float, float] = curvilinear_cosy.convert_to_curvilinear_coords(pos_0[0], pos_0[1])
            pos_lon, pos_lat = curvilinear_pos

            # Initialization
            ccosy_orientation = get_orientation_at_position(curvilinear_cosy, pos_0)
            orientation_diff = orientation_0 - ccosy_orientation
            v_lon: float = v_0 * np.cos(orientation_diff)
            v_lat: float = v_0 * np.sin(orientation_diff)

            for t in range(initial_time_step, initial_time_step + self._config.num_steps_prediction):
                # Prediction
                pos_lon = pos_lon + np.cos(orientation_diff) * (v_lon + 0.5 * a_0 * dt) * dt
                pos_lat = pos_lat + v_lat * dt

                v_lon = v_lon + a_0 * dt  # You can only accelerate in the longitudinal direction

                # Get the cartesian positions, etc.
                pred_pos = curvilinear_cosy.convert_to_cartesian_coords(pos_lon, pos_lat)
                ccosy_orientation_next = get_orientation_at_position(curvilinear_cosy, pred_pos)
                pred_orientation = ccosy_orientation_next + orientation_diff

                pred_state_list[t].set_value("position", pred_pos)
                pred_state_list[t].set_value("orientation", pred_orientation)
                pred_state_list[t].set_value("velocity", np.sqrt(v_lon**2 + v_lat**2))
                pred_state_list[t].set_value("acceleration", a_0)

            pred_trajectory = Trajectory(pred_state_list[0].time_step, pred_state_list)
            pred_sc.dynamic_obstacles[idx].prediction.trajectory = pred_trajectory

        return pred_sc


class ConstantAccelerationCurvilinearPredictor(PredictorInterface):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super().__init__(config=config)

    def predict(self, sc: Scenario, initial_time_step: int = 0) -> Scenario:
        pred_sc = copy.deepcopy(sc)

        for idx, dyno in enumerate(sc.dynamic_obstacles):
            pred_state_list = [CustomState(time_step=i) for i in range(self._config.num_steps_prediction)]

            dt = self._config.dt
            if sc.dt != dt:
                print(f"Warning: dt from config ({dt}) is not the same as dt from the scenario ({sc.dt})")

            v_0: float = dyno.initial_state.velocity
            a_0: float = dyno.initial_state.acceleration
            pos_0: np.ndarray = dyno.initial_state.position
            orientation_0: float = dyno.initial_state.orientation
            yaw_rate_0: float = dyno.initial_state.yaw_rate

            merged_lanelets, merged_lanelets_id = get_merged_laneletes_from_position(sc.lanelet_network, pos_0)
            curvilinear_cosy: CurvilinearCoordinateSystem = create_cosy_from_lanelet(merged_lanelets[0])
            curvilinear_pos: Tuple[float, float] = curvilinear_cosy.convert_to_curvilinear_coords(pos_0[0], pos_0[1])
            pos_lon, pos_lat = curvilinear_pos

            # Initialization
            ccosy_orientation = get_orientation_at_position(curvilinear_cosy, pos_0)
            orientation_diff = orientation_0 - ccosy_orientation  # yaw
            v_lon = v_0 * np.cos(orientation_diff)
            v_lat = v_0 * np.sin(orientation_diff)

            for t in range(initial_time_step, initial_time_step + self._config.num_steps_prediction):
                # Prediction
                pos_lon = pos_lon + np.cos(orientation_diff) * (v_lon + 0.5 * a_0 * dt) * dt
                pos_lat = pos_lat + v_lat * dt

                orientation_diff = orientation_diff + yaw_rate_0 * dt

                v_lon = v_0 * np.cos(orientation_diff) + a_0 * dt
                v_lat = v_0 * np.sin(orientation_diff)

                # Get the cartesian positions, etc.
                pred_pos = curvilinear_cosy.convert_to_cartesian_coords(pos_lon, pos_lat)
                ccosy_orientation_next = get_orientation_at_position(curvilinear_cosy, pred_pos)
                pred_orientation = ccosy_orientation_next + orientation_diff

                pred_state_list[t].set_value("position", pred_pos)
                pred_state_list[t].set_value("orientation", pred_orientation)
                pred_state_list[t].set_value("velocity", np.sqrt(v_lon**2 + v_lat**2))
                pred_state_list[t].set_value("acceleration", a_0)

            pred_trajectory = Trajectory(pred_state_list[0].time_step, pred_state_list)
            pred_sc.dynamic_obstacles[idx].prediction.trajectory = pred_trajectory

        return pred_sc

