import copy

import numpy as np
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import State
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.route_matcher import create_cosy_from_lanelet, get_orientation_at_position
from commonroad_dc.geometry.util import compute_orientation_from_polyline, compute_pathlength_from_polyline
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from crpred.predictor_interface import PredictorInterface
from crpred.utility.config import PredictorParams
from crpred.utility.common import get_merged_laneletes_from_position


class ConstantVelocityLinearPredictor(PredictorInterface):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super().__init__(config=config)

    def predict(self, sc: Scenario, initial_time_step: int = 0) -> Scenario:
        pred_sc = copy.deepcopy(sc)

        for idx, dyno in enumerate(sc.dynamic_obstacles):
            trajectory: Trajectory = dyno.prediction.trajectory
            # state_list: list[State] = trajectory.state_list
            pred_state_list: list[State] = copy.deepcopy(trajectory.state_list[: self._config.num_steps_prediction])

            dt = self._config.dt
            if sc.dt != dt:
                print(f"Warning: dt from config ({dt}) is not the same as dt from the scenario ({sc.dt})")

            v_0: float = dyno.initial_state.velocity
            pos_0: np.ndarray = dyno.initial_state.position
            orientation_0: float = dyno.initial_state.orientation

            merged_lanelets, merged_lanelets_id = get_merged_laneletes_from_position(sc.lanelet_network, pos_0)

            # reference_trajectory: np.ndarray = merged_lanelets[0].center_vertices  # (n, 2)
            curvilinear_cosy: CurvilinearCoordinateSystem = create_cosy_from_lanelet(merged_lanelets[0])
            curvilinear_pos: tuple[float, float] = curvilinear_cosy.convert_to_curvilinear_coords(pos_0[0], pos_0[1])
            pos_lon, pos_lat = curvilinear_pos

            ccosy_orientation = get_orientation_at_position(curvilinear_cosy, pos_0)

            # reference_orientations: np.ndarray = compute_orientation_from_polyline(reference_trajectory)  # (n, )
            # reference_path_lengths: np.ndarray = compute_pathlength_from_polyline(reference_trajectory)  # (n, )
            # reference_orientation_interpolated: float = np.interp(pos_lon, reference_path_lengths, reference_orientations)

            orientation_diff = orientation_0 - ccosy_orientation

            delta_v_lon_0: float = v_0 * np.cos(orientation_diff) * dt
            delta_v_lat_0: float = v_0 * np.sin(orientation_diff) * dt

            for t in range(initial_time_step, initial_time_step + self._config.num_steps_prediction):
                if t >= (len(pred_state_list)):
                    break
                # Prediction
                pos_lon = pos_lon + delta_v_lon_0
                pos_lat = pos_lat + delta_v_lat_0

                # Get the cartesian positions, etc.
                pred_pos = curvilinear_cosy.convert_to_cartesian_coords(pos_lon, pos_lat)
                ccosy_orientation_next = get_orientation_at_position(curvilinear_cosy, pred_pos)
                pred_orientation = ccosy_orientation_next + orientation_diff

                pred_state_list[t].position = pred_pos
                pred_state_list[t].orientation = pred_orientation
                pred_state_list[t].velocity = v_0
                pred_state_list[t].acceleration = 0

            pred_trajectory = Trajectory(pred_state_list[0].time_step, pred_state_list)
            pred_sc.dynamic_obstacles[idx].prediction.trajectory = pred_trajectory

        return pred_sc


class ConstantVelocityLinearPredictorV2(PredictorInterface):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super().__init__(config=config)

    def predict(self, sc: Scenario, initial_time_step: int = 0) -> Scenario:
        pred_sc = copy.deepcopy(sc)

        for idx, dyno in enumerate(sc.dynamic_obstacles):
            trajectory: Trajectory = dyno.prediction.trajectory
            pred_state_list: list[State] = copy.deepcopy(trajectory.state_list[: self._config.num_steps_prediction])

            dt = self._config.dt
            if sc.dt != dt:
                print(f"Warning: dt from config ({dt}) is not the same as dt from the scenario ({sc.dt})")

            v_0: float = dyno.initial_state.velocity
            pos_0: np.ndarray = dyno.initial_state.position
            orientation_0: float = dyno.initial_state.orientation

            delta_v = np.array([v_0 * np.cos(orientation_0) * dt, v_0 * np.sin(orientation_0) * dt])
            pred_pos = pos_0

            for t in range(initial_time_step, initial_time_step + self._config.num_steps_prediction):
                if t >= (len(pred_state_list)):
                    break

                pred_pos = pred_pos + delta_v
                pred_orientation = orientation_0

                pred_state_list[t].position = pred_pos
                pred_state_list[t].orientation = pred_orientation
                pred_state_list[t].velocity = v_0

            pred_trajectory = Trajectory(pred_state_list[0].time_step, pred_state_list)
            pred_sc.dynamic_obstacles[idx].prediction.trajectory = pred_trajectory

        return pred_sc


class ConstantVelocityCurvilinearPredictor(PredictorInterface):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super().__init__(config=config)

    def predict(self, sc: Scenario, initial_time_step: int = 0) -> Scenario:
        pred_sc = copy.deepcopy(sc)

        for idx, dyno in enumerate(sc.dynamic_obstacles):
            trajectory: Trajectory = dyno.prediction.trajectory
            # state_list: list[State] = trajectory.state_list
            pred_state_list: list[State] = copy.deepcopy(trajectory.state_list[: self._config.num_steps_prediction])

            dt = self._config.dt
            if sc.dt != dt:
                print(f"Warning: dt from config ({dt}) is not the same as dt from the scenario ({sc.dt})")

            v_0: float = dyno.initial_state.velocity
            pos_0: np.ndarray = dyno.initial_state.position
            orientation_0: float = dyno.initial_state.orientation
            yaw_rate_0: float = dyno.initial_state.yaw_rate

            merged_lanelets, merged_lanelets_id = get_merged_laneletes_from_position(sc.lanelet_network, pos_0)

            # reference_trajectory: np.ndarray = merged_lanelets[0].center_vertices  # (n, 2)
            curvilinear_cosy: CurvilinearCoordinateSystem = create_cosy_from_lanelet(merged_lanelets[0])
            curvilinear_pos: tuple[float, float] = curvilinear_cosy.convert_to_curvilinear_coords(pos_0[0], pos_0[1])
            pos_lon, pos_lat = curvilinear_pos

            # Initialize
            ccosy_orientation = get_orientation_at_position(curvilinear_cosy, pos_0)
            orientation_diff = orientation_0 - ccosy_orientation  # yaw

            v_lon = v_0 * np.cos(orientation_diff)
            v_lat = v_0 * np.sin(orientation_diff)

            for t in range(initial_time_step, initial_time_step + self._config.num_steps_prediction):
                if t >= (len(pred_state_list)):
                    break
                # Prediction
                pos_lon = pos_lon + v_lon * dt
                pos_lat = pos_lat + v_lat * dt

                orientation_diff = orientation_diff + yaw_rate_0 * dt

                v_lon = v_0 * np.cos(orientation_diff)
                v_lat = v_0 * np.sin(orientation_diff)

                # Get the cartesian positions, etc.
                pred_pos = curvilinear_cosy.convert_to_cartesian_coords(pos_lon, pos_lat)
                ccosy_orientation_next = get_orientation_at_position(curvilinear_cosy, pred_pos)
                pred_orientation = ccosy_orientation_next + orientation_diff

                pred_state_list[t].position = pred_pos
                pred_state_list[t].orientation = pred_orientation
                pred_state_list[t].velocity = v_0
                pred_state_list[t].acceleration = 0

            pred_trajectory = Trajectory(pred_state_list[0].time_step, pred_state_list)
            pred_sc.dynamic_obstacles[idx].prediction.trajectory = pred_trajectory

        return pred_sc
