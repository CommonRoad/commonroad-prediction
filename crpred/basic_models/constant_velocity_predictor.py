import copy

import numpy as np
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import State
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.route_matcher import (create_cosy_from_lanelet,
                                               get_orientation_at_position)
from commonroad_dc.geometry.util import (compute_orientation_from_polyline,
                                         compute_pathlength_from_polyline)
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from crpred.predictor_interface import PredictorInterface
from crpred.utility.config import PredictorParams


class ConstantVelocityLinearPredictor(PredictorInterface):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super().__init__(config=config)

    def predict(self, sc: Scenario, initial_time_step: int = 0) -> Scenario:
        print()
        pred_sc = copy.deepcopy(sc)

        for idx, dyno in enumerate(sc.dynamic_obstacles):
            trajectory: Trajectory = dyno.prediction.trajectory
            state_list: list[State] = trajectory.state_list
            pred_state_list: list[State] = copy.deepcopy(
                trajectory.state_list[: self._config.num_steps_prediction]
            )

            dt = self._config.dt
            if sc.dt != dt:
                print(
                    f"Warning: dt from config ({dt}) is not the same as dt from the scenario"
                    f" ({sc.dt})"
                )

            v_0: float = dyno.initial_state.velocity
            pos_0: np.ndarray = dyno.initial_state.position
            orientation_0: float = dyno.initial_state.orientation

            # Find lanelet of the dynamic obstacle
            lanelet_id_list = sc.lanelet_network.find_lanelet_by_position([pos_0])
            if not lanelet_id_list:
                print(
                    f"Warning: Dynamic obstacle (id: {dyno.obstacle_id}) cannot be assigned to a"
                    " lanelet."
                )
                return

            current_lanelet = sc.lanelet_network.find_lanelet_by_id(lanelet_id_list[0][0])
            (
                merged_lanelets,
                merged_lanelets_id,
            ) = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                current_lanelet, sc.lanelet_network, 300
            )

            reference_trajectory: np.ndarray = merged_lanelets[0].center_vertices  # (n, 2)
            curvilinear_cosy: CurvilinearCoordinateSystem = create_cosy_from_lanelet(
                merged_lanelets[0]
            )

            curvilinear_pos: tuple[float, float] = curvilinear_cosy.convert_to_curvilinear_coords(
                pos_0[0], pos_0[1]
            )
            pos_lon, pos_lat = curvilinear_pos

            reference_orientation: np.ndarray = compute_orientation_from_polyline(
                reference_trajectory
            )  # (n, )
            reference_path_length: np.ndarray = compute_pathlength_from_polyline(
                reference_trajectory
            )  # (n, )

            orientation_interpolated: float = np.interp(
                pos_lon, reference_path_length, reference_orientation
            )
            orientation_diff = orientation_0 - orientation_interpolated

            delta_v_lon_0: float = v_0 * np.cos(orientation_diff) * dt
            delta_v_lat_0: float = v_0 * np.sin(orientation_diff) * dt

            for t in range(
                initial_time_step, initial_time_step + self._config.num_steps_prediction
            ):
                if t >= (len(pred_state_list)):
                    break
                # pos_next = pos_curr + v_0 * dt
                pos_lon = pos_lon + delta_v_lon_0
                pos_lat = pos_lat + delta_v_lat_0

                pred_pos = curvilinear_cosy.convert_to_cartesian_coords(pos_lon, pos_lat)
                pred_orientation = get_orientation_at_position(curvilinear_cosy, pred_pos)

                pred_state_list[t].position = pred_pos
                pred_state_list[t].orientation = pred_orientation
                pred_state_list[t].velocity = v_0

            pred_trajectory = Trajectory(pred_state_list[0].time_step, pred_state_list)
            pred_sc.dynamic_obstacles[idx].prediction.trajectory = pred_trajectory

        return pred_sc
