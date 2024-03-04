from typing import List

import numpy as np
from commonroad.scenario.state import CustomState
from commonroad_dc.costs.route_matcher import get_orientation_at_position
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from crpred.basic_models.motion_model_predictor import (
    InitialStateValues,
    MotionModelPredictor,
)
from crpred.utility.config import PredictorParams


class ConstantAccelerationLinearPredictor(MotionModelPredictor):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super().__init__(config=config)

    def _predict_states(
        self,
        initial_values: InitialStateValues,
        dt: float,
        curvilinear_cosy: CurvilinearCoordinateSystem,
        prediction_range: range,
    ) -> List[CustomState]:
        pred_state_list: List[CustomState] = []

        v_lon: float = initial_values.v * np.cos(initial_values.orientation_in_ccosy)
        v_lat: float = initial_values.v * np.sin(initial_values.orientation_in_ccosy)
        p_lon = initial_values.p_lon
        p_lat = initial_values.p_lat

        for t in prediction_range:
            p_lon = (
                p_lon
                + np.cos(initial_values.orientation_in_ccosy) * (v_lon + 0.5 * initial_values.acceleration * dt) * dt
            )
            p_lat = p_lat + v_lat * dt

            v_lon = v_lon + initial_values.acceleration * dt  # acceleration only in longitudinal direction

            # Get the cartesian positions, etc.
            pred_pos = curvilinear_cosy.convert_to_cartesian_coords(p_lon, p_lat)
            ccosy_orientation_next = get_orientation_at_position(curvilinear_cosy, pred_pos)
            pred_orientation = ccosy_orientation_next + initial_values.orientation_in_ccosy

            pred_state = CustomState(
                time_step=t,
                position=pred_pos,
                orientation=pred_orientation,
                velocity=np.sqrt(v_lon**2 + v_lat**2),
                acceleration=initial_values.acceleration,
            )
            pred_state_list.append(pred_state)

        return pred_state_list


class ConstantAccelerationCurvilinearPredictor(MotionModelPredictor):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super().__init__(config=config)

    def _predict_states(
        self,
        initial_values: InitialStateValues,
        dt: float,
        curvilinear_cosy: CurvilinearCoordinateSystem,
        prediction_range: range,
    ) -> List[CustomState]:
        pred_state_list: List[CustomState] = []

        v_lon: float = initial_values.v * np.cos(initial_values.orientation_in_ccosy)
        v_lat: float = initial_values.v * np.sin(initial_values.orientation_in_ccosy)
        p_lon = initial_values.p_lon
        p_lat = initial_values.p_lat
        orientation_in_ccosy = initial_values.orientation_in_ccosy

        for t in prediction_range:
            p_lon = p_lon + np.cos(orientation_in_ccosy) * (v_lon + 0.5 * initial_values.acceleration * dt) * dt
            p_lat = p_lat + v_lat * dt

            orientation_in_ccosy = orientation_in_ccosy + initial_values.yaw_rate * dt
            prev_state_v = np.sqrt(v_lon**2 + v_lat**2)
            v_lon = prev_state_v * np.cos(orientation_in_ccosy) + initial_values.acceleration * dt
            v_lat = prev_state_v * np.sin(orientation_in_ccosy)

            # Get the cartesian positions, etc.
            pred_pos = curvilinear_cosy.convert_to_cartesian_coords(p_lon, p_lat)
            ccosy_orientation_next = get_orientation_at_position(curvilinear_cosy, pred_pos)
            pred_orientation = ccosy_orientation_next + orientation_in_ccosy

            pred_state = CustomState(
                time_step=t,
                position=pred_pos,
                orientation=pred_orientation,
                velocity=np.sqrt(v_lon**2 + v_lat**2),
                acceleration=initial_values.acceleration,
                yaw_rate=initial_values.yaw_rate,
            )
            pred_state_list.append(pred_state)

        return pred_state_list
