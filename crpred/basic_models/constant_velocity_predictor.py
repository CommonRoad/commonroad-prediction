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


class ConstantVelocityLinearPredictor(MotionModelPredictor):
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

        delta_v_lon_0: float = initial_values.v * np.cos(initial_values.orientation_in_ccosy) * dt
        delta_v_lat_0: float = initial_values.v * np.sin(initial_values.orientation_in_ccosy) * dt
        p_lon = initial_values.p_lon
        p_lat = initial_values.p_lat

        for t in prediction_range:
            p_lon = p_lon + delta_v_lon_0
            p_lat = p_lat + delta_v_lat_0

            # Get the cartesian positions, etc.
            pred_pos = curvilinear_cosy.convert_to_cartesian_coords(p_lon, p_lat)
            ccosy_orientation_next = get_orientation_at_position(curvilinear_cosy, pred_pos)
            pred_orientation = ccosy_orientation_next + initial_values.orientation_in_ccosy

            pred_state = CustomState(
                time_step=t,
                position=pred_pos,
                orientation=pred_orientation,
                velocity=initial_values.v,
            )
            pred_state_list.append(pred_state)

        return pred_state_list


class ConstantVelocityCurvilinearPredictor(MotionModelPredictor):
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

        p_lon = initial_values.p_lon
        p_lat = initial_values.p_lat
        v_lon = initial_values.v * np.cos(initial_values.orientation_in_ccosy)
        v_lat = initial_values.v * np.sin(initial_values.orientation_in_ccosy)
        orientation_in_ccosy = initial_values.orientation_in_ccosy

        for t in prediction_range:
            p_lon = p_lon + v_lon * dt
            p_lat = p_lat + v_lat * dt

            orientation_in_ccosy = orientation_in_ccosy + initial_values.yaw_rate * dt
            v_lon = initial_values.v * np.cos(orientation_in_ccosy)
            v_lat = initial_values.v * np.sin(orientation_in_ccosy)

            # Get the cartesian positions, etc.
            pred_pos = curvilinear_cosy.convert_to_cartesian_coords(p_lon, p_lat)
            ccosy_orientation_next = get_orientation_at_position(curvilinear_cosy, pred_pos)
            pred_orientation = ccosy_orientation_next + orientation_in_ccosy

            pred_state = CustomState(
                time_step=t,
                position=pred_pos,
                orientation=pred_orientation,
                velocity=initial_values.v,
                yaw_rate=initial_values.yaw_rate,
            )
            pred_state_list.append(pred_state)

        return pred_state_list
