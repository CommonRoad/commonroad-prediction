import copy
from dataclasses import dataclass
from typing import Tuple, List

from commonroad.prediction.prediction import Prediction, Occupancy
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.costs.route_matcher import create_cosy_from_lanelet, get_orientation_at_position
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem

from crpred.predictor_interface import PredictorInterface
from crpred.utility.common import get_merged_laneletes_from_position
from crpred.utility.config import PredictorParams


@dataclass
class InitialStateValues:
    p_lon: float
    p_lat: float
    v: float
    orientation_in_ccosy: float
    acceleration: float
    yaw_rate: float


class MotionModelPredictor(PredictorInterface):
    def __init__(self, config: PredictorParams = PredictorParams()):
        """
        Initialize a MotionModelPredictor instance.

        :param config: Configuration parameters for the predictor.
        """
        super().__init__(config=config)

    def predict(self, sc: Scenario, initial_time_step: int = 0) -> Scenario:
        """
        Predict the future states of dynamic obstacles in the given scenario using a motion model.

        :param sc: Scenario containing no predictions for obstacles.
        :param initial_time_step: Time step to start prediction.
        :return: CommonRoad scenario containing predictions.
        """
        pred_sc = copy.deepcopy(sc)
        dt = self._config.dt
        if sc.dt != dt:
            print(f"Warning: dt from config ({dt}) is not the same as dt from the scenario ({sc.dt})")

        for idx, dyno in enumerate(sc.dynamic_obstacles):
            if dyno.prediction:
                steps_in_scenario = len(dyno.prediction.trajectory.state_list)
                steps_to_predict = self._config.num_steps_prediction \
                    if self._config.num_steps_prediction <= steps_in_scenario \
                    else steps_in_scenario
            else:
                steps_to_predict = self._config.num_steps_prediction

            initial_state: InitialState = dyno.initial_state

            # Maybe move these checks into the child classes to make them model-specific
            assert initial_state.velocity is not None
            assert initial_state.acceleration is not None
            assert initial_state.yaw_rate is not None
            assert initial_state.position is not None
            assert initial_state.orientation is not None

            # Calculate the curvilinear coordinates
            merged_lanelets, merged_lanelets_id = get_merged_laneletes_from_position(
                sc.lanelet_network, initial_state.position)
            curvilinear_cosy: CurvilinearCoordinateSystem = create_cosy_from_lanelet(merged_lanelets[0])

            curvilinear_pos: Tuple[float, float] = curvilinear_cosy.convert_to_curvilinear_coords(
                initial_state.position[0], initial_state.position[1])
            pos_lon, pos_lat = curvilinear_pos

            # Calculate the orientation in the curvilinear coordinate system
            ccosy_orientation = get_orientation_at_position(curvilinear_cosy, initial_state.position)
            orientation_diff = initial_state.orientation - ccosy_orientation

            initial_state_values = InitialStateValues(
                p_lon=pos_lon,
                p_lat=pos_lat,
                v=initial_state.velocity,
                orientation_in_ccosy=orientation_diff,
                acceleration=initial_state.acceleration,
                yaw_rate=initial_state.yaw_rate,
            )

            pred_state_list = self._predict_states(
                initial_state_values,
                dt,
                curvilinear_cosy,
                range(initial_time_step, initial_time_step + steps_to_predict),
            )
            assert len(pred_state_list) == steps_to_predict

            pred_trajectory = Trajectory(initial_time_step, pred_state_list)

            # Check if a prediction object exists in the dynamic obstacle
            if not pred_sc.dynamic_obstacles[idx].prediction:
                pred_sc.dynamic_obstacles[idx].prediction = Prediction(
                    initial_time_step=initial_time_step,
                    occupancy_set=[Occupancy(i, dyno.obstacle_shape) for i in
                                   range(initial_time_step, initial_time_step + steps_to_predict)]
                )

            pred_sc.dynamic_obstacles[idx].prediction.trajectory = pred_trajectory

        return pred_sc

    def _predict_states(self, initial_values: InitialStateValues, dt: float,
                        curvilinear_cosy: CurvilinearCoordinateSystem, prediction_range: range) -> List[CustomState]:
        """
        Abstract method for calculating model-specific predictions for future states.

        This method calculates and returns a list of predicted future states for a dynamic obstacle based on the
        provided initial values and time step range.

        :param initial_values: PredictionValues instance containing model parameters.
        :param dt: Time step duration.
        :param curvilinear_cosy: CurvilinearCoordinateSystem instance representing the curvilinear coordinate system.
        :param prediction_range: Range of time steps for which to calculate predictions.
        :return: A list of CustomState instances representing predicted future states.
        """
        raise NotImplementedError()
