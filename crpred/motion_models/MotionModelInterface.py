from typing import Dict

from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.scenario import Scenario
from motion_planner_config.configuration_builder import Configuration

from motion_planner_components.prediction.uncertain_states.UncertainStatesPredictor import PredictorInterface
from commonroad.scenario.trajectory import *

from motion_planner_components.prediction.utility.prediction import generate_initial_previous_states
from motion_planner_components.prediction.utility import visualization as util_visualization


class MotionModelInterface(PredictorInterface):
    """"
    Describe the implemented Motion Model
    """

    def __init__(self, config: Configuration, previous_states: Optional[Dict[int, List[State]]] = None):
        super(MotionModelInterface, self).__init__(config=config)
        if previous_states:
            self.prev_states = previous_states
        else:
            self.prev_states = generate_initial_previous_states(self.configuration)
        self.trajectory_predicted: Dict[int, TrajectoryPrediction] = {}

    def state_prediction_generator(self, state_list: List[State]) -> List[State]:
        """
        Function should predict n future states given states of the past for a dynamic obstacle
        """
        pass

    def predict(self) -> Scenario:
        """
        Calls the state_prediction_generator() function for every dynamic obstacle giving it the state list of
        the past. Construct the TrajectoryPrediction Objects from the returned state list.
        Return a scenario with predictions for each dynamic obstacle
        """
        result: Dict[int, TrajectoryPrediction] = {}

        for do in self.scenario.dynamic_obstacles:
            # Prediction Trajectory
            prediction_state_list: List[State] = self.state_prediction_generator(self.prev_states[do.obstacle_id])
            initial_time_step: int = prediction_state_list[0].time_step
            prediction_trajectory: Trajectory = Trajectory(initial_time_step, prediction_state_list)

            # Make TrajectoryPrediction
            shape: Shape = do.obstacle_shape
            prediction: TrajectoryPrediction = TrajectoryPrediction(prediction_trajectory, shape)
            result[do.obstacle_id] = prediction
            do.prediction = prediction

        self.trajectory_predicted = result
        return self.scenario

    def visualize(self):
        """
        Visualize the prediction
        """
        util_visualization.plot_scenario(self.scenario, step_end=self.num_steps_prediction,
                                         predictor_type="motion_model")
