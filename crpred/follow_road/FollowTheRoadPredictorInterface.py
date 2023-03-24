from typing import Dict

from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import *
from sympy.geometry import Point2D

from motion_planner_components.prediction.PredictorInterface import PredictorInterface
from motion_planner_config.configuration_builder import Configuration
from motion_planner_components.prediction.utility.prediction import generate_initial_previous_states
from motion_planner_components.prediction.utility import visualization as util_visualization


class FollowTheRoadPredictorInterface(PredictorInterface):

    def __init__(self, config: Configuration, previous_states: Optional[Dict[int, List[State]]] = None):
        super(FollowTheRoadPredictorInterface, self).__init__(config=config)
        if previous_states:
            self.prev_states = previous_states
        else:
            self.prev_states = generate_initial_previous_states(self.configuration)
        self.trajectory_predicted: Dict[int, TrajectoryPrediction] = {}

    def follow_lane(self, lane_points: List[Point2D], state: State) -> State:
        """
        Return the next State that the vehicle would drive if we assume that the vehicle follows the road
        """
        pass

    def state_prediction_generator(self, state_list: List[State], dynamic_obstacle: DynamicObstacle) -> List[State]:
        """
        Predict the list State list for a dynamic obstacle
        Calls the follow_lane function, and formats the result
        """
        prediction_state_list: List[State] = []

        last_state: State = state_list[-1]
        lane_id = self.scenario.lanelet_network.find_lanelet_by_position([last_state.position])[0][0]
        lane: Lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lane_id)

        # Convert the center lane into a path - List of Points
        lane_points: List[Point2D] = [Point2D(p[0], p[1]) for p in lane.center_vertices.tolist()]

        for i in range(0, self.num_steps_prediction):
            last_state = self.follow_lane(lane_points, last_state)
            prediction_state_list.append(last_state)

        return prediction_state_list

    def predict_dynamic_obstacle(self, dynamic_obstacle: DynamicObstacle):
        """
        Predict the path of a given dynamic obstacle and add it to object
        """
        # Prediction Trajectory
        prediction_state_list: List[State] = self.state_prediction_generator(self.prev_states[dynamic_obstacle.obstacle_id], dynamic_obstacle)
        initial_time_step: int = prediction_state_list[0].time_step
        prediction_trajectory: Trajectory = Trajectory(initial_time_step, prediction_state_list)

        # Make TrajectoryPrediction
        shape: Shape = dynamic_obstacle.obstacle_shape
        prediction: TrajectoryPrediction = TrajectoryPrediction(prediction_trajectory, shape)
        self.trajectory_predicted[dynamic_obstacle.obstacle_id] = prediction
        dynamic_obstacle.prediction = prediction

    def predict(self) -> Scenario:
        """
        Predict the path of the given dynamic Obstacles
        """
        for dynamic_obstacle in self.scenario.dynamic_obstacles:
            self.predict_dynamic_obstacle(dynamic_obstacle)

        return self.scenario

    def visualize(self):
        """
        Visualize the prediction
        """
        util_visualization.plot_scenario(self.scenario, step_end=self.num_steps_prediction,
                                         predictor_type="follow_road")
