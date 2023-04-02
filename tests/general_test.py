import copy
import os
from copy import deepcopy
from typing import Tuple, Dict, List, Type, Optional
from math import sqrt
from numpy import ndarray
from statistics import median, mean
from random import sample

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import State
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.trajectory import Trajectory

from crpred.predictor_interface import PredictorInterface
from crpred.advanced_models.idm_predictor import IDMPredictor
# from crpred.advanced_models.mobil_predictor import MOBILPredictor
from crpred.ground_truth_predictor import GroundTruthPredictor
from crpred.utility.config import PredictorParams
from crpred.utility.common import clear_obstacle_trajectory


def shorten_scenario_trajectory(original_scenario: Scenario, remaining_steps: int) -> Optional[Scenario]:
    """
    Shorten the given scenario. Remove all steps after n remaining_steps
    """
    s: Scenario = deepcopy(original_scenario)

    for obstacle in s.dynamic_obstacles:
        short_state_list: List[State] = obstacle.prediction.trajectory.state_list[:remaining_steps]
        obstacle.prediction.trajectory.state_list = short_state_list

    return s


def get_scenarios_from_files(n: int):
    """
    Reading in n random scenarios from the trajectory_prediction_tests/scenarios folder
    """
    result: List[Scenario] = []
    print(f"Reading in {str(n)} scenarios")

    for dir_path, _, filenames in os.walk(os.path.join(os.getcwd(), 'scenarios/')):
        for f in sample(filenames, n):
            path = os.path.abspath(os.path.join(dir_path, f))
            print(f"Reading in {f} ", end="")
            s, _ = CommonRoadFileReader(path).open(lanelet_assignment=True)
            result.append(s)
            print(u'\u2713')  # checkmark

    return result


def calc_difference(s: Scenario, p: Scenario) -> Dict[int, Dict[int, float]]:
    """
    Calculate the distance between the predicted and the Original Trajectory
    @returns The distance between every common time step for all dynamic obstacles
    """
    result = {}

    for dyno in s.dynamic_obstacles:
        s_id: int = dyno.obstacle_id
        original_trajectory: Trajectory = dyno.prediction.trajectory
        predicted_trajectory: Trajectory = p.obstacle_by_id(s_id).prediction.trajectory

        first_common_time_step: int = max(original_trajectory.initial_time_step,
                                          predicted_trajectory.initial_time_step)
        last_common_time_step: int = min(original_trajectory.state_list[-1].time_step,
                                         predicted_trajectory.state_list[-1].time_step)

        common_steps: List[Tuple[State, State]] = list(zip(
            [s for s in original_trajectory.state_list
             if first_common_time_step <= s.time_step <= last_common_time_step],
            [s for s in predicted_trajectory.state_list
             if first_common_time_step <= s.time_step <= last_common_time_step]))

        distances: Dict[int, float] = {}

        for (original_step, predicted_step) in common_steps:
            time_step = original_step.time_step
            original_position: ndarray = original_step.position
            original_position_x: int = original_position[0]
            original_position_y: int = original_position[1]

            predicted_position: ndarray = predicted_step.position
            predicted_position_x: int = predicted_position[0]
            predicted_position_y: int = predicted_position[1]

            distance = sqrt(((original_position_x - predicted_position_x) ** 2)
                            + ((original_position_y - predicted_position_y) ** 2))

            distances[time_step] = distance

        result[s_id] = distances

    return result


def cost_function(obstacle_distances: Dict[int, Dict[int, float]]) -> float:
    """
    Given the distance from the original position to the predicted for every dynamic obstacle.
    Calculate how good the prediction is as a single number to make comparison easier
    """
    return mean([mean_distance(d) for (o, d) in obstacle_distances.items()])


def mean_distance(distance: Dict[int, float]) -> float:
    """
    Calculate the mean between all objects in a directory
    """
    return mean([d for (s, d) in distance.items()])


def median_distance(distance: Dict[int, float]) -> float:
    """
    Calculate the median between all objects in a directory
    """
    return median([d for (s, d) in distance.items()])


def trajectory_prediction_test(predictor_clss: List[Type[PredictorInterface]], future_states_range: List[int],
                               details: bool = False, visualize: bool = False):
    """
    :param predictor_clss: A list of predictor classes that should be tested
    :param future_states_range: A list how many future states should be predicted.
    [5, 10, 15] test all given predictors with 5, 10 and 15 future steps
    :param details: Should details be printed
    :param visualize: Should the solutions be visualized
    """
    scenarios = get_scenarios_from_files(1)

    # set path to configurations and get configuration object
    configs = [PredictorParams()]

    for sc in scenarios:
        new_sc = copy.deepcopy(sc)
        for future_states in future_states_range:
            print(f"Future States: {future_states}")
            for predictor_cls in predictor_clss:
                all_distances = []

                for config in configs:
                    config.num_steps_prediction = future_states

                    ground_truth_predictor = GroundTruthPredictor(config)
                    ground_truth = ground_truth_predictor.predict(sc)

                    predictor: PredictorInterface = predictor_cls(config)
                    prediction: Scenario = predictor.predict(new_sc)

                    obstacle_distances: Dict[int, Dict[int, float]] = calc_difference(ground_truth, prediction)

                    cost = cost_function(obstacle_distances)
                    all_distances.append(cost)

                    if visualize:
                        predictor.visualize(new_sc)
                        ground_truth_predictor.visualize(sc)

                    if details:
                        print(f"{new_sc.scenario_id}: {cost}")

                print(f"Over all mean distance for {predictor_cls.__name__}: {mean(all_distances)}")


if __name__ == "__main__":
    predictors: List[Type[PredictorInterface]] = [IDMPredictor] #, MOBILPredictor]
    trajectory_prediction_test(predictors, list(range(50, 51)), True, True)
