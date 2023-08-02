import copy
from pathlib import Path
from statistics import mean
from typing import Dict, List, Type

from commonroad.scenario.scenario import Scenario
from test_utilities import (calc_difference, cost_function,
                            get_scenarios_from_files,
                            shorten_scenario_trajectory)

from crpred.advanced_models.idm_predictor import IDMPredictor
from crpred.basic_models.constant_velocity_predictor import \
    ConstantVelocityLinearPredictor
# from crpred.advanced_models.mobil_predictor import MOBILPredictor
from crpred.ground_truth_predictor import GroundTruthPredictor
from crpred.predictor_interface import PredictorInterface
from crpred.utility.config import PredictorParams
from crpred.utility.visualization import plot_scenario


def trajectory_prediction_test(
    predictor_clss: List[Type[PredictorInterface]],
    future_states_range: List[int],
    details: bool = False,
    visualize: bool = False,
):
    """
    :param predictor_clss: A list of predictor classes that should be tested
    :param future_states_range: A list how many future states should be predicted.
    [5, 10, 15] test all given predictors with 5, 10 and 15 future steps
    :param details: Should details be printed
    :param visualize: Should the solutions be visualized
    """
    scenarios = get_scenarios_from_files(1, Path("tests/scenarios"))

    # set path to configurations and get configuration object
    configs = [PredictorParams()]

    for sc in scenarios:
        new_sc = copy.deepcopy(sc)
        for future_states in future_states_range:
            for predictor_cls in predictor_clss:
                all_distances = []

                for config in configs:
                    print(
                        f"\n### Testing {predictor_cls.__name__} with {future_states} future"
                        " states."
                    )
                    config.num_steps_prediction = future_states

                    ground_truth_predictor = GroundTruthPredictor(config)
                    ground_truth = ground_truth_predictor.predict(sc)

                    predictor: PredictorInterface = predictor_cls(config)
                    prediction: Scenario = predictor.predict(new_sc)

                    original_position = (
                        ground_truth.dynamic_obstacles[0]
                        .prediction.trajectory.state_list[0]
                        .position
                    )
                    predicted_position = (
                        prediction.dynamic_obstacles[0].prediction.trajectory.state_list[0].position
                    )
                    print(original_position)
                    print(predicted_position)

                    obstacle_distances: Dict[int, Dict[int, float]] = calc_difference(
                        ground_truth, prediction
                    )

                    cost = cost_function(obstacle_distances)
                    all_distances.append(cost)

                    if visualize:
                        output_dir = Path(f"output/{str(sc.scenario_id)}/{predictor_cls.__name__}")
                        # Plot the prediction
                        plot_scenario(
                            prediction,
                            step_end=config.num_steps_prediction,
                            predictor_type=predictor,
                            plot_occupancies=True,
                            save_plots=True,
                            save_gif=True,
                            path_output=output_dir.joinpath("prediction"),
                        )
                        # Plot the ground truth
                        plot_scenario(
                            ground_truth,
                            step_end=config.num_steps_prediction,
                            predictor_type=ground_truth_predictor,
                            plot_occupancies=True,
                            save_plots=True,
                            save_gif=True,
                            path_output=output_dir.joinpath("ground_truth"),
                        )
                        # predictor.visualize(new_sc)
                        # ground_truth_predictor.visualize(sc)

                    if details:
                        print(f"{new_sc.scenario_id}: {cost}")

                print(f"Over all mean distance for {predictor_cls.__name__}: {mean(all_distances)}")


if __name__ == "__main__":
    predictors: List[Type[PredictorInterface]] = [
        ConstantVelocityLinearPredictor,
        # MOBILPredictor,
    ]  
    trajectory_prediction_test(predictors, list(range(50, 51)), True, True)
