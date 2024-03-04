import copy
from pathlib import Path
from statistics import mean
from typing import Dict, List, Type

from commonroad.scenario.scenario import Scenario
from utilities import calc_scenario_difference, cost_function, get_scenarios_from_files

from crpred.basic_models.constant_acceleration_predictor import (
    ConstantAccelerationCurvilinearPredictor,
    ConstantAccelerationLinearPredictor,
)
from crpred.basic_models.constant_velocity_predictor import (
    ConstantVelocityCurvilinearPredictor,
    ConstantVelocityLinearPredictor,
)
from crpred.ground_truth_predictor import GroundTruthPredictor
from crpred.predictor_interface import PredictorInterface
from crpred.utility.config import PredictorParams
from crpred.utility.visualization import plot_scenario


def main():
    num_steps_prediction = 50
    print_details = True
    visualize = True

    predictors: List[Type[PredictorInterface]] = [
        ConstantVelocityLinearPredictor,
        ConstantVelocityCurvilinearPredictor,
        ConstantAccelerationLinearPredictor,
        ConstantAccelerationCurvilinearPredictor,
        # MOBILPredictor,
    ]

    scenario_dir = Path("scenarios")
    if not scenario_dir.exists():
        scenario_dir = Path("tutorials") / "scenarios"
    scenarios = get_scenarios_from_files(1, scenario_dir)

    # set path to configurations and get configuration object
    configs = [PredictorParams(num_steps_prediction=num_steps_prediction)]

    for sc in scenarios:
        print(f"\n### Scenario: {sc.scenario_id}")
        new_sc = copy.deepcopy(sc)
        plotted_ground_truth = False

        for predictor_cls in predictors:
            all_config_distances = []

            for config in configs:
                print(f"### Predicting {predictor_cls.__name__} with {num_steps_prediction} future states.")
                ground_truth_predictor = GroundTruthPredictor(config)
                ground_truth = ground_truth_predictor.predict(sc)

                predictor: PredictorInterface = predictor_cls(config)
                prediction: Scenario = predictor.predict(new_sc)

                obstacle_distances: Dict[int, Dict[int, float]] = calc_scenario_difference(ground_truth, prediction)

                cost = cost_function(obstacle_distances)
                all_config_distances.append(cost)

                if visualize:
                    output_dir = Path(__file__).parent / f"output/{str(sc.scenario_id)}"
                    plot_scenario(
                        prediction,
                        step_end=config.num_steps_prediction,
                        plot_occupancies=True,
                        save_plots=True,
                        save_gif=True,
                        path_output=output_dir / predictor_cls.__name__,
                    )

                    if not plotted_ground_truth:
                        plot_scenario(
                            ground_truth,
                            step_end=config.num_steps_prediction,
                            plot_occupancies=True,
                            save_plots=True,
                            save_gif=True,
                            path_output=output_dir / "ground_truth",
                        )
                        plotted_ground_truth = True

                if print_details:
                    print(f"{new_sc.scenario_id} Costs: {cost}")

            if len(all_config_distances) > 1:
                print(f"Overall mean distance for {predictor_cls.__name__}: {mean(all_config_distances)}")


if __name__ == "__main__":
    main()
