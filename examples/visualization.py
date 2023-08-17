from pathlib import Path
from typing import Type

from matplotlib import pyplot as plt
from crpred.basic_models.constant_acceleration_predictor import ConstantAccelerationLinearPredictor
from crpred.basic_models.constant_velocity_predictor import ConstantVelocityLinearPredictor, ConstantVelocityLinearPredictorV2
from crpred.predictor_interface import PredictorInterface
from crpred.utility.config import PredictorParams
from crpred.utility.visualization import visualize_prediction
from examples.utilities import get_scenarios_from_files


def main():
    predictors: list[Type[PredictorInterface]] = [
        ConstantVelocityLinearPredictor,
        ConstantVelocityLinearPredictorV2,
        # ConstantAccelerationLinearPredictor,
    ]
    config = PredictorParams()

    scenario = get_scenarios_from_files(1, Path("scenarios"))[0]
    fig = visualize_prediction(scenario, start_step=0, end_step=30)
    fig.suptitle("Original")

    for predictor_class in predictors:
        predictor = predictor_class(config)

        predicted_scenario = predictor.predict(scenario)
        fig = visualize_prediction(predicted_scenario, start_step=0, end_step=30)
        fig.suptitle(f"{predictor_class.__name__}")
    
    plt.show()


if __name__ == "__main__":
    main()