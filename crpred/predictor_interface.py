from abc import ABC
from commonroad.scenario.scenario import Scenario

from crpred.utility.config import PredictorParams
from crpred.utility.visualization import plot_scenario


class PredictorInterface(ABC):
    """Base class for prediction."""

    def __init__(self, config: PredictorParams = PredictorParams()):
        """
        Initialization of predictor interface.
        :param config: Prediction configuration parameters.
        """
        self._config = config
        self.predictor = None

    def predict(self, sc: Scenario, initial_time_step: int = 0) -> Scenario:
        """
        Abstract method for performing predictions.

        :param sc: Scenario containing no predictions for obstacles.
        :param initial_time_step: Time step to start prediction.
        :return: CommonRoad scenario containing predictions.
        """

    # def visualize(self, sc: Scenario, save_plots: bool = True):
    #     """Visualize the prediction."""
    #     plot_scenario(sc, step_end=self._config.num_steps_prediction, predictor_type=self.predictor,
    #                   plot_occupancies=True, save_plots=save_plots)
