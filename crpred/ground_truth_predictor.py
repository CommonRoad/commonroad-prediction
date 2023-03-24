from commonroad.scenario.scenario import Scenario
from crpred.predictor_interface import PredictorInterface
from crpred.utility.config import PredictorParams


class GroundTruthPredictor(PredictorInterface):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super().__init__(config=config)

    def predict(self, sc: Scenario) -> Scenario:
        for obstacle in sc.dynamic_obstacles:
            obstacle.prediction.trajectory.state_list = \
                [state for state in obstacle.prediction.trajectory.state_list[:self._config.num_steps_prediction]
                 if state.time_step <= self._config.num_steps_prediction]

        return sc
