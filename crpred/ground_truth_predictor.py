from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from crpred.predictor_interface import PredictorInterface
from crpred.utility.config import PredictorParams


class GroundTruthPredictor(PredictorInterface):
    """Applies prediction stored in scenario."""

    def __init__(self, config: PredictorParams = PredictorParams()):
        """
        Initialization of ground truth prediction.

        :param config: Prediction config parameters.
        """
        super().__init__(config=config)

    def predict(self, sc: Scenario, initial_time_step: int = 0) -> Scenario:
        """
        Applies ground truth prediction.

        :param sc: CommonRoad scenario.
        :param initial_time_step: Time step at which prediction should start.
        :return: CommonRoad scenario containing prediction.
        """
        for obstacle in sc.dynamic_obstacles:
            state_list = [state for state in
                          obstacle.prediction.trajectory.state_list[:self._config.num_steps_prediction]
                          if state.time_step <= self._config.num_steps_prediction]
            traj = Trajectory(state_list[0].time_step, state_list)
            obstacle.prediction.trajectory = traj

        return sc
