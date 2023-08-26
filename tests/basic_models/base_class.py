from pathlib import Path

import pytest
from commonroad.common.file_reader import CommonRoadFileReader

from crpred.basic_models.motion_model_predictor import MotionModelPredictor
from crpred.utility.config import PredictorParams
from tests.utilities import create_default_scenario


class MotionModelPredictorTest:
    def setup_class(self):
        self.test_config = PredictorParams(num_steps_prediction=50)
        self.predictor = MotionModelPredictor(config=self.test_config)

    def test_scenario_creation(self):
        scenario = create_default_scenario()
        pred_scenario = self.predictor.predict(scenario)
        assert pred_scenario.dt == scenario.dt
        assert len(pred_scenario.dynamic_obstacles[
                       0].prediction.trajectory.state_list) == self.test_config.num_steps_prediction

    def test_system(self):
        test_scenarios_dir = Path('system_test_scenarios')
        scenario_paths = test_scenarios_dir.glob('*.xml')

        for path in scenario_paths:
            scenario, _ = CommonRoadFileReader(path).open(lanelet_assignment=True)
            pred_sc = self.predictor.predict(scenario)
            assert pred_sc.dynamic_obstacles[0].__getattribute__("prediction")

    def test_prediction_const_velocity(self):
        raise NotImplementedError

    def test_prediction_const_acceleration(self):
        raise NotImplementedError

    def test_prediction_const_yaw_rate(self):
        raise NotImplementedError


# Run the tests
if __name__ == "__main__":
    pytest.main()
