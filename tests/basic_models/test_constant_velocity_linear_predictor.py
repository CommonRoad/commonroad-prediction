import numpy as np
import pytest
from commonroad.scenario.state import InitialState

import tests.utilities as test_utils
from crpred.basic_models.constant_velocity_predictor import ConstantVelocityLinearPredictor
from crpred.utility.config import PredictorParams
from tests.basic_models.base_class import MotionModelPredictorTest


class TestConstantVelocityLinearPredictor(MotionModelPredictorTest):
    def setup_class(self):
        self.test_config = PredictorParams(num_steps_prediction=50)
        self.predictor = ConstantVelocityLinearPredictor(config=self.test_config)

    def test_prediction_const_velocity(self):
        velocity = 10.0
        scenario = test_utils.create_const_velocity_straight_scenario(velocity)
        pred_sc = self.predictor.predict(scenario)
        dyno = pred_sc.dynamic_obstacles[0]

        for state in dyno.prediction.trajectory.state_list:
            assert state.velocity == velocity
            assert state.orientation == 0.0

        final_state = dyno.initial_state.__getattribute__(
            "position") + self.test_config.num_steps_prediction * scenario.dt * np.array([velocity, 0])
        np.testing.assert_array_equal(dyno.prediction.trajectory.state_list[-1].position, final_state)

    def test_prediction_const_acceleration(self):
        acceleration = 1.0
        scenario = test_utils.create_const_acceleration_straight_scenario(acceleration)
        pred_sc = self.predictor.predict(scenario)
        dyno = pred_sc.dynamic_obstacles[0]

        for state in dyno.prediction.trajectory.state_list:
            assert state.velocity == 0.0
            assert state.orientation == 0.0

        final_state = dyno.initial_state.__getattribute__("position")
        np.testing.assert_array_equal(dyno.prediction.trajectory.state_list[-1].position, final_state)

    def test_prediction_const_yaw_rate(self):
        yaw_rate = 0.1
        velocity = 10.0
        scenario = test_utils.create_const_yaw_rate_straight_scenario(yaw_rate, velocity)
        pred_sc = self.predictor.predict(scenario)
        dyno = pred_sc.dynamic_obstacles[0]

        for state in dyno.prediction.trajectory.state_list:
            assert state.velocity == 10.0
            assert state.orientation == 0.0

        final_state = dyno.initial_state.__getattribute__(
            "position") + self.test_config.num_steps_prediction * scenario.dt * np.array([velocity, 0])
        np.testing.assert_array_equal(dyno.prediction.trajectory.state_list[-1].position, final_state)


# Run the tests
if __name__ == "__main__":
    pytest.main()
