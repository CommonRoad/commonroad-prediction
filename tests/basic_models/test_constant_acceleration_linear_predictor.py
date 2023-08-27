import numpy as np
import pytest

import tests.utilities as test_utils
from crpred.basic_models.constant_acceleration_predictor import ConstantAccelerationLinearPredictor
from crpred.utility.config import PredictorParams
from tests.basic_models.base_class import MotionModelPredictorTest


class TestConstantAccelerationLinearPredictor(MotionModelPredictorTest):
    def setup_class(self):
        self.test_config = PredictorParams(num_steps_prediction=50)
        self.predictor = ConstantAccelerationLinearPredictor(config=self.test_config)

    def test_prediction_const_velocity(self):
        velocity = 10.0
        scenario = test_utils.create_const_velocity_straight_scenario(velocity)
        pred_sc = self.predictor.predict(scenario)
        dyno = pred_sc.dynamic_obstacles[0]

        for state in dyno.prediction.trajectory.state_list:
            assert state.velocity == velocity
            assert state.acceleration == 0.0
            assert state.orientation == 0.0

        final_state = dyno.initial_state.__getattribute__(
            "position") + self.test_config.num_steps_prediction * scenario.dt * np.array([velocity, 0])
        np.testing.assert_array_equal(dyno.prediction.trajectory.state_list[-1].position, final_state)

    def test_prediction_const_acceleration(self):
        acceleration = 1.0
        scenario = test_utils.create_const_acceleration_straight_scenario(acceleration)
        pred_sc = self.predictor.predict(scenario)
        dyno = pred_sc.dynamic_obstacles[0]

        for i, state in enumerate(dyno.prediction.trajectory.state_list):
            assert state.velocity == pytest.approx((i + 1) * scenario.dt * acceleration, 0.001)
            assert state.orientation == 0.0
            assert state.acceleration == 1.0

        final_state = dyno.initial_state.__getattribute__(
            "position") + 0.5 * (self.test_config.num_steps_prediction * scenario.dt) ** 2 * np.array([acceleration, 0])
        np.testing.assert_array_almost_equal(dyno.prediction.trajectory.state_list[-1].position, final_state)

    def test_prediction_const_yaw_rate(self):
        yaw_rate = 0.1
        scenario = test_utils.create_const_yaw_rate_straight_scenario(yaw_rate)
        pred_sc = self.predictor.predict(scenario)
        dyno = pred_sc.dynamic_obstacles[0]

        for state in dyno.prediction.trajectory.state_list:
            assert state.velocity == 0.0
            assert state.orientation == 0.0

        final_state = dyno.initial_state.__getattribute__("position")
        np.testing.assert_array_equal(dyno.prediction.trajectory.state_list[-1].position, final_state)


# Run the tests
if __name__ == "__main__":
    pytest.main()
