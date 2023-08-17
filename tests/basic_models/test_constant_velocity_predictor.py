import pytest
import numpy as np

from crpred.basic_models.constant_velocity_predictor import ConstantVelocityLinearPredictor
from crpred.utility.config import PredictorParams
from commonroad.scenario.state import State, InitialState
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType


# from your_module import ConstantVelocityLinearPredictor  # Import your class implementation
# from your_module import Scenario, PredictorParams, DynamicObstacle, Trajectory, State


# Import other necessary dependencies or classes

class TestConstantVelocityLinearPredictor:

    def test_predict(self):
        # Create a test scenario and initial parameters
        test_config = PredictorParams()  # You may customize this as needed
        predictor = ConstantVelocityLinearPredictor(config=test_config)

        initial_state = InitialState(
            position=np.array([0.0, 0.0]),
            orientation=0.0,
            velocity=10.0,
            acceleration=0.0,
            time_step=0
        )
        test_dyno = DynamicObstacle(
            obstacle_id=0,
            obstacle_type=ObstacleType.CAR,
            obstacle_shape=Rectangle(3, 2),
            initial_state=initial_state,
        )

        test_sc = Scenario(dt=0.1)  # You may customize this as needed
        test_sc._dynamic_obstacles = [test_dyno]

        # Call the predict method
        pred_sc = predictor.predict(test_sc)

        # Perform assertions on the predicted scenario
        assert len(pred_sc.dynamic_obstacles) == 1
        pred_dyno = pred_sc.dynamic_obstacles[0]
        pred_trajectory = pred_dyno.prediction.trajectory
        pred_state_list = pred_trajectory.state_list

        assert len(pred_state_list) == test_config.num_steps_prediction
        assert pred_state_list[0].position[0] == pytest.approx(0.0)
        assert pred_state_list[0].position[1] == pytest.approx(0.0)
        assert pred_state_list[0].orientation == pytest.approx(0.0)
        assert pred_state_list[0].velocity == pytest.approx(10.0)
        assert pred_state_list[0].acceleration == pytest.approx(0.0)

        # Add more assertions as needed based on your implementation and requirements


# Run the tests
if __name__ == "__main__":
    pytest.main()
