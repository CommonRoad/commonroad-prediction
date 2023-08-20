import pytest

from crpred.basic_models.constant_velocity_predictor import ConstantVelocityLinearPredictor
from crpred.utility.config import PredictorParams
from commonroad.common.file_reader import CommonRoadFileReader


class TestConstantVelocityLinearPredictor:
    def setup_class(self):
        self.test_config = PredictorParams(num_steps_prediction=60)  # You may customize this as needed
        self.predictor = ConstantVelocityLinearPredictor(config=self.test_config)

    def test_predict(self):
        test_sc, _ = CommonRoadFileReader("USA_US101-3_1_T-1.xml").open(lanelet_assignment=True)
        # Call the predict method
        pred_sc = self.predictor.predict(test_sc)

        # Perform assertions on the predicted scenario
        # assert len(pred_sc.dynamic_obstacles) == 1
        pred_dyno = pred_sc.dynamic_obstacles[0]
        pred_trajectory = pred_dyno.prediction.trajectory
        pred_state_list = pred_trajectory.state_list

        # assert len(pred_state_list) == self.test_config.num_steps_prediction
        assert pred_state_list[0].position[0] == pytest.approx(0.0)
        assert pred_state_list[0].position[1] == pytest.approx(0.0)
        assert pred_state_list[0].orientation == pytest.approx(0.0)
        assert pred_state_list[0].velocity == pytest.approx(10.0)
        assert pred_state_list[0].acceleration == pytest.approx(0.0)

        # Add more assertions as needed based on your implementation and requirements


# Run the tests
if __name__ == "__main__":
    pytest.main()
