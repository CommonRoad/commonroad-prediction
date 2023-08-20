import numpy as np
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import State, InitialState
from commonroad.scenario.trajectory import Trajectory


def create_test_scenario() -> Scenario:
    shape = Rectangle(3, 2)

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
        prediction=TrajectoryPrediction(
            trajectory=Trajectory(
                initial_time_step=0,
                state_list=[State(time_step=i) for i in range(10)],
            ),
            shape=shape,
        )
    )

    test_sc = Scenario(dt=0.1)  # You may customize this as needed
    # test_sc._dynamic_obstacles = [test_dyno]
    test_sc._dynamic_obstacles = {0: test_dyno}
    return test_sc
