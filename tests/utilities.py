import numpy as np
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario, LaneletNetwork, Lanelet
from commonroad.scenario.state import State, InitialState
from commonroad.scenario.trajectory import Trajectory


def create_straight_scenario(initial_state: InitialState, dt: float = 0.1) -> Scenario:
    shape = Rectangle(3, 2)

    test_dyno = DynamicObstacle(
        obstacle_id=0,
        obstacle_type=ObstacleType.CAR,
        obstacle_shape=shape,
        initial_state=initial_state,
    )

    test_sc = Scenario(dt)
    test_sc.add_objects(_create_straight_lanelet_network())
    test_sc._dynamic_obstacles = {0: test_dyno}
    return test_sc


def create_const_velocity_straight_scenario(velocity: float) -> Scenario:
    return create_straight_scenario(initial_state=InitialState(
        position=np.array([0.0, 0.0]),
        orientation=0.0,
        velocity=velocity,
        acceleration=0.0,
        yaw_rate=0.0,
        time_step=0
    ))


def create_const_acceleration_straight_scenario(acceleration: float) -> Scenario:
    return create_straight_scenario(initial_state=InitialState(
        position=np.array([0.0, 0.0]),
        orientation=0.0,
        velocity=0.0,
        acceleration=acceleration,
        yaw_rate=0.0,
        time_step=0
    ))


def create_const_yaw_rate_straight_scenario(yaw_rate: float) -> Scenario:
    return create_straight_scenario(initial_state=InitialState(
        position=np.array([0.0, 0.0]),
        orientation=0.0,
        velocity=0.0,
        acceleration=0.0,
        yaw_rate=yaw_rate,
        time_step=0
    ))


def create_default_scenario() -> Scenario:
    return create_straight_scenario(InitialState(
        position=np.array([0.0, 0.0]),
        orientation=0.0,
        velocity=0.0,
        acceleration=0.0,
        yaw_rate=0.0,
        time_step=0
    ))


def _create_straight_lanelet_network() -> LaneletNetwork:
    # Create the left, right, and center vertices of the lane
    x = np.arange(0, 100)
    center_vertices = np.dstack((x, np.zeros_like(x)))[0]
    left_vertices = np.dstack((x, 2 * np.ones_like(x)))[0]
    right_vertices = np.dstack((x, -2 * np.ones_like(x)))[0]

    lanelet_network = LaneletNetwork.create_from_lanelet_list(
        [Lanelet(
            left_vertices=left_vertices,
            right_vertices=right_vertices,
            center_vertices=center_vertices,
            lanelet_id=0,
        )]
    )

    return lanelet_network
