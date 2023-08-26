import numpy as np
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork


def clear_obstacle_trajectory(sc: Scenario, initial_time_step: int = 0):
    """
    Clears obstacle trajectories of scenario.

    :param sc: CommonRoad scenario.
    :param initial_time_step: Initial time step of obstacles.
    """
    # clear existing prediction trajectories and lanelet assignment
    for obstacle in sc.dynamic_obstacles:
        # clear dynamic obstacles on lanelets (except for the first step)
        for time_step in range(initial_time_step + 1, len(obstacle.prediction.trajectory.state_list) + 1):
            for lanelet in sc.lanelet_network.lanelets:
                try:
                    lanelet.dynamic_obstacles_on_lanelet.get(time_step).discard(obstacle.obstacle_id)

                except AttributeError:
                    lanelet.dynamic_obstacles_on_lanelet[time_step] = set()

        obstacle.prediction = None


def get_merged_laneletes_from_position(lanelet_network: LaneletNetwork, position: np.ndarray):
    lanelet_id_list = lanelet_network.find_lanelet_by_position([position])
    if not any(lanelet_id_list):
        raise ValueError(f"Position {position} cannot be assigned to a lanelet.")

    current_lanelet = lanelet_network.find_lanelet_by_id(lanelet_id_list[0][0])
    merged_lanelets, merged_lanelets_id = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
        current_lanelet, lanelet_network, 300
    )
    return merged_lanelets, merged_lanelets_id
