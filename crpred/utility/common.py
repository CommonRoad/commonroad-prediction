from commonroad.scenario.scenario import Scenario


def clear_obstacle_trajectory(sc: Scenario, initial_time_step: int = 0):
    # clear existing prediction trajectories and lanelet assignment
    for obstacle in sc.dynamic_obstacles:
        # clear dynamic obstacles on lanelets (except for the first step)
        for time_step in range(initial_time_step + 1, len(obstacle.prediction.trajectory.state_list) + 1):
            for lanelet in sc.lanelet_network.lanelets:
                try:
                    lanelet.dynamic_obstacles_on_lanelet.get(time_step).discard(obstacle.obstacle_id)

                except AttributeError:
                    lanelet.dynamic_obstacles_on_lanelet[time_step] = set()

        obstacle.prediction.center_lanelet_assignment = dict()
        obstacle.prediction.shape_lanelet_assignment = dict()
        obstacle.prediction.trajectory.state_list.clear()
        obstacle.prediction.occupancy_set.clear()
        obstacle.prediction.trajectory.initial_time_step = initial_time_step + 1