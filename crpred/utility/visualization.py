import logging
import time
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer

logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def plot_scenario(scenario: Scenario, figsize: Tuple = (25, 15),
                  step_start: int = 0, step_end: int = 10, steps: List[int] = None,
                  plot_limits: List = None, path_output: str = None,
                  save_gif: bool = True, duration: float = None,
                  save_plots: bool = True, show_lanelet_label: bool = False,
                  predictor_type: str = None, plot_occupancies: bool = False):
    """
    Plots scenarios with predicted motions.
    """
    path_output = path_output or "./plots/" + str(scenario.scenario_id)
    Path(path_output).mkdir(parents=True, exist_ok=True)

    plot_limits = plot_limits if plot_limits else compute_plot_limits_from_lanelet_network(scenario.lanelet_network)
    if steps:
        steps = [step for step in steps if step <= step_end + 1]
    else:
        steps = range(step_start, step_end + 1)
    duration = duration if duration else scenario.dt

    renderer = MPRenderer(plot_limits=plot_limits, figsize=figsize)
    time_stamps = {}
    for step in steps:
        time_step = step
        if save_plots:
            # clear previous plot
            plt.cla()
        else:
            # create new figure
            plt.figure(figsize=figsize)
            renderer = MPRenderer(plot_limits=plot_limits)

        # plot scenario and planning problem
        scenario.draw(renderer, draw_params={"dynamic_obstacle": {"draw_icon": True},
                                             "trajectory": {"draw_trajectory": True},
                                             "time_begin": time_step,
                                             "lanelet": {"show_label": show_lanelet_label}})

        # settings and adjustments
        plt.rc("axes", axisbelow=True)
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.set_title(f"$t = {time_step / 10.0:.1f}$ [s]", fontsize=28)
        ax.set_xlabel(f"$s$ [m]", fontsize=28)
        ax.set_ylabel("$d$ [m]", fontsize=28)
        plt.margins(0, 0)
        renderer.render()

        if save_plots:
            time_stamp = int(time.time())
            time_stamps[step] = time_stamp
            save_fig(save_gif, path_output, step, time_stamp, predictor_type)

        else:
            plt.show()

    if plot_occupancies:
        time_begin = steps[0]
        if save_plots:
            # clear previous plot
            plt.cla()
        else:
            # create new figure
            plt.figure(figsize=figsize)
            renderer = MPRenderer(plot_limits=plot_limits)

        # plot scenario and planning problem
        scenario.draw(renderer, draw_params={"dynamic_obstacle": {"draw_icon": True},
                                             "time_begin": time_begin,
                                             "lanelet": {"show_label": show_lanelet_label}})

        for obs in scenario.dynamic_obstacles:
            [occ.draw(renderer) for occ in obs.prediction.occupancy_set if occ.time_step in steps]

        # settings and adjustments
        plt.rc("axes", axisbelow=True)
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.set_title(f"$t = {time_begin}$ [s]", fontsize=28)
        ax.set_xlabel(f"$s$ [m]", fontsize=28)
        ax.set_ylabel("$d$ [m]", fontsize=28)
        plt.margins(0, 0)
        renderer.render()


def compute_plot_limits_from_lanelet_network(lanelet_network: LaneletNetwork, margin: int = 10):
    list_vertices_x = []
    list_vertices_y = []
    for lanelet in lanelet_network.lanelets:
        vertex_center = lanelet.center_vertices
        list_vertices_x.extend(list(vertex_center[:, 0]))
        list_vertices_y.extend(list(vertex_center[:, 1]))

    x_min, x_max = min(list_vertices_x), max(list_vertices_x)
    y_min, y_max = min(list_vertices_y), max(list_vertices_y)
    plot_limits = [x_min - margin, x_max + margin, y_min - margin, y_max + margin]

    return plot_limits
