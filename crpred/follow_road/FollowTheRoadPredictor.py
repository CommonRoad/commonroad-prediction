from typing import Dict
from sympy.geometry import Point2D, Segment2D, Circle

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import *

from motion_planner_components.prediction.follow_road.FollowTheRoadPredictorInterface import \
    FollowTheRoadPredictorInterface
from motion_planner_components.prediction.follow_road.util import point_smallest_delta_angle, \
    circle_line_segment_collision_detection
from motion_planner_config.configuration_builder import Configuration


class FollowTheRoadPredictor(FollowTheRoadPredictorInterface):

    def __init__(self, config: Configuration, previous_states: Optional[Dict[int, List[State]]] = None):
        super().__init__(config, previous_states)

    @staticmethod
    def points_on_lane_with_distance(center: Point2D,
                                     distance: float,
                                     lane_points: List[Point2D]) -> List[Point2D]:
        """
        Return a list of points which are on the lane and have distance to point
        Geometry: Circle around point, points where lane crosses circle
        """
        circle: Circle = Circle(center, distance)
        intersections: List[Point2D] = []

        for p1, p2 in zip(lane_points, lane_points[1:]):
            segment: Segment2D = Segment2D(p1, p2)

            if not circle_line_segment_collision_detection(center, distance, segment):
                continue

            intersects: List[Point2D] = circle.intersection(segment)

            for i in intersects:
                if not intersections.__contains__(i):
                    intersections.append(i)

        return intersections

    def follow_lane(self, lane_points: List[Point2D], state: State) -> State:
        """
        Return the next p
        """
        # TODO: SET CORRECT DIRECTION
        current_position: Point2D = Point2D(state.position[0], state.position[1])
        position_delta: float = state.velocity / 10

        points_on_lane_with_distance: List[Point2D] = \
            FollowTheRoadPredictor.points_on_lane_with_distance(current_position,
                                                                position_delta,
                                                                lane_points)

        selected_point: Point2D = point_smallest_delta_angle(current_position,
                                                             state.orientation,
                                                             points_on_lane_with_distance)

        position_x: float = round(selected_point.x, 1)
        position_y: float = round(selected_point.y, 1)
        position: np.array = np.array([position_x, position_y])

        kwargs = {
            "velocity": state.velocity,
            "orientation": state.orientation,
            "time_step": state.time_step + 1,
            "position": position
        }

        return State(**kwargs)
