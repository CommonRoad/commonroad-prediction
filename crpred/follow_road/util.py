from typing import List, Union
from sympy.geometry import Point2D, Line, Segment2D
from math import cos, sin


def point_smallest_delta_angle(center_point: Point2D, orientation: float, points: List[Point2D]) -> Point2D:
    """
    Choose the point the has the smallest angle with the current position-orientation
    """
    naive_future_line_point: Point2D = Point2D(cos(orientation), sin(orientation))
    naive_path: Line = Line(center_point, naive_future_line_point)

    best_predicted_point: Union[Point2D, None] = None
    best_predicted_point_angle: float = float('inf')

    for p in points:
        if p == center_point:
            continue

        angle: float = naive_path.angle_between(Line(center_point, p))

        if best_predicted_point is None or angle < best_predicted_point_angle:
            best_predicted_point = p
            best_predicted_point_angle = angle

    return best_predicted_point


def circle_line_segment_collision_detection(center: Point2D,
                                            radius: float,
                                            segment: Segment2D) -> bool:
    """
    Check if a circle and a lane segment can not collide
    https://www.baeldung.com/cs/circle-line-segment-collision-detection#line-segment-amp-circle-intersection
    """

    return segment.distance(center) <= radius


def partition_line_segment(p1: Point2D, p2: Point2D, d: float) -> Point2D:
    """
    Partition the line segment, d is in [0,1]
    So d=0.5 returns the center, and d=0 the frist point
    """
    if not 0 <= d <= 1:
        raise Exception("partition distance has to be between 0 and 1")

    return Point2D((d * (p1.x + p2.x)), (d * (p1.y + p2.y)))
