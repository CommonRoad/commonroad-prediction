from sympy.geometry import Point2D
from motion_planner_components.prediction.follow_road.FollowTheRoadPredictorInterface import \
    FollowTheRoadPredictorInterface
from motion_planner_components.prediction.follow_road.util import point_smallest_delta_angle
from motion_planner_config.configuration_builder import Configuration
import commonroad.scenario.lanelet as lanelet
from commonroad_dc.geometry.util import compute_pathlength_from_polyline, compute_orientation_from_polyline
from commonroad_dc.costs.route_matcher import *
from commonroad.scenario.trajectory import *


class FollowTheRoadPredictor3(FollowTheRoadPredictorInterface):

    def __init__(self, config: Configuration, previous_states: Optional[Dict[int, List[State]]] = None):
        super().__init__(config, previous_states)

    @staticmethod
    def closest_n_point_on_lane(center: Point2D, lane_points: List[Point2D], n: int) -> List[Point2D]:
        """
        Find the closest n lanelet Points given a point.
        """
        lane_points_copy: List[Point2D] = lane_points.copy()
        closet_points: List[Point2D] = []
        distances: Dict[Point2D, float] = {}

        for p in lane_points_copy:
            distances[p] = center.distance(p)

        for i in range(n - 1):
            p: Point2D = min(lane_points_copy, key=lambda t: distances[t])
            closet_points.append(p)
            lane_points_copy.remove(p)

        return closet_points

    def follow_lane(self, lane_points: List[Point2D], state: State) -> State:
        """
        Return the next State that the vehicle would drive if we assume that the vehicle follows the road
        """
        # naive next point using ConstantVelocityLinearPredictor
        velocity = state.velocity
        orientation = state.orientation
        position: np.ndarray = state.position

        # find current lanelet the ego vehicule is currently on
        lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([position])[0][0]
        current_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
        merged_lanelet, list2 = lanelet.Lanelet.all_lanelets_by_merging_successors_from_lanelet(current_lanelet,
                                                                                self.scenario.lanelet_network, 300)
        ref_path = merged_lanelet[0].center_vertices

        # create curvilinear coordinate system
        curvilinear_cosy = create_cosy_from_lanelet(merged_lanelet[0])

        wheelbase = self.configuration.vehicle.ego.wheelbase
        x, y = position

        if self.configuration.planning.reference_point == "CENTER":
            p_lon, p_lat = curvilinear_cosy.convert_to_curvilinear_coords(x, y)

        elif self.configuration.planning.reference_point == "REAR":
            p_lon, p_lat = curvilinear_cosy.convert_to_curvilinear_coords(
                x - wheelbase / 2 * np.cos(orientation), y - wheelbase / 2 * np.sin(orientation))

        else:
            raise Exception(f"Unknown reference point: {self.configuration.planning.reference_point}")

        ref_orientation = compute_orientation_from_polyline(ref_path)
        ref_path_length = compute_pathlength_from_polyline(ref_path)
        orientation_interpolated = np.interp(p_lon, ref_path_length, ref_orientation)
        v_lon: float = velocity * np.cos(orientation - orientation_interpolated)
        v_lat: float = velocity * np.sin(orientation - orientation_interpolated)

        delta_p_lon: float = np.cos(orientation - orientation_interpolated) * v_lon * 0.1
        delta_p_lat: float = np.sin(orientation - orientation_interpolated) * v_lat * 0.1

        try:
            position_car = curvilinear_cosy.convert_to_cartesian_coords(p_lon + delta_p_lon, p_lat + delta_p_lat)
        except ValueError as e:
            print(position_car)

        naive_next_point: Point2D = Point2D(position_car[0], position_car[1])

        closest_points: List[Point2D] =\
            FollowTheRoadPredictor3.closest_n_point_on_lane(naive_next_point, lane_points, 3)
        selected_point = point_smallest_delta_angle(Point2D(position[0], position[1]), orientation, closest_points)

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
