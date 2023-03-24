
from statistics import mean
from motion_planner_config.configuration_builder import Configuration
import commonroad.scenario.lanelet as lanelet
from commonroad_dc.geometry.util import compute_pathlength_from_polyline, compute_orientation_from_polyline
from commonroad_dc.costs.route_matcher import *


from motion_planner_components.prediction.motion_models.MotionModelInterface import MotionModelInterface
from motion_planner_components.prediction.motion_models.util import _difference
from commonroad.scenario.trajectory import *


class ConstantAccelerationCurvilinearPredictor(MotionModelInterface):
    """"
    A Curveilinear, constant acceleration motion model
    Look into the paper "Comparison and Evaluation of Advanced Motion Models for Vehicle Tracking" by
    Robin Schubert, Eric Richter, Gerd Wanielik for reference
    """

    def __init__(self, config: Configuration, previous_states: Optional[Dict[int, List[State]]] = None):
        super().__init__(config, previous_states)

    def state_prediction_generator(self, state_list: List[State]) -> List[State]:
        """
        Next states are generated assuming acceleration and constant turn rate of the vehicle
        We assume that vehicle is going to turn the same as in the past and also accelerate as in the past
        """
        prediction_state_list: List[State] = []
        last_state: State = state_list[-1]
        last_velocity: float = last_state.velocity
        init_time_step: int = 0
        position: np.ndarray = last_state.position

        if len([s.orientation for s in state_list]) == 1:
            avg_orientation_delta = [s.orientation for s in state_list][-1]
        else:
            avg_orientation_delta: float = mean(_difference([s.orientation for s in state_list]))  # Turn Rate

        if len([s.velocity for s in state_list]) == 1:
            avg_velocity_delta: float = [s.velocity for s in state_list][-1]
        else:
            avg_velocity_delta: float = mean(_difference([s.velocity for s in state_list]))  # acceleration


        current_state = np.array([0, 0], dtype=np.float64)
        current_state[0] = position[0]
        current_state[1] = position[1]
        lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([current_state])[0][0]
        # find current lanelet the ego vehicule is currently on
        lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([position])[0][0]
        current_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
        list1, list2 = lanelet.Lanelet.all_lanelets_by_merging_successors_from_lanelet(current_lanelet,
                                                                                       self.scenario.lanelet_network,
                                                                                       300)
        ref_path = list1[0].center_vertices

        curvilinear_cosy = create_cosy_from_lanelet(list1[0])

        #####################################
        wheelbase = self.configuration.vehicle.ego.wheelbase
        x, y = position
        orientation = last_state.orientation
        v = last_velocity

        if self.configuration.planning.reference_point == "CENTER":
            p_lon, p_lat = curvilinear_cosy.convert_to_curvilinear_coords(x, y)

        elif self.configuration.planning.reference_point == "REAR":
            p_lon, p_lat = curvilinear_cosy.convert_to_curvilinear_coords(
                x - wheelbase / 2 * np.cos(orientation), y - wheelbase / 2 * np.sin(orientation))

        else:
            raise Exception(f"Unknown reference point: {self.configuration.planning.reference_point}")

        ref_orientation = compute_orientation_from_polyline(ref_path)
        ref_path_length = compute_pathlength_from_polyline(ref_path)


        current_state = np.array([0, 0], dtype=np.float64)
        current_state[0] = position[0]
        current_state[1] = position[1]
        lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([current_state])[0][0]
        current_lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
        ref_path, list2 = lanelet.Lanelet.all_lanelets_by_merging_successors_from_lanelet(
            current_lanelet, self.scenario.lanelet_network, 300)
        merged_lanelet, list2 = lanelet.Lanelet.all_lanelets_by_merging_successors_from_lanelet(current_lanelet,
                                                                                self.scenario.lanelet_network, 300)
        ref_path = merged_lanelet[0].center_vertices

        # create curvilinear coordinate system
        curvilinear_cosy = create_cosy_from_lanelet(merged_lanelet[0])

        wheelbase = self.configuration.vehicle.ego.wheelbase
        x, y = position
        orientation = last_state.orientation
        v = last_velocity

        if self.configuration.planning.reference_point == "CENTER":
            p_lon, p_lat = curvilinear_cosy.convert_to_curvilinear_coords(x, y)

        elif self.configuration.planning.reference_point == "REAR":
            p_lon, p_lat = curvilinear_cosy.convert_to_curvilinear_coords(
                x - wheelbase / 2 * np.cos(orientation), y - wheelbase / 2 * np.sin(orientation))

        else:
            raise Exception(f"Unknown reference point: {self.configuration.planning.reference_point}")

        v_lon = v
        v_lat = v
        ref_orientation = compute_orientation_from_polyline(ref_path)
        ref_path_length = compute_pathlength_from_polyline(ref_path)

        for i in range(1,  self.num_steps_prediction+1):
            orientation_interpolated = np.interp(p_lon, ref_path_length, ref_orientation)
            velocity: float = last_velocity + avg_velocity_delta * i
            v_lon: float = v * np.cos(orientation - orientation_interpolated)
            v_lat: float = v * np.sin(orientation - orientation_interpolated)
            orientation: int = last_state.orientation + (avg_orientation_delta * i)
            delta_x: float = math.cos(orientation) * v_lon * 0.1
            delta_y: float = math.sin(orientation) * v_lat * 0.1
            position_x: float = position[0] + (i * delta_x)
            position_y: float = position[1] + (i * delta_y)
            orientation_interpolated = np.interp(p_lon, ref_path_length, ref_orientation)
            v_lon: float = v_lon * np.cos(orientation - orientation_interpolated)
            v_lat: float = v_lat * np.sin(orientation - orientation_interpolated)
            orientation: float = orientation + avg_orientation_delta * i
            delta_p_lon: float = np.cos(orientation - orientation_interpolated) * v_lon * 0.1
            delta_p_lat: float = np.sin(orientation - orientation_interpolated) * v_lat * 0.1
            p_lon: float = p_lon + delta_p_lon
            p_lat: float = p_lat + delta_p_lat

            try:
                position = curvilinear_cosy.convert_to_cartesian_coords(p_lon, p_lat)
            except ValueError as e:
                print(position)

            kwargs: Dict[Any] = {
                "velocity": last_velocity + i * avg_velocity_delta,
                "orientation": get_orientation_at_position(curvilinear_cosy, position),
                "time_step": init_time_step + i,
                "position": position
            }

            state: State = State(**kwargs)
            prediction_state_list.append(state)

        return prediction_state_list

