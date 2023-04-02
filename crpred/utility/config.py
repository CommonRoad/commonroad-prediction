import dataclasses
import inspect
from enum import Enum
from dataclasses import dataclass, field
import pathlib
from typing import Dict, Union, Any, List
from omegaconf import OmegaConf


class PredictorType(Enum):
    """Enum containing all possible predictor types defined."""

    # ConstantAccelerationCurvilinear: motion model predictor
    CONSTANT_ACCELERATION_CURVILINEAR = "constant_acceleration_curvilinear"
    # ConstantVelocityCurvilinear: motion model predictor
    CONSTANT_VELOCITY_CURVILINEAR = "constant_velocity_curvilinear"
    # ConstantAccelerationLinear: motion model predictor
    CONSTANT_ACCELERATION_LINEAR = "constant_acceleration_linear"
    # ConstantVelocityLinear: motion model predictor
    CONSTANT_VELOCITY_LINEAR = "constant_velocity_linear"
    # IDM: intelligent driver model predictor
    IDM = "idm"
    # MOBIL: minimizing overall braking induced by lane change predictor
    MOBIL = "mobil"
    # FOLLOW_ROAD_1: follow road predictor 1
    FOLLOW_ROAD_1 = "follow_road_1"
    # FOLLOW_ROAD_2: follow road predictor 2
    FOLLOW_ROAD_2 = "follow_road_2"
    # FOLLOW_ROAD_3: follow road predictor 3
    FOLLOW_ROAD_3 = "follow_road_3"
    # GROUND_TRUTH: ground truth prediction (noninteractive scenarios only)
    GROUND_TRUTH = "ground_truth"

    @classmethod
    def values(cls) -> List[str]:
        """Extract all predictor names"""
        return [item.value for item in cls]


def _dict_to_params(dict_params: Dict, cls: Any) -> Any:
    """
    Converts dictionary to parameter class.

    :param dict_params: Dictionary containing parameters.
    :param cls: Parameter dataclass to which dictionary should be converted to.
    :return: Parameter class.
    """
    fields = dataclasses.fields(cls)
    cls_map = {f.name: f.type for f in fields}
    kwargs = {}
    for k, v in cls_map.items():
        if k not in dict_params:
            continue
        if inspect.isclass(v) and issubclass(v, BaseParam):
            kwargs[k] = _dict_to_params(dict_params[k], cls_map[k])
        else:
            kwargs[k] = dict_params[k]
    return cls(**kwargs)


@dataclass
class BaseParam:
    """CommonRoad-Prediction base parameters."""

    dt: float = 0.1
    num_steps_prediction: int = 60
    predictor: PredictorType = PredictorType.GROUND_TRUTH
    __initialized: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        """Post initialization of base parameter class."""
        # pylint: disable=unused-private-member
        self.__initialized = True
        # Make sure that the base parameters are propagated to all sub-parameters
        # This cannot be done in the init method, because the sub-parameters are not yet initialized.
        # This is not a noop, as it calls the __setattr__ method.
        # Do not remove!
        self.dt = self.dt
        self.num_steps_prediction = self.num_steps_prediction
        self.predictor = self.predictor

    def __getitem__(self, item: str) -> Any:
        """
        Getter for base parameter value.

        :param: Item for which content should be returned.
        :return: Item value.
        """
        try:
            value = self.__getattribute__(item)
        except AttributeError as e:
            raise KeyError(f"{item} is not a parameter of {self.__class__.__name__}") from e
        return value

    def __setitem__(self, key: str, value: Any):
        """
        Setter for item.

        :param key: Name of item.
        :param value: Value of item.
        """
        try:
            self.__setattr__(key, value)
        except AttributeError as e:
            raise KeyError(f"{key} is not a parameter of {self.__class__.__name__}") from e

    @classmethod
    def load(cls, file_path: Union[pathlib.Path, str], validate_types: bool = True) -> 'BaseParam':
        """
        Loads config file and creates parameter class.

        :param file_path: Path to yaml file containing config parameters.
        :param validate_types:  Boolean indicating whether loaded config should be validated against CARLA parameters.
        :return: Base parameter class.
        """
        file_path = pathlib.Path(file_path)
        assert file_path.suffix == ".yaml", f"File type {file_path.suffix} is unsupported! Please use .yaml!"
        loaded_yaml = OmegaConf.load(file_path)
        if validate_types:
            OmegaConf.merge(OmegaConf.structured(PredictorParams), loaded_yaml)
        params = _dict_to_params(OmegaConf.to_object(loaded_yaml), cls)
        return params

    def save(self, file_path: Union[pathlib.Path, str]):
        """
        Save config parameters to yaml file.

        :param file_path: Path where yaml file should be stored.
        """
        # Avoid saving private attributes
        dict_cfg = dataclasses.asdict(self, dict_factory=lambda items: {key: val for key, val in items if
                                                                        not key.startswith("_")})
        OmegaConf.save(OmegaConf.create(dict_cfg), file_path, resolve=True)


@dataclass
class IDMParams(BaseParam):
    """Parameters of IDM prediction algorithm."""

    v_desired: float = 20.0
    dist_min: float = 4.0
    time_headway: float = 1.0
    a_lon_max: float = 3.0
    a_lon_min: float = -6.0
    a_lon_comfort: float = 1.5
    coef: float = 4


@dataclass
class MobilParams(BaseParam):
    """Parameters of Mobil prediction algorithm."""

    v_desired: float = 20.0
    dist_min: float = 4.0
    time_headway: float = 2.0
    a_lon_max: float = 3.0
    a_lon_min: float = -6.0
    a_lon_comfort: float = 1.5
    coef: float = 4.0
    b_safe: float = 2.0
    p: float = 0.1
    a_th: float = 0.1
    a_bias: float = 0.3
    v_crit: float = 10.0
    assert a_th >= 0, "<MOBIL> a_th is negative"


@dataclass
class PredictorParams(BaseParam):
    """All CR-Prediction parameters"""

    idm: IDMParams = field(default_factory=IDMParams)
    mobil: MobilParams = field(default_factory=MobilParams)
