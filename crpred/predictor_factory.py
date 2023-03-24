from typing import Optional, Dict, List

from commonroad.scenario.obstacle import State

from crpred.utility.config import PredictorType, PredictorParams
from crpred.predictor_interface import PredictorInterface
from crpred.advanced_models.idm_predictor import IDMPredictor
from crpred.advanced_models.mobil_predictor import MOBILPredictor
from crpred.motion_models.ConstantAccelerationCurvilinearPredictor import ConstantAccelerationCurvilinearPredictor
from crpred.motion_models.ConstantVelocityCurvilinearPredictor import ConstantVelocityCurvilinearPredictor
from crpred.motion_models.ConstantAccelerationLinearPredictor import ConstantAccelerationLinearPredictor
from crpred.motion_models.ConstantVelocityLinearPredictor import ConstantVelocityLinearPredictor
from crpred.follow_road.FollowTheRoadPredictor import FollowTheRoadPredictor
from crpred.follow_road.FollowTheRoadPredictor2 import FollowTheRoadPredictor2
from crpred.follow_road.FollowTheRoadPredictor3 import FollowTheRoadPredictor3
from crpred.ground_truth_predictor import GroundTruthPredictor


class PredictorFactory:
    dict_predictors = {
        PredictorType.CONSTANT_ACCELERATION_CURVILINEAR: ConstantAccelerationCurvilinearPredictor,
        PredictorType.CONSTANT_VELOCITY_CURVILINEAR: ConstantVelocityCurvilinearPredictor,
        PredictorType.CONSTANT_ACCELERATION_LINEAR: ConstantAccelerationLinearPredictor,
        PredictorType.CONSTANT_VELOCITY_LINEAR: ConstantVelocityLinearPredictor,
        PredictorType.IDM: IDMPredictor,
        PredictorType.MOBIL: MOBILPredictor,
        PredictorType.FOLLOW_ROAD_1: FollowTheRoadPredictor,
        PredictorType.FOLLOW_ROAD_2: FollowTheRoadPredictor2,
        PredictorType.FOLLOW_ROAD_3: FollowTheRoadPredictor3,
        PredictorType.GROUND_TRUTH: GroundTruthPredictor
    }

    @staticmethod
    def create_predictor(config: PredictorParams, previous_states: Optional[Dict[int, List[State]]] = None) \
            -> PredictorInterface:
        predictor_type = PredictorType(config.predictor)

        if previous_states:
            predictor = PredictorFactory.dict_predictors[predictor_type](config=config, previous_states=previous_states)
        else:
            predictor = PredictorFactory.dict_predictors[predictor_type](config=config)

        return predictor
