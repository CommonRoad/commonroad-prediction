from typing import Dict, List, Optional

from commonroad.scenario.obstacle import State

from crpred.advanced_models.idm_predictor import IDMPredictor
from crpred.advanced_models.mobil_predictor import MOBILPredictor
from crpred.ground_truth_predictor import GroundTruthPredictor
from crpred.predictor_interface import PredictorInterface
from crpred.utility.config import PredictorParams, PredictorType


class PredictorFactory:
    """Factory class for different predictors."""

    dict_predictors = {
        PredictorType.IDM: IDMPredictor,
        PredictorType.MOBIL: MOBILPredictor,
        PredictorType.GROUND_TRUTH: GroundTruthPredictor,
    }

    @staticmethod
    def create_predictor(
        config: PredictorParams, previous_states: Optional[Dict[int, List[State]]] = None
    ) -> PredictorInterface:
        """
        Creates predictor.

        :param config: Predictor configuration parameters.
        :param previous_states: History of states of obstacle.
        :return: CommonRoad predictor interface.
        """
        predictor_type = PredictorType(config.predictor)

        if previous_states:
            #  predictor = PredictorFactory.dict_predictors[predictor_type](config=config,
            #  previous_states=previous_states)
            predictor = None
        else:
            predictor = PredictorFactory.dict_predictors[predictor_type](config=config)

        return predictor
