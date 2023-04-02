from crpred.predictor_interface import PredictorInterface
from crpred.utility.config import PredictorParams
from commonroad.scenario.scenario import Scenario


class AgentPredictor(PredictorInterface):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super().__init__(config=config)
        self.list_agents = []

    def _assign_agents_to_lanelets(self, sc: Scenario, time_step: int):
        for agent in self.list_agents:
            if not agent.time_step_current == time_step:
                continue

            for id_lanelet_current in agent.set_ids_lanelets_current:
                lanelet = sc.lanelet_network.find_lanelet_by_id(id_lanelet_current)
                try:
                    lanelet.dynamic_obstacles_on_lanelet[time_step].add(agent.agent_id)

                except KeyError:
                    lanelet.dynamic_obstacles_on_lanelet[time_step] = {agent.id_agent}

    def predict(self, sc: Scenario, initial_time_step: int = 0):
        """
        Abstract method for performing predictions.

        :param sc: Scenario containing no predictions for obstacles.
        :param initial_time_step: Time step to start prediction.
        :return: CommonRoad scenario containing predictions.
        """
