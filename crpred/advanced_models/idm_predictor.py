from commonroad.scenario.scenario import Scenario

from crpred.advanced_models.agent_predictor import AgentPredictor
from crpred.advanced_models.idm_agent import IDMAgent
from crpred.utility.common import clear_obstacle_trajectory
from crpred.utility.config import PredictorParams


class IDMPredictor(AgentPredictor):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super().__init__(config=config)

    def predict(self, sc: Scenario, initial_time_step: int = 0) -> Scenario:
        clear_obstacle_trajectory(sc, initial_time_step)
        self.list_agents: list[IDMAgent] = []
        for obs in sc.dynamic_obstacles:
            self.list_agents.append(IDMAgent(obs, sc))

        for time_step in range(initial_time_step + 1, initial_time_step + self._config.num_steps_prediction + 1):
            for agent in self.list_agents:
                agent.step_forward(time_step)

            self._assign_agents_to_lanelets(sc, time_step)

        return sc

    # def visualize(self, sc: Scenario, step=0):
    #     util_visualization.plot_scenario(sc, step_start=step, step_end=step + self._config.num_steps_prediction,
    #                                      predictor_type="idm", plot_occupancies=True)
