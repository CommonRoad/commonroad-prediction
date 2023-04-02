from crpred.advanced_models.agent_predictor import AgentPredictor
from crpred.advanced_models.mobil_agent import MOBILAgent
from crpred.utility import visualization as util_visualization
from crpred.utility.config import PredictorParams


class MOBILPredictor(AgentPredictor):
    def __init__(self, config: PredictorParams = PredictorParams()):
        super(MOBILPredictor, self).__init__(config=config)
        self.num_steps_prediction = config.planning.time_steps_computation

        MOBILAgent.cls_scenario_current = self.scenario
        for obs in self.scenario.dynamic_obstacles:
            self.list_agents.append(MOBILAgent(obs))

    def predict(self):
        for time_step in range(1, self.num_steps_prediction + 1):
            for agent in self.list_agents:
                agent.step_forward(time_step)

            self._assign_agents_to_lanelets(time_step)
        return self.scenario
