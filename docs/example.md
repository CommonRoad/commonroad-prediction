The subsequent code snippet shows a minimal example on how to use the CommonRoad Prediction framework.
Further examples can be found [here](https://github.com/commonroad/commonroad-prediction/tutorials).

```Python
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader

from crpred.basic_models.constant_velocity_predictor import (
    ConstantVelocityCurvilinearPredictor,
)
from crpred.utility.config import PredictorParams
from crpred.utility.visualization import plot_scenario

# General settings
num_steps_prediction = 50
scenario_path = Path("scenarios/DEU_Muc-3_1_T-1.xml")
sc, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
output_dir = Path(__file__).parent / f"output/{str(sc.scenario_id)}"
config = PredictorParams(num_steps_prediction=num_steps_prediction)

# Execute prediction
predictor = ConstantVelocityCurvilinearPredictor(config)
prediction = predictor.predict(sc)

# Create and Store visualization
plot_scenario(
    prediction,
    step_end=config.num_steps_prediction,
    plot_occupancies=True,
    save_plots=True,
    save_gif=True,
    path_output=output_dir.joinpath("ground_truth"),
)
```