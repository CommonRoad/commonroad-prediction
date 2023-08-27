# CommonRoad-Prediction
A collection and interface for CommonRoad prediction algorithms.

## Project status
Currently implemented and tested models:
- [x] Constant Velocity Linear Predictor
- [x] Constant Velocity Curvilinear Predictor
- [x] Constant Acceleration Linear Predictor
- [x] Constant Acceleration Curvilinear Predictor
- [ ] Intelligent Driver Model (IDM) Predictor
- [ ] The Lane-Changing Model MOBIL Predictor

## Installation
It is recommended to use conda as an environment manager.

Clone the repository and install it with pip in development mode.
```shell
git clone git@gitlab.lrz.de:cps/commonroad/commonroad-prediction.git
pip install -e .
```

### Requirements
- Python >= 3.8
- CommonRoad >= 2023.2
- Imageio >= 2.31.1


## Examples
An example script with visualizations is provided in `examples/visualize_predictors.py`

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
CommonRoad-Prediction is open-source software distributed under the BSD License.
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
