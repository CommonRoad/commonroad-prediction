# CommonRoad-Prediction
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/commonroad-prediction.svg)](https://pypi.python.org/pypi/commonroad-prediction/)  
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)  
[![PyPI version fury.io](https://badge.fury.io/py/commonroad-prediction.svg)](https://pypi.python.org/pypi/commonroad-prediction/)
[![PyPI download month](https://img.shields.io/pypi/dm/commonroad-prediction.svg?label=PyPI%20downloads)](https://pypi.python.org/pypi/commonroad-prediction/) 
[![PyPI download week](https://img.shields.io/pypi/dw/commonroad-prediction.svg?label=PyPI%20downloads)](https://pypi.python.org/pypi/commonroad-prediction/)   
[![PyPI license](https://img.shields.io/pypi/l/commonroad-prediction.svg)](https://pypi.python.org/pypi/commonroad-prediction/)

A collection and interface for CommonRoad-based prediction algorithms.

## Project status
Currently implemented and tested models:   

- Constant Velocity Linear Predictor [1]
- Constant Velocity Curvilinear Predictor [1]
- Constant Acceleration Linear Predictor [1]
- Constant Acceleration Curvilinear Predictor [1]

In development:
- Intelligent Driver Model (IDM) Predictor [2]
- Lane-Changing Model MOBIL Predictor [3]

We highly welcome your contribution.
If you want to contribute a prediction algorithm, please create an issue/pull request in our [GitHub repository](https://github.com/commonroad/commonroad-prediction).


## Installation and Usage
We recommend to use PyCharm (Professional) as IDE.  
### Usage in other projects
We provide an PyPI package which can be installed with the following command
```shell
pip install commonroad-prediction
```

### Development
It is recommended to use [poetry](https://python-poetry.org/) as an environment manager.
Clone the repository and install it with poetry.
```shell
git clone git@github.com:commonroad/commonroad-prediction.git
poetry shell
poetry install
```

### Examples
We recommend to use PyCharm (Professional) as IDE. 
An example script for visualizing predictions is provided [here](example.md).


## Documentation
You can generate the documentation within your activated Poetry environment using.
```bash
poetry shell
mkdocs build
```
The documentation will be located under site, where you can open `index.html` in your browser to view it.
For updating the documentation you can also use the live preview:
```bash
poetry shell
mkdocs serve
```

## Authors
Responsible: Roland Stolz, Sebastian Maierhofer


## References
The implemented algorithms are based on the subsequent publications:  

[1] R. Schubert, E. Richter and G. Wanielik, 
"Comparison and evaluation of advanced motion models for vehicle tracking,"
Proc. of the IEEE Int. Conf. on Information Fusion, 2008, pp. 1-6.

[2] M. Treiber, A. Hennecke, and D. Helbing, 
"Congested traffic states in empirical observations and microscopic simulations,"
Physical Review E, vol. 62, no. 2, pp. 1805–1824, 2000.

[3] A. Kesting, M. Treiber, and D. Helbing, 
“General lane-changing model MOBIL for car-following models,” 
Transportation Research Record, vol. 1999, pp. 86–94, Jan. 2007