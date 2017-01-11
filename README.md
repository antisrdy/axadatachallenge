# AXA Data Challenge
This challenge aimed at developping models for an inbound call forecasting system.

Our methodology allowed us to score .39 on the public leaderboard.

## Main lessons
- Regression: due to the special form of the loss function (exponential in underestimation, linear in overestimation), what worked best for us was to come back to loss function minimization, that is using optimization algorithm like L-BFGS-B. Plus later, make estimations more robust with an average with boosting methods.
- Feature Engineering:
    - Despite a pretty large dataset, including a large range of administrative features, we ended up using only time series related to the number of calls, that is we used only previous numbers of calls to predict future numbers of calls. It means our models mainly rely on seasonality utilization.
    - Main tricks were to play with combinations of previous days and weeks.

## Requirements
- Basics of python, including numpy, scipy.optimize, pandas, ...
- XGBoost. More details at http://xgboost.readthedocs.io/en/latest/build.html

## Usage
Open a terminal
~~~
git clone https://github.com/antisrdy/axadatachallenge
cd axadatachallenge
~~~
Either, one may look at cross-validations results. Then run:
~~~
python run_all.sh 0
~~~
Or, one may just want to get predictions. Then run:
~~~
python run_all.sh 2
~~~
For both, run:
~~~
python run_all.sh 1
~~~

