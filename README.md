# AXA Data Challenge
This challenge aimed at developping models for an inbound call forecasting system.

Our methodology allowed us to score .66 (top 3) on the public leaderboard.

## Main lessons
- Regression: due to the special form of the loss function (exponential in underestimation, linear in overestimation), what worked best for us was to come back to loss function minimization, that is using optimization algorithm like L-BFGS-B
- Feature Engineering:
    - Despite a pretty large dataset, including a large range of administrative features, we ended up using only time series related to the number of calls, that is we used only the past of the feature to be predicted.
    - Main tricks we used was to play with combinations of previous days and weeks, means over the last available weeks.

## Requirements
- Basics of python, including numpy, scipy.optimize, ...
- Pandas

## Usage
Open a terminal
~~~
git clone https://github.com/antisrdy/axadatachallenge
cd axadatachallenge
~~~
To fit models as we did, and observe cross-validations results:
~~~
python run_all.py
~~~
