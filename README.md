# AXA Data Challenge
This challenge aimed at developping models for an inbound call forecasting system. More precisely, the idea was to predict the number of calls for 26 different call centers.

Our methodology allowed us to score .39 on the public leaderboard.

For more details about our methodology, please read the report located in the root directory.

**Note that the zip dropped onto the Moodle platform does not enable to run code (due to size constraints). Minimum requirement to run some code is cloning the github repository, which comes with relevant data.**

## Main lessons
- Regression: due to the special form of the loss function (exponential in underestimation, linear in overestimation), what worked best for us was to come back to loss function minimization, that is using optimization algorithm like L-BFGS-B. Later, we made estimations more robust with a voting system, including previous mentionned regressions and boosting methods.
- Feature Engineering:
    - Despite a pretty large dataset, including a large range of administrative features, we ended up using only time series related to the number of calls, that is we used only previous numbers of calls to predict future numbers of calls. It means our models mainly relied on seasonality utilization.
    - Main tricks were to play with combinations of previous days and weeks.

Below is a Figure showing the kind of predictions we got (same kind of predictions apply for the remaining 25 centres).


## Requirements
- Basics of python, including numpy, scipy.optimize, pandas, ...
- XGBoost. More details at http://xgboost.readthedocs.io/en/latest/build.html

## Usage
Open a terminal
~~~
git clone https://github.com/antisrdy/axadatachallenge
cd axadatachallenge
~~~
Either one may want to look at cross-validations results. Then run:
~~~
./run_all.sh 0
~~~
Or, one may just want to get predictions. Then run:
~~~
./run_all.sh 2
~~~
To get both insights, run:
~~~
./run_all.sh 1
~~~

Once all predictions are computed (**they are already precomputed in ./data/preds_full_range on github**), run the following to build the submission file:
~~~
python ./code/submission.py
~~~
