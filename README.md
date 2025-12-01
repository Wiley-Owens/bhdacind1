# bhdacind1


This project now uses the real `nfldata.csv` file (in the repository root) and contains a cleaned, documented pipeline in `analysis.ipynb`.

Key features:

 - Notebook (`analysis.ipynb`) contains a full cleaning pipeline for `nfldata.csv`, focusing on 4th-down offensive attempts (pass/run) and training a logistic regression classifier that returns a probability and a binary "go" decision (threshold = 0.45 / 45%).
- A smoke test script (`run_smoke_test.py`) performs the same cleaning and training steps on `nfldata.csv` and prints a worked example prediction.

How to run the smoke test (from PowerShell):

```powershell
C:/Users/owens/miniconda3/Scripts/conda.exe run -p C:\Users\owens\miniconda3 --no-capture-output python .\bhdacind1\run_smoke_test.py
```

Dashboard (interactive Plotly + Dash app)
---------------------------------------
After running the notebook cells that train & save the final model (or running the smoke-test), you can start the Dash app which loads the saved model and scaler and provides an interactive prediction UI and dataset visualizations.

1) Install dash and plotly if you don't already have them:

```powershell
C:/Users/owens/miniconda3/Scripts/conda.exe run -p C:\Users\owens\miniconda3 --no-capture-output pip install dash plotly
```

2) Run the app:

```powershell
C:/Users/owens/miniconda3/Scripts/conda.exe run -p C:\Users\owens\miniconda3 --no-capture-output python .\bhdacind1\run_dash_app.py
```

Visit http://127.0.0.1:8050 in your browser to interact with the dashboard.

Dashboard additions
-------------------
The dashboard now includes an extra visualization: a bar chart showing the model's mean predicted probability of converting a 4th-down play for each `yardline_100` value (the standard 100-based field position). This gives a quick, at-a-glance view of how conversion likelihood changes across the field.

If you want me to plug your real play-by-play dataset into the notebook and tune the model (or change the decision threshold), tell me where the CSV is and how your columns are named and I will make the updates. 
Testing
-------
We added a small unit test suite for the dashboard prediction callback that uses mocks so tests run quickly without loading the full dataset or saved model files.

Run the unit tests (PowerShell / conda-run example):

```powershell
C:/Users/owens/miniconda3/Scripts/conda.exe run -p C:/Users/owens/miniconda3 --no-capture-output python -m pytest -q bhdacind1/tests/test_dash_prediction_unit.py
```

Note about warnings
-------------------
We fixed a user-visible sklearn warning by ensuring the callback passes a pandas DataFrame with named columns to the scaler (StandardScaler) when making predictions. This keeps logs clean for both local runs and CI.


