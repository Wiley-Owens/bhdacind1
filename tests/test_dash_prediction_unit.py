import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np


HERE = Path(__file__).resolve().parent.parent
MODULE_PATH = HERE / 'run_dash_app.py'


class _FakeScaler:
    def transform(self, X):
        # Accept DataFrame or numpy array; return numpy array
        if hasattr(X, 'values'):
            return X.values.astype(float)
        return np.asarray(X, dtype=float)


class _FakeModel:
    def __init__(self, n_features):
        self.n_features = n_features
        # provide fake coefficients and importances so the plotting branch works
        self.coef_ = np.zeros((1, n_features))
        self.feature_importances_ = np.zeros(n_features)

    def predict_proba(self, X):
        n = X.shape[0]
        # return predictable probabilities (second column is P(class=1))
        return np.vstack([np.ones(n) * 0.4, np.ones(n) * 0.6]).T


def _make_mock_read_csv(*args, **kwargs):
    cols = ['down','play_type','fourth_down_converted','ydstogo','yardline_100','qtr','game_seconds_remaining','score_differential','shotgun','no_huddle','qb_dropback','pass_length','pass_location']
    data = [
        [4, 'pass', 1, 4, 55, 4, 120, 3, 1, 0, 1, 'short', 'left'],
        [4, 'run', 0, 3, 60, 4, 90, -1, 0, 0, 0, None, None],
        [4, 'pass', 1, 2, 30, 2, 200, 7, 1, 1, 1, 'deep', 'right']
    ]
    df = pd.DataFrame(data, columns=cols)
    # return 0-rows when nrows=0 requested
    if kwargs.get('nrows', None) == 0:
        return df.head(0)
    if 'usecols' in kwargs and kwargs['usecols'] is not None:
        return df.loc[:, [c for c in kwargs['usecols'] if c in df.columns]]
    return df


def _import_with_mocks(monkeypatch):
    import joblib

    # monkeypatch pandas.read_csv before import so module-level reads use the small DataFrame
    monkeypatch.setattr('pandas.read_csv', lambda *a, **k: _make_mock_read_csv(*a, **k))

    # Patch joblib.load so the module gets fake model + scaler without needing files
    real_load = joblib.load

    def fake_load(path):
        p = str(path)
        if 'final_model' in p:
            return _FakeModel(n_features=14)
        elif 'scaler' in p:
            return _FakeScaler()
        return real_load(path)

    monkeypatch.setattr('joblib.load', fake_load)

    spec = importlib.util.spec_from_file_location('dashmod', str(MODULE_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_click_returns_empty(monkeypatch):
    mod = _import_with_mocks(monkeypatch)
    out = mod.make_prediction(0, 4, 55, 4, 120, 3, 1, 0, 1, 'short', 'left')
    assert out == ('', {})


def test_valid_prediction_returns_tuple(monkeypatch):
    mod = _import_with_mocks(monkeypatch)
    txt, fig = mod.make_prediction(1, 4, 55, 4, 120, 3, 1, 0, 1, 'short', 'left')
    assert isinstance(txt, str)
    assert 'Predicted probability' in txt
    assert fig is not None


def test_missing_values_handled(monkeypatch):
    mod = _import_with_mocks(monkeypatch)
    txt, fig = mod.make_prediction(1, None, None, None, None, None, None, None, None, None, None)
    assert isinstance(txt, str)
    assert ('Error computing prediction' not in txt)
