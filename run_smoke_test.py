"""
Smoke test using the real nfldata.csv file.

This script loads only a few pre-play columns from the large NFL dataset, filters to 4th-down pass/run attempts
with a non-null `fourth_down_converted` label, performs light cleaning / encoding, trains a logistic regression model,
and demonstrates the helper that returns a probability and a 0/1 "go" decision using a 0.45 (45%) threshold.

Run from the project root using the configured Python environment.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

R = 2025
import os
import joblib
script_dir = os.path.dirname(__file__)
csv_path = os.path.abspath(os.path.join(script_dir, '..', 'nfldata.csv'))

# Minimal set of columns to read (keeps memory footprint smaller than reading the full file)
read_cols = [
    'down','play_type','fourth_down_converted','ydstogo','yardline_100','qtr','game_seconds_remaining',
    'score_differential','shotgun','no_huddle','qb_dropback','pass_length','pass_location'
]

df0 = pd.read_csv(csv_path, nrows=0)
available = [c for c in read_cols if c in df0.columns]
print('columns available:', available)

df = pd.read_csv(csv_path, usecols=available, low_memory=True)

# Cleaning & selection: choose real 4th-down offensive attempts (pass / run) and ensure a target label exists.
df = df[(df['down'] == 4) & (df['play_type'].isin(['pass', 'run'])) & (df['fourth_down_converted'].notnull())]
df = df[df['ydstogo'].notnull() & df['yardline_100'].notnull()].copy()
df['fourth_down_converted'] = df['fourth_down_converted'].astype(int)

print('Usable rows for modeling:', len(df))

# Candidate numeric/base features
base_features = ['ydstogo', 'yardline_100', 'qtr', 'game_seconds_remaining', 'score_differential', 'shotgun','no_huddle','qb_dropback']
cat_features = [c for c in ['pass_length','pass_location'] if c in df.columns]

for col in ['shotgun','no_huddle','qb_dropback']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

for c in cat_features:
    df[c] = df[c].fillna('NA').astype(str)

X = df[base_features].copy()
if cat_features:
    X = pd.concat([X, pd.get_dummies(df[cat_features], prefix=cat_features, dummy_na=False)], axis=1)

y = df['fourth_down_converted'].astype(int)

# If a features file exists (saved by our training pipeline / selection script), prefer that ordering
features_file = os.path.join(script_dir, 'models', 'features.joblib')
if os.path.exists(features_file):
    try:
        user_features = joblib.load(features_file)
        # Ensure the feature columns requested are present in X — otherwise fall back to the full set
        present = [f for f in user_features if f in X.columns]
        if len(present) > 0:
            X = X[present].copy()
            print('Using saved features list from models/features.joblib —', len(present), 'features')
        else:
            print('Saved feature list found but none of the features are present in the current dataset; using full prepared matrix.')
    except Exception as exc:
        print('Unable to load models/features.joblib — using full feature matrix:', exc)

print('Prepared rows', len(X), 'features', X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=R, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=R, solver='liblinear', max_iter=1000)
model.fit(X_train_scaled, y_train)

def decide(play, threshold=0.45, features=X.columns.tolist()):
    """Produce probability and binary decision (0/1) given a single play mapping using the same feature layout used in training.

    Expected keys in play mapping: the base numeric keys (ydstogo, yardline_100, qtr, game_seconds_remaining, score_differential,
    shotgun, no_huddle, qb_dropback) plus optionally 'pass_length' and 'pass_location'.
    """
    play_map = dict(play)
    ordered = np.zeros(len(features), dtype=float)
    for i, fname in enumerate(features):
        if fname in play_map:
            ordered[i] = float(play_map[fname])
        elif '_' in fname:
            base, cat = fname.split('_', 1)
            if base in play_map and str(play_map[base]) == cat:
                ordered[i] = 1.0
    vec = ordered.reshape(1, -1)
    vec_scaled = scaler.transform(vec)
    p = model.predict_proba(vec_scaled)[0,1]
    return p, int(p >= threshold)

# Example: choose a plausible 4th-and-4 pass attempt in the middle of the field
example = {
    'ydstogo': 4,
    'yardline_100': 55,
    'qtr': 4,
    'game_seconds_remaining': 120,
    'score_differential': 3,
    'shotgun': 1,
    'no_huddle': 0,
    'qb_dropback': 1,
    'pass_length': 'short',
    'pass_location': 'left'
}

p, d = decide(example)
print('example probability', p, 'decision', d)
