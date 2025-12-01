"""
select_features.py

Compute statistically significant or otherwise-important features for the project
and save the final feature ordering to models/features.joblib so the training, smoke
tests, and dashboard all use the same canonical feature set.

Behavior:
 - Attempts to use statsmodels.Logit to compute p-values and selects features with p < 0.05
 - If statsmodels is not available (or selection fails), falls back to picking top-N features
   by absolute coefficient from a LogisticRegression fit

Output: models/features.joblib (list of feature names)

Run:
  python -m bhdacind1.select_features

"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def build_feature_matrix(df):
    base_features = ['ydstogo', 'yardline_100', 'qtr', 'game_seconds_remaining', 'score_differential', 'shotgun', 'no_huddle', 'qb_dropback']
    cat_features = [c for c in ['pass_length', 'pass_location'] if c in df.columns]

    for col in ['shotgun', 'no_huddle', 'qb_dropback']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    for c in cat_features:
        df[c] = df[c].fillna('NA').astype(str)

    X = df[base_features].copy()
    if cat_features:
        X = pd.concat([X, pd.get_dummies(df[cat_features], prefix=cat_features, dummy_na=False)], axis=1)

    return X


def main(csv_path='nfldata.csv', p_thresh=0.05, top_k_fallback=10):
    repo_root = os.path.dirname(os.path.dirname(__file__))
    csv_abspath = os.path.join(repo_root, csv_path)

    # Validate CSV
    if not os.path.exists(csv_abspath):
        raise FileNotFoundError(f"CSV not found at {csv_abspath}")

    # Load dataset subset
    read_cols = ['down','play_type','fourth_down_converted','ydstogo','yardline_100','qtr','game_seconds_remaining','score_differential','shotgun','no_huddle','qb_dropback','pass_length','pass_location']
    df0 = pd.read_csv(csv_abspath, nrows=0)
    available = [c for c in read_cols if c in df0.columns]

    print('Available columns:', available)

    df = pd.read_csv(csv_abspath, usecols=available, low_memory=True)
    df = df[(df['down'] == 4) & (df['play_type'].isin(['pass', 'run'])) & (df['fourth_down_converted'].notnull())]
    df = df[df['ydstogo'].notnull() & df['yardline_100'].notnull()].copy()
    df['fourth_down_converted'] = df['fourth_down_converted'].astype(int)

    X = build_feature_matrix(df)
    y = df['fourth_down_converted'].astype(int)

    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    selected = None

    # Try statsmodels p-values if available
    try:
        import statsmodels.api as sm
        print('Using statsmodels Logit for p-value selection')
        Xc = sm.add_constant(X)
        logit = sm.Logit(y, Xc)
        res = logit.fit(disp=False, maxiter=200)
        pvals = res.pvalues.drop(labels='const', errors='ignore')
        sig = pvals[pvals < p_thresh]
        if len(sig) > 0:
            selected = sig.index.tolist()
            print(f'Selected {len(selected)} features by p-value < {p_thresh}:', selected)
        else:
            print('No features passed p-value threshold')
            raise RuntimeError('no significant p-values')

    except Exception as exc:
        print('P-value selection failed or not available (statsmodels missing or failed):', exc)
        print('Falling back to LogisticRegression coefficient selection')
        lr = LogisticRegression(solver='liblinear', max_iter=2000)
        lr.fit(X.fillna(0), y)
        coefs = np.abs(lr.coef_.ravel())
        df_coefs = pd.DataFrame({'feature': X.columns, 'abs_coef': coefs}).sort_values('abs_coef', ascending=False).head(top_k_fallback)
        selected = df_coefs['feature'].tolist()
        print(f'Selected top {len(selected)} features by coefficient magnitude:', selected)

    features_path = os.path.join(models_dir, 'features.joblib')
    joblib.dump(selected, features_path)
    print('Saved features list to', features_path)


if __name__ == '__main__':
    main()
