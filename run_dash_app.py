"""
Minimal Plotly Dash dashboard to interact with the trained 4th-down model.

Features:
    shotgun, no_huddle, qb_dropback, pass_length, pass_location) and show probability + binary decision (threshold=0.45).

Run (PowerShell):
    # Use forward slashes or escape backslashes when running in PowerShell
    C:/Users/owens/miniconda3/Scripts/conda.exe run -p C:/Users/owens/miniconda3 --no-capture-output pip install dash plotly
    C:/Users/owens/miniconda3/Scripts/conda.exe run -p C:/Users/owens/miniconda3 --no-capture-output python ./bhdacind1/run_dash_app.py

"""
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / 'models' / 'final_model.joblib'
SCALER_PATH = BASE / 'models' / 'scaler.joblib'
DATA_PATH = BASE.parent / 'nfldata.csv'

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise RuntimeError('Model or scaler not found. Please run the notebook training cells to create model files in models/ before starting the dashboard.')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load and prepare a small cleaned dataset for visualizations (same filters as notebook)
cols = ['down','play_type','fourth_down_converted','ydstogo','yardline_100','qtr','game_seconds_remaining','score_differential','shotgun','no_huddle','qb_dropback','pass_length','pass_location']
df0 = pd.read_csv(DATA_PATH, nrows=0)
available = [c for c in cols if c in df0.columns]
df = pd.read_csv(DATA_PATH, usecols=available, low_memory=True)
df = df[(df['down']==4) & (df['play_type'].isin(['pass','run'])) & (df['fourth_down_converted'].notnull())]
df = df[df['ydstogo'].notnull() & df['yardline_100'].notnull()].copy()

for col in ['shotgun','no_huddle','qb_dropback']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

cat_features = [c for c in ['pass_length','pass_location'] if c in df.columns]
for c in cat_features:
    df[c] = df[c].fillna('NA').astype(str)

# Build feature matrix consistent with training
base_features = ['ydstogo','yardline_100','qtr','game_seconds_remaining','score_differential','shotgun','no_huddle','qb_dropback']
X_vis = df[base_features].copy()
if cat_features:
    X_vis = pd.concat([X_vis, pd.get_dummies(df[cat_features], prefix=cat_features, dummy_na=False)], axis=1)

# Align X_vis to the feature ordering the saved scaler/model expects.
# Prefer the scaler's known feature names when available, otherwise fall back to the
# saved models/features.joblib list (if present) or the derived columns from the dataset.
expected_features = None
try:
    # StandardScaler and many sklearn estimators expose `feature_names_in_` after fit
    expected_features = list(scaler.feature_names_in_)
except Exception:
    expected_features = None

features_file = BASE / 'models' / 'features.joblib'
if expected_features is not None:
    # Ensure every expected feature exists in X_vis; if missing, add a zero column so transform won't error
    for f in expected_features:
        if f not in X_vis.columns:
            X_vis[f] = 0
    # Reorder to the expected feature ordering and fill any remaining NA values
    X_vis = X_vis[expected_features].fillna(0)
    feature_cols = expected_features
    print('Aligned dashboard input to scaler.feature_names_in_ (used for model/scaler).')
else:
    # No explicit expected feature ordering from scaler; fall back to saved file or derived columns
    if features_file.exists():
        try:
            saved_features = joblib.load(features_file)
            present = [f for f in saved_features if f in X_vis.columns]
            if len(present) > 0:
                feature_cols = present
                X_vis = X_vis[feature_cols].copy()
                print('Using saved feature list from models/features.joblib')
            else:
                feature_cols = X_vis.columns.tolist()
                X_vis = X_vis.fillna(0)
                print('Saved feature list found but features missing in dataset; using derived features instead')
        except Exception:
            feature_cols = X_vis.columns.tolist()
            X_vis = X_vis.fillna(0)
            print('Unable to load models/features.joblib; using derived features')
    else:
        feature_cols = X_vis.columns.tolist()
        X_vis = X_vis.fillna(0)

# compute probabilities using the saved scaler + model
X_scaled_vis = scaler.transform(X_vis)
probs_vis = model.predict_proba(X_scaled_vis)[:,1]
df['pred_prob'] = probs_vis

# Friendly metadata descriptions for the most relevant input variables.
# We generate a short, user-friendly explanation for each input the dashboard accepts.
# If your CSV contains more detailed metadata fields we could use them instead; for now
# these are concise human-readable descriptions derived from the dataset column names.
variable_descriptions = {
    'ydstogo': 'Yards to go — how many yards are needed to gain a new first down.',
    'yardline_100': 'Yardline (100-based) — distance from the opponent\'s endzone (1=close to opponent, 99=near own endzone).',
    'qtr': 'Quarter of the game (1-4).',
    'game_seconds_remaining': 'Seconds remaining in the game (total seconds until end).',
    'score_differential': "Offense score minus opponent's score (positive means the offense is leading).",
    'shotgun': 'Shotgun formation indicator (1 if the offense is in shotgun, 0 otherwise).',
    'no_huddle': 'No-huddle offense indicator (1 if offense used no-huddle).',
    'qb_dropback': 'Quarterback dropback indicator (1 if QB dropback occurred).',
    'pass_length': 'Pass length (categorical): short / deep / NA for run plays.',
    'pass_location': 'Pass target location (categorical): left / middle / right / NA.'
}

# --- Aggregate by game_seconds_remaining and compute mean predicted probability ---
# We aggregate on game_seconds_remaining and compute average predicted probability for each time bucket.
if 'game_seconds_remaining' in df.columns:
    df_by_time = df.groupby('game_seconds_remaining', as_index=False)['pred_prob'].mean()
    # Sort ascending by time for natural visualization (low time to high time)
    df_by_time = df_by_time.sort_values('game_seconds_remaining')
else:
    df_by_time = None

# Add a simple smoothing column using pandas rolling window so we do not require statsmodels
if df_by_time is not None and 'pred_prob' in df_by_time.columns:
    # Use a small centered rolling window to smooth the mean probabilities for plotting
    df_by_time['smoothed'] = df_by_time['pred_prob'].rolling(window=5, min_periods=1, center=True).mean()
else:
    # Ensure variable exists even when data is missing
    df_by_time = None

# --- Also aggregate by yardline for secondary visualization ---
if 'yardline_100' in df.columns:
    df_by_yardline = df.groupby('yardline_100', as_index=False)['pred_prob'].mean()
    # Sort ascending so viewers can read from opponent's endzone outward (1 = closest to opponent endzone)
    df_by_yardline = df_by_yardline.sort_values('yardline_100')
else:
    df_by_yardline = None

# Prepare a figure for time-vs-mean-probability that does not require statsmodels
if df_by_time is not None:
    try:
        # Plot the smoothed mean probability as a single clean line (do not show raw points)
        # We use px.line on the smoothed column — this removes markers and avoids statsmodels.
        if 'smoothed' in df_by_time.columns:
            fig_by_time = px.line(df_by_time, x='game_seconds_remaining', y='smoothed', labels={'game_seconds_remaining': 'Seconds Remaining', 'smoothed': 'Mean Predicted Probability'}, title='Mean predicted probability by game time')
        else:
            # fallback: plot the raw mean values as a line if smoothing wasn't applied
            fig_by_time = px.line(df_by_time, x='game_seconds_remaining', y='pred_prob', labels={'game_seconds_remaining': 'Seconds Remaining', 'pred_prob': 'Mean Predicted Probability'}, title='Mean predicted probability by game time')

        # Flip the x-axis so chart reads from most time remaining -> least time remaining (left -> right)
        fig_by_time.update_xaxes(autorange='reversed')

        # Add vertical lines to mark quarter boundaries at 2700, 1800, 900 seconds remaining.
        quarter_lines = []
        for seconds in (2700, 1800, 900):
            quarter_lines.append(dict(
                type='line', x0=seconds, x1=seconds, xref='x', yref='paper', y0=0, y1=1,
                line=dict(color='gold', width=1, dash='dash')
            ))
        # Add them to the layout shapes and ensure y-axis fits [0,1]
        fig_by_time.update_layout(shapes=quarter_lines)
        fig_by_time.update_yaxes(range=[0, 1])
    except Exception:
        # if building the plot fails for any reason, fall back to an empty figure
        fig_by_time = {}
else:
    fig_by_time = {}

def create_app():
    """Create and return a Dash app instance.

    We import Dash lazily here so the module can be imported without Dash being
    installed (avoids editor/CI import warnings). The app uses the module-level
    model/scaler/data variables that are loaded when this module imports.
    """
    from dash import Dash, html, dcc, Input, Output, State  # type: ignore[reportMissingImports]

    app = Dash(__name__)

    app.layout = html.Div([
        # Header
        html.Div([
            html.H1('4th-down: Go-for-it decision dashboard', style={'margin': '8px 0 4px 0'}),
            html.P('Interactive explorer: give a single-play input and get a probability + a go/no-go decision (threshold = 0.45 / 45%).', style={'margin': 0, 'color': '#666'})
        ], style={'padding': '12px 16px', 'borderBottom': '1px solid #eee', 'backgroundColor': '#f7fbff'}),

        # Body: left = controls / help, right = charts
        html.Div([
            html.Div([
                html.H3('Play inputs', style={'marginTop': 0}),
                html.Div([
                    html.Label('Yards to go'),
                    dcc.Input(id='ydstogo', type='number', value=4, style={'width': '100%'}),
                ], style={'marginBottom': '8px'}),

                html.Div([
                    html.Label('Yardline (100-based)'),
                    dcc.Input(id='yardline_100', type='number', value=55, style={'width': '100%'}),
                ], style={'marginBottom': '8px'}),

                html.Div([
                    html.Label('Quarter'),
                    dcc.Input(id='qtr', type='number', value=4, style={'width': '100%'}),
                ], style={'marginBottom': '8px'}),

                html.Div([
                    html.Label('Game seconds remaining'),
                    dcc.Input(id='game_seconds_remaining', type='number', value=120, style={'width': '100%'}),
                ], style={'marginBottom': '8px'}),

                html.Div([
                    html.Label('Score differential'),
                    dcc.Input(id='score_differential', type='number', value=3, style={'width': '100%'}),
                ], style={'marginBottom': '8px'}),

                html.Div(style={'display': 'flex', 'gap': '8px', 'marginTop': '6px'}, children=[
                    html.Div([html.Label('Shotgun    (0/1)'), dcc.Input(id='shotgun', type='number', value=1, style={'width': '100%'} )], style={'flex': '1'}),
                    html.Div([html.Label('No-huddle (0/1)'), dcc.Input(id='no_huddle', type='number', value=0, style={'width': '100%'})], style={'flex': '1'}),
                    html.Div([html.Label('QB dropback (0/1)'), dcc.Input(id='qb_dropback', type='number', value=1, style={'width': '100%'})], style={'flex': '1'})
                ]),

                html.Div([html.Label('Pass length'), dcc.Dropdown(id='pass_length', options=[{'label': v, 'value': v} for v in sorted(df['pass_length'].dropna().unique())], value='short')], style={'marginTop': '10px'}),
                html.Div([html.Label('Pass location'), dcc.Dropdown(id='pass_location', options=[{'label': v, 'value': v} for v in sorted(df['pass_location'].dropna().unique())], value='left')], style={'marginTop': '8px'}),

                html.Button('Predict', id='predict_button', n_clicks=0, style={'marginTop': '12px', 'backgroundColor': '#0370d6', 'color': 'white', 'border': 'none', 'padding': '8px 12px', 'borderRadius': '4px'}),
                html.Div(id='prediction_output', style={'marginTop': '10px', 'fontWeight': '600'}),

                # Help / variable definitions in a collapsible details block
                html.Details([html.Summary('Input descriptions / help', style={'cursor': 'pointer'}),
                              html.Ul([html.Li([html.B(k), ': ', variable_descriptions.get(k, '')]) for k in ['ydstogo','yardline_100','qtr','game_seconds_remaining','score_differential','shotgun','no_huddle','qb_dropback','pass_length','pass_location']])], style={'marginTop': '14px', 'fontSize': '14px'})

            ], style={'width': '32%', 'padding': '18px', 'backgroundColor': '#ffffff', 'borderRadius': '8px', 'boxShadow': '0 2px 6px rgba(20,20,20,0.07)'}),

            html.Div([
                dcc.Graph(id='prob_by_time', figure=fig_by_time, style={'height': '330px'}),
                dcc.Graph(id='prob_by_yardline', figure=(px.bar(df_by_yardline, x='yardline_100', y='pred_prob', labels={'yardline_100': 'Yardline (100-based)', 'pred_prob': 'Mean predicted probability'}, title='Mean predicted probability by yardline') if df_by_yardline is not None else {}), style={'height': '320px'}),
                dcc.Graph(id='feature_coefs', style={'height': '420px'})
            ], style={'flex': '1'})

        ], style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start', 'padding': '18px'}),

        # Footer / small note
        html.Div('Model probabilities are generated by a trained classifier and converted to a go/no-go decision at threshold 0.45 (45%). Use the inputs on the left to experiment.', style={'padding': '8px 16px', 'color': '#555', 'fontSize': '13px'})

    ], style={'fontFamily': 'Inter, Arial, sans-serif', 'maxWidth': '1200px', 'margin': '12px auto', 'backgroundColor': '#fafcff'})

    # Register the callback with the app (we reference the module-level make_prediction function)
    app.callback(
        Output('prediction_output', 'children'),
        Output('feature_coefs', 'figure'),
        Input('predict_button', 'n_clicks'),
        State('ydstogo', 'value'),
        State('yardline_100', 'value'),
        State('qtr', 'value'),
        State('game_seconds_remaining', 'value'),
        State('score_differential', 'value'),
        State('shotgun', 'value'),
        State('no_huddle', 'value'),
        State('qb_dropback', 'value'),
        State('pass_length', 'value'),
        State('pass_location', 'value')
    )(make_prediction)

    return app


def make_prediction(n_clicks, ydstogo, yardline_100, qtr, game_seconds_remaining, score_differential, shotgun, no_huddle, qb_dropback, pass_length, pass_location):
    # Local imports for detailed error output only when this callback executes
    import traceback

    # If user hasn't clicked predict yet, return empty outputs (no prediction)
    if n_clicks == 0:
        return '', {}

    try:
        play = {
            'ydstogo': ydstogo,
            'yardline_100': yardline_100,
            'qtr': qtr,
            'game_seconds_remaining': game_seconds_remaining,
            'score_differential': score_differential,
            'shotgun': shotgun,
            'no_huddle': no_huddle,
            'qb_dropback': qb_dropback,
            'pass_length': pass_length,
            'pass_location': pass_location
        }

        # Fill missing numeric values with reasonable defaults (user may leave inputs empty)
        numeric_defaults = {
            'ydstogo': 0,
            'yardline_100': 50,
            'qtr': 1,
            'game_seconds_remaining': 900,
            'score_differential': 0,
            'shotgun': 0,
            'no_huddle': 0,
            'qb_dropback': 0
        }
        for k, default in numeric_defaults.items():
            if play.get(k) is None:
                play[k] = default

        # Ensure categorical empties become 'NA' strings so one-hot mapping works
        if not play.get('pass_length'):
            play['pass_length'] = 'NA'
        if not play.get('pass_location'):
            play['pass_location'] = 'NA'

        # Build an ordered numeric vector matching the feature order used at training
        features = feature_cols
        ordered = np.zeros(len(features), dtype=float)

        # Populate numeric and one-hot indicator features from the single-play dict
        for i, f in enumerate(features):
            # Direct numeric features: exact key match
            if f in play and play[f] is not None:
                try:
                    ordered[i] = float(play[f])
                except Exception:
                    # If a numeric conversion fails, leave as zero
                    ordered[i] = 0.0
            # One-hot dummies created during training will look like 'pass_length_short'
            elif '_' in f:
                base, cat = f.split('_', 1)
                if base in play and str(play[base]) == cat:
                    ordered[i] = 1.0

        # Scale / predict. We expect scaler and model to accept the same feature ordering
        # Create a DataFrame that preserves the same feature names/order the scaler
        # and model were trained with. This prevents sklearn warnings about missing
        # feature names and is safer for scalers trained with named features.
        X_raw = pd.DataFrame([ordered], columns=features)
        X_scaled = scaler.transform(X_raw)
        p = float(model.predict_proba(X_scaled)[0, 1])
        d = int(p >= 0.45)

        # Build a small visualization of the most-important features so users can inspect
        if hasattr(model, 'coef_'):
            coefs = model.coef_.ravel()
            df_coef = pd.DataFrame({'feature': features, 'coef': coefs}).sort_values('coef')
            fig = px.bar(df_coef.tail(20), x='coef', y='feature', orientation='h', title='Top positive coefficients')
        elif hasattr(model, 'feature_importances_'):
            df_coef = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(20)
            fig = px.bar(df_coef, x='importance', y='feature', orientation='h', title='Top feature importances')
        else:
            fig = {}

        txt = f'Predicted probability = {p:.3f} - Decision = {d} (1=GO, 0=NO GO)'
        return txt, fig

    except Exception as err:
        # Print full traceback to the server console for debugging and return a friendly message to user
        traceback.print_exc()
        return f'Error computing prediction: {err}', {}


    # end make_prediction

    # end module


if __name__ == '__main__':
    # New Dash versions use app.run instead of app.run_server
    app = create_app()
    app.run(debug=True, host='127.0.0.1', port=8050)
