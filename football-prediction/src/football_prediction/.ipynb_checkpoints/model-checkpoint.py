import pandas as pd
from sklearn.linear_model import PoissonRegressor

import math
import pandas as pd
from sklearn.linear_model import PoissonRegressor



def get_feature_columns() -> list[str]:
    return [
        "team_avg_goals_scored_last_5",
        "opp_avg_goals_scored_last_5",
        "team_avg_points_last_5",
        "opp_avg_points_last_5",
    ]


def prepare_training_data(team_model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = get_feature_columns()
    target_col = "goals"

    model_df = team_model_df.dropna(subset=feature_cols + [target_col]).copy()

    X = model_df[feature_cols].copy()
    y = model_df[target_col].copy()

    return X, y


def train_poisson_model(
    team_model_df: pd.DataFrame,
    alpha: float = 1.0,
    max_iter: int = 1000,
) -> tuple[PoissonRegressor, pd.DataFrame, pd.Series]:
    X, y = prepare_training_data(team_model_df)

    model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
    model.fit(X, y)

    return model, X, y


def _get_latest_team_row(
    team_model_df: pd.DataFrame,
    team_name: str,
    is_home: int,
) -> pd.Series:
    team_rows = team_model_df[
        (team_model_df["team"] == team_name) &
        (team_model_df["is_home"] == is_home)
    ].copy()

    team_rows = team_rows.sort_values("date")

    if team_rows.empty:
        raise ValueError(f"Keine Daten für Team '{team_name}' mit is_home={is_home} gefunden.")

    latest_row = team_rows.iloc[-1]
    return latest_row


def build_prediction_row(
    team_model_df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    feature_cols = get_feature_columns()

    home_row = _get_latest_team_row(team_model_df, home_team, is_home=1)
    away_row = _get_latest_team_row(team_model_df, away_team, is_home=0)

    row = {
        "team_avg_goals_scored_last_5": home_row["team_avg_goals_scored_last_5"],
        "opp_avg_goals_scored_last_5": away_row["team_avg_goals_scored_last_5"],
        "team_avg_points_last_5": home_row["team_avg_points_last_5"],
        "opp_avg_points_last_5": away_row["team_avg_points_last_5"],
    }

    prediction_df = pd.DataFrame([row])
    prediction_df = prediction_df[feature_cols]

    if prediction_df.isna().any().any():
        raise ValueError(
            f"Vorhersage für '{home_team}' gegen '{away_team}' nicht möglich, "
            "weil benötigte Feature-Werte fehlen."
        )

    return prediction_df


def predict_match(
    model,
    team_model_df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> dict:
    match_features = build_prediction_row(
        team_model_df=team_model_df,
        home_team=home_team,
        away_team=away_team,
    )

    predicted_goals = float(model.predict(match_features)[0])
    rounded_goals = int(round(predicted_goals))

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_home_goals": predicted_goals,
        "rounded_home_goals": rounded_goals,
    }

def predict_match_full(
    home_model,
    away_model,
    team_model_df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> dict:

    home_features = build_prediction_row(
        team_model_df=team_model_df,
        home_team=home_team,
        away_team=away_team,
    )

    away_features = build_prediction_row(
        team_model_df=team_model_df,
        home_team=away_team,
        away_team=home_team,
    )

    home_goals = float(home_model.predict(home_features)[0])
    away_goals = float(away_model.predict(away_features)[0])

    home_goals_rounded = int(round(home_goals))
    away_goals_rounded = int(round(away_goals))

    if home_goals_rounded > away_goals_rounded:
        winner = home_team
    elif away_goals_rounded > home_goals_rounded:
        winner = away_team
    else:
        winner = "Draw"

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_home_goals": home_goals,
        "predicted_away_goals": away_goals,
        "rounded_home_goals": home_goals_rounded,
        "rounded_away_goals": away_goals_rounded,
        "winner": winner,
    }

def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

    

def predict_match_full2(
    home_model,
    away_model,
    team_model_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    max_goals: int = 10,
) -> dict:
    home_features = build_prediction_row(
        team_model_df=team_model_df,
        home_team=home_team,
        away_team=away_team,
    )

    away_features = build_prediction_row(
        team_model_df=team_model_df,
        home_team=away_team,
        away_team=home_team,
    )

    home_goals = float(home_model.predict(home_features)[0])
    away_goals = float(away_model.predict(away_features)[0])

    home_goals_rounded = int(round(home_goals))
    away_goals_rounded = int(round(away_goals))

    if home_goals_rounded > away_goals_rounded:
        winner = home_team
    elif away_goals_rounded > home_goals_rounded:
        winner = away_team
    else:
        winner = "Draw"

    outcome_probs = calculate_match_outcome_probabilities(
        home_lambda=home_goals,
        away_lambda=away_goals,
        max_goals=max_goals,
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_home_goals": home_goals,
        "predicted_away_goals": away_goals,
        "rounded_home_goals": home_goals_rounded,
        "rounded_away_goals": away_goals_rounded,
        "winner": winner,
        "home_win_probability": outcome_probs["home_win_probability"],
        "draw_probability": outcome_probs["draw_probability"],
        "away_win_probability": outcome_probs["away_win_probability"],
        "score_probabilities": outcome_probs["score_probabilities"],
    }



def calculate_match_outcome_probabilities(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = 10,
) -> dict:
    home_goal_probs = [poisson_pmf(k, home_lambda) for k in range(max_goals + 1)]
    away_goal_probs = [poisson_pmf(k, away_lambda) for k in range(max_goals + 1)]

    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0

    score_probs = {}

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = home_goal_probs[home_goals] * away_goal_probs[away_goals]
            score_probs[(home_goals, away_goals)] = prob

            if home_goals > away_goals:
                home_win_prob += prob
            elif home_goals == away_goals:
                draw_prob += prob
            else:
                away_win_prob += prob

    total_prob = home_win_prob + draw_prob + away_win_prob

    return {
        "home_win_probability": home_win_prob,
        "draw_probability": draw_prob,
        "away_win_probability": away_win_prob,
        "total_probability": total_prob,
        "score_probabilities": score_probs,
    }