import math
import pandas as pd

from football_prediction.model import (
    poisson_pmf,
    calculate_match_outcome_probabilities,
    predict_match_full,
    predict_match_full2,
)


class DummyModel:
    def __init__(self, prediction: float):
        self.prediction = prediction

    def predict(self, X):
        return [self.prediction]


def fake_build_prediction_row(team_model_df, home_team, away_team):
    return pd.DataFrame(
        {
            "home_team": [home_team],
            "away_team": [away_team],
        }
    )


def test_poisson_pmf_zero():
    result = poisson_pmf(0, 2.0)
    expected = math.exp(-2.0)
    assert abs(result - expected) < 1e-10


def test_calculate_match_outcome_probabilities_total_close_to_one():
    result = calculate_match_outcome_probabilities(
        home_lambda=1.5,
        away_lambda=1.2,
        max_goals=10,
    )

    assert 0.99 < result["total_probability"] <= 1.0


def test_calculate_match_outcome_probabilities_contains_score_probs():
    result = calculate_match_outcome_probabilities(
        home_lambda=1.0,
        away_lambda=1.0,
        max_goals=3,
    )

    assert (0, 0) in result["score_probabilities"]
    assert (1, 0) in result["score_probabilities"]
    assert (3, 3) in result["score_probabilities"]


def test_predict_match_full_home_win(monkeypatch):
    monkeypatch.setattr(
        "football_prediction.model.build_prediction_row",
        fake_build_prediction_row,
    )

    home_model = DummyModel(2.4)
    away_model = DummyModel(1.2)
    team_model_df = pd.DataFrame({"dummy": [1]})

    result = predict_match_full(
        home_model=home_model,
        away_model=away_model,
        team_model_df=team_model_df,
        home_team="Arsenal",
        away_team="Chelsea",
    )

    assert result["home_team"] == "Arsenal"
    assert result["away_team"] == "Chelsea"
    assert result["rounded_home_goals"] == 2
    assert result["rounded_away_goals"] == 1
    assert result["winner"] == "Arsenal"


def test_predict_match_full_draw(monkeypatch):
    monkeypatch.setattr(
        "football_prediction.model.build_prediction_row",
        fake_build_prediction_row,
    )

    home_model = DummyModel(1.4)
    away_model = DummyModel(1.4)
    team_model_df = pd.DataFrame({"dummy": [1]})

    result = predict_match_full(
        home_model=home_model,
        away_model=away_model,
        team_model_df=team_model_df,
        home_team="Liverpool",
        away_team="Chelsea",
    )

    assert result["rounded_home_goals"] == 1
    assert result["rounded_away_goals"] == 1
    assert result["winner"] == "Draw"


def test_predict_match_full2_probabilities_exist(monkeypatch):
    monkeypatch.setattr(
        "football_prediction.model.build_prediction_row",
        fake_build_prediction_row,
    )

    home_model = DummyModel(2.0)
    away_model = DummyModel(1.0)
    team_model_df = pd.DataFrame({"dummy": [1]})

    result = predict_match_full2(
        home_model=home_model,
        away_model=away_model,
        team_model_df=team_model_df,
        home_team="Arsenal",
        away_team="Chelsea",
        max_goals=10,
    )

    assert "home_win_probability" in result
    assert "draw_probability" in result
    assert "away_win_probability" in result
    assert "score_probabilities" in result

    total = (
        result["home_win_probability"]
        + result["draw_probability"]
        + result["away_win_probability"]
    )
    assert 0.99 < total <= 1.0