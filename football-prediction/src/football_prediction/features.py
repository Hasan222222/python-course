import pandas as pd


def prepare_features_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def create_team_match_table(df: pd.DataFrame) -> pd.DataFrame:
    home_df = df[["date", "home_team", "away_team", "home_goals", "away_goals", "result"]].copy()
    home_df["team"] = home_df["home_team"]
    home_df["opponent"] = home_df["away_team"]
    home_df["goals_scored"] = home_df["home_goals"]
    home_df["goals_conceded"] = home_df["away_goals"]
    home_df["is_home"] = 1
    home_df["points"] = home_df["result"].map({"H": 3, "D": 1, "A": 0})

    away_df = df[["date", "home_team", "away_team", "home_goals", "away_goals", "result"]].copy()
    away_df["team"] = away_df["away_team"]
    away_df["opponent"] = away_df["home_team"]
    away_df["goals_scored"] = away_df["away_goals"]
    away_df["goals_conceded"] = away_df["home_goals"]
    away_df["is_home"] = 0
    away_df["points"] = away_df["result"].map({"H": 0, "D": 1, "A": 3})

    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df = team_df[["date", "team", "opponent", "is_home", "goals_scored", "goals_conceded", "points"]]
    team_df = team_df.sort_values(["team", "date"]).reset_index(drop=True)
    return team_df


def add_rolling_team_features(team_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    team_df = team_df.copy()

    team_df["avg_goals_scored_last_5"] = (
        team_df.groupby("team")["goals_scored"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    team_df["avg_goals_conceded_last_5"] = (
        team_df.groupby("team")["goals_conceded"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    team_df["avg_points_last_5"] = (
        team_df.groupby("team")["points"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    return team_df


def merge_team_features_into_matches(df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    home_features = team_df[["date", "team", "avg_goals_scored_last_5", "avg_goals_conceded_last_5", "avg_points_last_5"]].copy()
    home_features = home_features.rename(columns={
        "team": "home_team",
        "avg_goals_scored_last_5": "home_avg_goals_scored_last_5",
        "avg_goals_conceded_last_5": "home_avg_goals_conceded_last_5",
        "avg_points_last_5": "home_avg_points_last_5"
    })

    away_features = home_features.rename(columns={
        "home_team": "away_team",
        "home_avg_goals_scored_last_5": "away_avg_goals_scored_last_5",
        "home_avg_goals_conceded_last_5": "away_avg_goals_conceded_last_5",
        "home_avg_points_last_5": "away_avg_points_last_5"
    })

    df = df.merge(home_features, on=["date", "home_team"], how="left")
    df = df.merge(away_features, on=["date", "away_team"], how="left")

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_features_data(df)
    team_df = create_team_match_table(df)
    team_df = add_rolling_team_features(team_df, window=5)
    df = merge_team_features_into_matches(df, team_df)
    return df

