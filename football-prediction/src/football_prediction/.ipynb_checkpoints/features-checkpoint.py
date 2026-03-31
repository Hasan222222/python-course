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
    team_df["goal_diff"] = team_df["goals_scored"] - team_df["goals_conceded"]

    team_df = team_df[[
        "date",
        "team",
        "opponent",
        "is_home",
        "goals_scored",
        "goals_conceded",
        "points",
        "goal_diff",
    ]]

    team_df = team_df.sort_values(["team", "date"]).reset_index(drop=True)
    return team_df


def add_rolling_team_features(team_df: pd.DataFrame, windows: list[int] = [3, 5, 10]) -> pd.DataFrame:
    team_df = team_df.copy()

    for window in windows:
        team_df[f"avg_goals_scored_last_{window}"] = (
            team_df.groupby("team")["goals_scored"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        team_df[f"avg_goals_conceded_last_{window}"] = (
            team_df.groupby("team")["goals_conceded"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        team_df[f"avg_points_last_{window}"] = (
            team_df.groupby("team")["points"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        team_df[f"avg_goal_diff_last_{window}"] = (
            team_df.groupby("team")["goal_diff"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    return team_df


def merge_team_features_into_matches(
    df: pd.DataFrame,
    team_df: pd.DataFrame,
    windows: list[int] = [3, 5, 10],
) -> pd.DataFrame:
    df = df.copy()

    home_cols = ["date", "team"]
    away_cols = ["date", "team"]

    for window in windows:
        home_cols.extend([
            f"avg_goals_scored_last_{window}",
            f"avg_goals_conceded_last_{window}",
            f"avg_points_last_{window}",
            f"avg_goal_diff_last_{window}",
        ])
        away_cols.extend([
            f"avg_goals_scored_last_{window}",
            f"avg_goals_conceded_last_{window}",
            f"avg_points_last_{window}",
            f"avg_goal_diff_last_{window}",
        ])

    home_features = team_df[home_cols].copy()
    home_features = home_features.rename(columns={
        "team": "home_team",
        **{
            f"avg_goals_scored_last_{window}": f"home_avg_goals_scored_last_{window}"
            for window in windows
        },
        **{
            f"avg_goals_conceded_last_{window}": f"home_avg_goals_conceded_last_{window}"
            for window in windows
        },
        **{
            f"avg_points_last_{window}": f"home_avg_points_last_{window}"
            for window in windows
        },
        **{
            f"avg_goal_diff_last_{window}": f"home_avg_goal_diff_last_{window}"
            for window in windows
        },
    })

    away_features = team_df[away_cols].copy()
    away_features = away_features.rename(columns={
        "team": "away_team",
        **{
            f"avg_goals_scored_last_{window}": f"away_avg_goals_scored_last_{window}"
            for window in windows
        },
        **{
            f"avg_goals_conceded_last_{window}": f"away_avg_goals_conceded_last_{window}"
            for window in windows
        },
        **{
            f"avg_points_last_{window}": f"away_avg_points_last_{window}"
            for window in windows
        },
        **{
            f"avg_goal_diff_last_{window}": f"away_avg_goal_diff_last_{window}"
            for window in windows
        },
    })

    df = df.merge(home_features, on=["date", "home_team"], how="left")
    df = df.merge(away_features, on=["date", "away_team"], how="left")

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_features_data(df)
    team_df = create_team_match_table(df)
    team_df = add_rolling_team_features(team_df, windows=[3, 5, 10])
    df = merge_team_features_into_matches(df, team_df, windows=[3, 5, 10])
    df = add_home_away_form_features(df, windows=[5])
    return df


def add_home_away_form_features(
    df: pd.DataFrame,
    windows: list[int] = [5],
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    home_df = df[["date", "home_team", "home_goals", "away_goals", "result"]].copy()
    home_df["team"] = home_df["home_team"]
    home_df["goals_scored"] = home_df["home_goals"]
    home_df["goals_conceded"] = home_df["away_goals"]
    home_df["points"] = home_df["result"].map({"H": 3, "D": 1, "A": 0})

    away_df = df[["date", "away_team", "home_goals", "away_goals", "result"]].copy()
    away_df["team"] = away_df["away_team"]
    away_df["goals_scored"] = away_df["away_goals"]
    away_df["goals_conceded"] = away_df["home_goals"]
    away_df["points"] = away_df["result"].map({"H": 0, "D": 1, "A": 3})

    home_form = home_df[["date", "team", "goals_scored", "goals_conceded", "points"]].copy()
    away_form = away_df[["date", "team", "goals_scored", "goals_conceded", "points"]].copy()

    for window in windows:
        home_form[f"avg_home_goals_scored_last_{window}"] = (
            home_form.groupby("team")["goals_scored"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        home_form[f"avg_home_goals_conceded_last_{window}"] = (
            home_form.groupby("team")["goals_conceded"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        home_form[f"avg_home_points_last_{window}"] = (
            home_form.groupby("team")["points"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        away_form[f"avg_away_goals_scored_last_{window}"] = (
            away_form.groupby("team")["goals_scored"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        away_form[f"avg_away_goals_conceded_last_{window}"] = (
            away_form.groupby("team")["goals_conceded"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        away_form[f"avg_away_points_last_{window}"] = (
            away_form.groupby("team")["points"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    home_merge_cols = ["date", "team"] + [
        col for col in home_form.columns if col.startswith("avg_home_")
    ]
    away_merge_cols = ["date", "team"] + [
        col for col in away_form.columns if col.startswith("avg_away_")
    ]

    home_features = home_form[home_merge_cols].rename(columns={"team": "home_team"})
    away_features = away_form[away_merge_cols].rename(columns={"team": "away_team"})

    df = df.merge(home_features, on=["date", "home_team"], how="left")
    df = df.merge(away_features, on=["date", "away_team"], how="left")

    return df
def build_team_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    home_df = df[[
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "home_avg_goals_scored_last_3",
        "home_avg_goals_scored_last_5",
        "home_avg_goals_scored_last_10",
        "home_avg_goals_conceded_last_3",
        "home_avg_goals_conceded_last_5",
        "home_avg_goals_conceded_last_10",
        "home_avg_points_last_3",
        "home_avg_points_last_5",
        "home_avg_points_last_10",
        "home_avg_goal_diff_last_3",
        "home_avg_goal_diff_last_5",
        "home_avg_goal_diff_last_10",
        "away_avg_goals_scored_last_3",
        "away_avg_goals_scored_last_5",
        "away_avg_goals_scored_last_10",
        "away_avg_goals_conceded_last_3",
        "away_avg_goals_conceded_last_5",
        "away_avg_goals_conceded_last_10",
        "away_avg_points_last_3",
        "away_avg_points_last_5",
        "away_avg_points_last_10",
        "away_avg_goal_diff_last_3",
        "away_avg_goal_diff_last_5",
        "away_avg_goal_diff_last_10",
        "avg_home_goals_scored_last_5",
        "avg_home_goals_conceded_last_5",
        "avg_home_points_last_5",
    ]].copy()

    home_df = home_df.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_goals": "goals",
        "home_avg_goals_scored_last_3": "team_avg_goals_scored_last_3",
        "home_avg_goals_scored_last_5": "team_avg_goals_scored_last_5",
        "home_avg_goals_scored_last_10": "team_avg_goals_scored_last_10",
        "home_avg_goals_conceded_last_3": "team_avg_goals_conceded_last_3",
        "home_avg_goals_conceded_last_5": "team_avg_goals_conceded_last_5",
        "home_avg_goals_conceded_last_10": "team_avg_goals_conceded_last_10",
        "home_avg_points_last_3": "team_avg_points_last_3",
        "home_avg_points_last_5": "team_avg_points_last_5",
        "home_avg_points_last_10": "team_avg_points_last_10",
        "home_avg_goal_diff_last_3": "team_avg_goal_diff_last_3",
        "home_avg_goal_diff_last_5": "team_avg_goal_diff_last_5",
        "home_avg_goal_diff_last_10": "team_avg_goal_diff_last_10",
        "away_avg_goals_scored_last_3": "opp_avg_goals_scored_last_3",
        "away_avg_goals_scored_last_5": "opp_avg_goals_scored_last_5",
        "away_avg_goals_scored_last_10": "opp_avg_goals_scored_last_10",
        "away_avg_goals_conceded_last_3": "opp_avg_goals_conceded_last_3",
        "away_avg_goals_conceded_last_5": "opp_avg_goals_conceded_last_5",
        "away_avg_goals_conceded_last_10": "opp_avg_goals_conceded_last_10",
        "away_avg_points_last_3": "opp_avg_points_last_3",
        "away_avg_points_last_5": "opp_avg_points_last_5",
        "away_avg_points_last_10": "opp_avg_points_last_10",
        "away_avg_goal_diff_last_3": "opp_avg_goal_diff_last_3",
        "away_avg_goal_diff_last_5": "opp_avg_goal_diff_last_5",
        "away_avg_goal_diff_last_10": "opp_avg_goal_diff_last_10",
        "avg_home_goals_scored_last_5": "venue_avg_goals_scored_last_5",
        "avg_home_goals_conceded_last_5": "venue_avg_goals_conceded_last_5",
        "avg_home_points_last_5": "venue_avg_points_last_5",
    })
    home_df["is_home"] = 1

    away_df = df[[
        "date",
        "away_team",
        "home_team",
        "away_goals",
        "away_avg_goals_scored_last_3",
        "away_avg_goals_scored_last_5",
        "away_avg_goals_scored_last_10",
        "away_avg_goals_conceded_last_3",
        "away_avg_goals_conceded_last_5",
        "away_avg_goals_conceded_last_10",
        "away_avg_points_last_3",
        "away_avg_points_last_5",
        "away_avg_points_last_10",
        "away_avg_goal_diff_last_3",
        "away_avg_goal_diff_last_5",
        "away_avg_goal_diff_last_10",
        "home_avg_goals_scored_last_3",
        "home_avg_goals_scored_last_5",
        "home_avg_goals_scored_last_10",
        "home_avg_goals_conceded_last_3",
        "home_avg_goals_conceded_last_5",
        "home_avg_goals_conceded_last_10",
        "home_avg_points_last_3",
        "home_avg_points_last_5",
        "home_avg_points_last_10",
        "home_avg_goal_diff_last_3",
        "home_avg_goal_diff_last_5",
        "home_avg_goal_diff_last_10",
        "avg_away_goals_scored_last_5",
        "avg_away_goals_conceded_last_5",
        "avg_away_points_last_5",
    ]].copy()

    away_df = away_df.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_goals": "goals",
        "away_avg_goals_scored_last_3": "team_avg_goals_scored_last_3",
        "away_avg_goals_scored_last_5": "team_avg_goals_scored_last_5",
        "away_avg_goals_scored_last_10": "team_avg_goals_scored_last_10",
        "away_avg_goals_conceded_last_3": "team_avg_goals_conceded_last_3",
        "away_avg_goals_conceded_last_5": "team_avg_goals_conceded_last_5",
        "away_avg_goals_conceded_last_10": "team_avg_goals_conceded_last_10",
        "away_avg_points_last_3": "team_avg_points_last_3",
        "away_avg_points_last_5": "team_avg_points_last_5",
        "away_avg_points_last_10": "team_avg_points_last_10",
        "away_avg_goal_diff_last_3": "team_avg_goal_diff_last_3",
        "away_avg_goal_diff_last_5": "team_avg_goal_diff_last_5",
        "away_avg_goal_diff_last_10": "team_avg_goal_diff_last_10",
        "home_avg_goals_scored_last_3": "opp_avg_goals_scored_last_3",
        "home_avg_goals_scored_last_5": "opp_avg_goals_scored_last_5",
        "home_avg_goals_scored_last_10": "opp_avg_goals_scored_last_10",
        "home_avg_goals_conceded_last_3": "opp_avg_goals_conceded_last_3",
        "home_avg_goals_conceded_last_5": "opp_avg_goals_conceded_last_5",
        "home_avg_goals_conceded_last_10": "opp_avg_goals_conceded_last_10",
        "home_avg_points_last_3": "opp_avg_points_last_3",
        "home_avg_points_last_5": "opp_avg_points_last_5",
        "home_avg_points_last_10": "opp_avg_points_last_10",
        "home_avg_goal_diff_last_3": "opp_avg_goal_diff_last_3",
        "home_avg_goal_diff_last_5": "opp_avg_goal_diff_last_5",
        "home_avg_goal_diff_last_10": "opp_avg_goal_diff_last_10",
        "avg_away_goals_scored_last_5": "venue_avg_goals_scored_last_5",
        "avg_away_goals_conceded_last_5": "venue_avg_goals_conceded_last_5",
        "avg_away_points_last_5": "venue_avg_points_last_5",
    })
    away_df["is_home"] = 0

    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df = team_df.sort_values("date").reset_index(drop=True)

    return team_df