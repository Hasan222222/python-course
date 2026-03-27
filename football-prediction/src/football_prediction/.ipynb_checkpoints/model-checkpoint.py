import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def prepare_model_data(df: pd.DataFrame):
    df = df.copy()

    feature_cols = [
        "home_avg_goals_scored_last_5",
        "away_avg_goals_scored_last_5",
        "home_avg_points_last_5",
        "away_avg_points_last_5",
    ]

    X = df[feature_cols]

    y = df["result"].map({
        "H": 0,
        "D": 1,
        "A": 2,
    })

    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report