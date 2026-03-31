import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from football_prediction.data_loader import load_and_merge_data
from football_prediction.preprocessing import clean_match_data
from football_prediction.features import create_features, build_team_level_dataset
from football_prediction.model import train_poisson_model, predict_match_full2, poisson_pmf


def save_visualizations(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Balkendiagramm: Heimsieg / Unentschieden / Auswärtssieg
    labels = ["Heimsieg", "Unentschieden", "Auswärtssieg"]
    values = [
        result["home_win_probability"],
        result["draw_probability"],
        result["away_win_probability"],
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("Wahrscheinlichkeit")
    plt.ylim(0, 1)
    plt.title(f"Ergebniswahrscheinlichkeiten: {result['home_team']} vs {result['away_team']}")
    plt.savefig(output_dir / "outcome_probabilities.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Heatmap exakter Ergebnisse
    max_goals_plot = 5
    score_matrix = np.zeros((max_goals_plot + 1, max_goals_plot + 1))

    for (home_goals, away_goals), prob in result["score_probabilities"].items():
        if home_goals <= max_goals_plot and away_goals <= max_goals_plot:
            score_matrix[home_goals, away_goals] = prob

    plt.figure(figsize=(8, 6))
    plt.imshow(score_matrix, origin="lower", aspect="auto")
    plt.colorbar(label="Wahrscheinlichkeit")
    plt.xticks(range(max_goals_plot + 1))
    plt.yticks(range(max_goals_plot + 1))
    plt.xlabel(f"Tore {result['away_team']}")
    plt.ylabel(f"Tore {result['home_team']}")
    plt.title(f"Heatmap exakter Ergebnisse: {result['home_team']} vs {result['away_team']}")
    plt.savefig(output_dir / "score_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Verteilung Heimtore
    home_lambda = result["predicted_home_goals"]
    goals = list(range(0, 8))
    home_probs = [poisson_pmf(k, home_lambda) for k in goals]

    plt.figure(figsize=(8, 5))
    plt.bar(goals, home_probs)
    plt.xlabel(f"Tore {result['home_team']}")
    plt.ylabel("Wahrscheinlichkeit")
    plt.title(f"Poisson-Verteilung Heimtore: {result['home_team']}")
    plt.xticks(goals)
    plt.savefig(output_dir / "home_goal_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Verteilung Auswärtstore
    away_lambda = result["predicted_away_goals"]
    away_probs = [poisson_pmf(k, away_lambda) for k in goals]

    plt.figure(figsize=(8, 5))
    plt.bar(goals, away_probs)
    plt.xlabel(f"Tore {result['away_team']}")
    plt.ylabel("Wahrscheinlichkeit")
    plt.title(f"Poisson-Verteilung Auswärtstore: {result['away_team']}")
    plt.xticks(goals)
    plt.savefig(output_dir / "away_goal_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    if len(sys.argv) != 3:
        print("Verwendung: python -m football_prediction <Heimteam> <Auswärtsteam>")
        print("Beispiel: python -m football_prediction Arsenal Chelsea")
        return

    home_team = sys.argv[1]
    away_team = sys.argv[2]

    print("Starte football_prediction ...")
    print(f"Gewähltes Spiel: {home_team} vs {away_team}")

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw"
    output_dir = project_root / "outputs"

    raw_data = load_and_merge_data(str(data_path))
    clean_data = clean_match_data(raw_data)
    featured_data = create_features(clean_data)
    team_model_df = build_team_level_dataset(featured_data)

    home_df = team_model_df[team_model_df["is_home"] == 1]
    away_df = team_model_df[team_model_df["is_home"] == 0]

    home_model, _, _ = train_poisson_model(home_df)
    away_model, _, _ = train_poisson_model(away_df)

    try:
        result = predict_match_full2(
            home_model=home_model,
            away_model=away_model,
            team_model_df=team_model_df,
            home_team=home_team,
            away_team=away_team,
        )
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {e}")
        return

    print()
    print(f"{result['home_team']} vs {result['away_team']}")
    print(
        f"Erwartete Tore: "
        f"{result['predicted_home_goals']:.2f} : {result['predicted_away_goals']:.2f}"
    )
    print(
        f"Gerundetes Ergebnis: "
        f"{result['rounded_home_goals']} : {result['rounded_away_goals']}"
    )
    print(f"Sieger-Tipp: {result['winner']}")
    print()
    print(f"Heimsieg: {result['home_win_probability']:.2%}")
    print(f"Unentschieden: {result['draw_probability']:.2%}")
    print(f"Auswärtssieg: {result['away_win_probability']:.2%}")

    save_visualizations(result, output_dir)

    print()
    print("Grafiken wurden gespeichert in:")
    print(output_dir / "outcome_probabilities.png")
    print(output_dir / "score_heatmap.png")
    print(output_dir / "home_goal_distribution.png")
    print(output_dir / "away_goal_distribution.png")


if __name__ == "__main__":
    main()