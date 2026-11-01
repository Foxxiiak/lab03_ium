"""Laboratorium 3: samodzielne trenowanie i zapis modelu."""

import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import load_data, split_data

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


def project_path(value):
    """Zamien sciezke z .env na bezwzgledna wzgledem projektu."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def main():
    # Wszystkie sciezki i hiperparametry pochodza z .env.
    data_path = project_path(os.environ["DATA_PATH"])
    model_path = project_path(os.environ["MODEL_PATH"])
    test_size = float(os.environ["TEST_SIZE"])
    seed = int(os.environ["RANDOM_SEED"])
    model_type = os.environ["MODEL_TYPE"]
    max_iter = int(os.environ["MAX_ITER"])
    c_value = float(os.environ["REGULARIZATION_C"])
    solver = os.environ["SOLVER"]

    if model_type != "logistic_regression":
        raise ValueError(f"Nieobslugiwany typ modelu: {model_type}")

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df, test_size, seed)

    # Pipeline zapamietuje skaler i wytrenowany klasyfikator.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c_value, max_iter=max_iter, solver=solver, random_state=seed
        )),
    ])
    model.fit(X_train, y_train)

    # Zapisujemy model, a nie dane testowe - ewaluacja odtworzy podzial.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Dane: {len(df)} obserwacji, {X_train.shape[1]} cech")
    print(f"Trening: {len(X_train)} | Test: {len(X_test)}")
    print(f"Model zapisano do: {model_path}")


if __name__ == "__main__":
    main()
